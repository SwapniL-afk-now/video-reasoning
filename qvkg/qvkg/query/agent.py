from __future__ import annotations

"""Agentic QA: VLM-driven iterative graph search with live frame escalation.

The VLM receives tools to search the knowledge graph and extract video frames.
It decides what to look up, when to drill deeper, and when to escalate to raw
video frames — no heuristic intent classification or fixed BFS traversal.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..schema import AnswerResult, VKGraph
from ..vllm_client import MCQ_REASONING_SAMPLING, QA_SAMPLING, extract_mcq_answer
from .frame_extractor import extract_frames_for_window, frames_to_b64_urls

# ---------------------------------------------------------------------------
# Tool definitions (passed to llm.chat via the tools= param)
# ---------------------------------------------------------------------------

GRAPH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_graph",
            "description": (
                "Semantic search over the video knowledge graph. Returns matching nodes "
                "with timestamps, types, and labels. "
                "node_types filters results — options: SpeechNode, ClipNode, ObjectNode, "
                "ActionNode, OCRNode, CharacterNode, SceneNode, StateChangeNode. "
                "t_start and t_end restrict to a time window (seconds)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":      {"type": "string", "description": "Text to search for"},
                    "node_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of node types to filter by",
                    },
                    "t_start": {"type": "number", "description": "Window start in seconds"},
                    "t_end":   {"type": "number", "description": "Window end in seconds"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_detail",
            "description": (
                "Get all nodes within a time window: speech, objects, actions, OCR text, "
                "state changes. Use after identifying a relevant region via list_episodes "
                "or search_graph to get fine-grained detail."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "t_start": {"type": "number", "description": "Window start in seconds"},
                    "t_end":   {"type": "number", "description": "Window end in seconds"},
                },
                "required": ["t_start", "t_end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_frames",
            "description": (
                "Extract actual video frames from a time window. Use when the graph "
                "lacks enough visual detail — e.g. to read exact on-screen text, "
                "identify a specific object, or verify a visual claim. "
                "Returns real frames from the raw video."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "t_start":    {"type": "number", "description": "Start time in seconds"},
                    "t_end":      {"type": "number", "description": "End time in seconds"},
                    "max_frames": {"type": "integer", "description": "Max frames to extract (default 4)"},
                },
                "required": ["t_start", "t_end"],
            },
        },
    },
]

AGENT_SYSTEM_PROMPT = """\
You are a video QA agent with access to a pre-built knowledge graph and the raw video.

The video episode structure is already provided in your context above.

Choose your first tool call based on the nature of the question:

If the question asks what is visually shown AT a specific timestamp
(e.g. "what text/price/name appears at 14:16", "what is on screen at this moment",
"what object is shown", "what is the person doing at X:XX"):
  → Call extract_frames(t_start, t_end) FIRST using the provided time reference.
  → Then use search_graph or get_scene_detail for additional context if needed.

If the question asks about events, sequences, ordering, or narrative
(e.g. "what is the last X introduced", "what happens during", "why does X occur",
"what are they looking for", "what do they find", "summarize"):
  → Call search_graph(query) to find relevant nodes across the video.
  → Call get_scene_detail(t_start, t_end) to drill into promising regions.
  → Call extract_frames only if you need to visually verify a specific moment.

Available tools:
- search_graph(query, node_types, t_start, t_end): semantic + keyword search over graph nodes
- get_scene_detail(t_start, t_end): all nodes (speech, objects, OCR, actions) in a time window
- extract_frames(t_start, t_end, max_frames): real video frames from the raw video

For multiple-choice questions, your FINAL answer must be exactly one letter: A, B, C, or D.\
"""


# ---------------------------------------------------------------------------
# Tool call parser
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parse Qwen3.5 XML tool call format from raw model output.

    Format:
        <tool_call>
        <function=name>
        <parameter=key>value</parameter>
        </function>
        </tool_call>
    """
    calls = []
    for block in re.finditer(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        inner = block.group(1).strip()
        fn_m = re.match(r"<function=(\w+)>(.*?)</function>", inner, re.DOTALL)
        if not fn_m:
            continue
        fn_name = fn_m.group(1)
        params_text = fn_m.group(2)
        args: Dict[str, Any] = {}
        for pm in re.finditer(r"<parameter=(\w+)>(.*?)</parameter>", params_text, re.DOTALL):
            key = pm.group(1)
            raw = pm.group(2).strip()
            # Parse typed values
            if raw.startswith("[") or raw.startswith("{"):
                try:
                    import json
                    raw = json.loads(raw)
                except Exception:
                    pass
            elif raw.replace(".", "", 1).lstrip("-").isdigit():
                raw = float(raw) if "." in raw else int(raw)
            args[key] = raw
        calls.append({"name": fn_name, "arguments": args})
    return calls


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class GraphToolExecutor:
    """Executes graph tool calls and returns (text_result, frame_urls)."""

    MAX_NODES_RETURNED = 30

    def __init__(
        self,
        graph: VKGraph,
        faiss_index: Optional[faiss.Index],
        siglip_encoder,
        video_path: Optional[str] = None,
    ):
        self.graph = graph
        self.index = faiss_index
        self.siglip = siglip_encoder
        self.video_path = video_path

    def execute(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Tuple[str, List[str]]:
        """Returns (text_result, list_of_b64_frame_urls)."""
        try:
            if tool_name == "list_episodes":
                return self._list_episodes(), []
            elif tool_name == "search_graph":
                return self._search_graph(**args), []
            elif tool_name == "get_scene_detail":
                return self._get_scene_detail(
                    float(args["t_start"]), float(args["t_end"])
                ), []
            elif tool_name == "extract_frames":
                return self._extract_frames(
                    float(args["t_start"]),
                    float(args["t_end"]),
                    int(args.get("max_frames", 4)),
                )
            else:
                return f"Unknown tool: {tool_name}", []
        except Exception as e:
            return f"Tool error ({tool_name}): {e}", []

    def _list_episodes(self) -> str:
        eps = sorted(
            [n for n in self.graph.nodes.values() if n.node_type == "EpisodeNode"],
            key=lambda n: n.t_start,
        )
        if not eps:
            # Fall back to scene-level if no episodes
            eps = sorted(
                [n for n in self.graph.nodes.values() if n.node_type == "SceneNode"],
                key=lambda n: n.t_start,
            )[:20]
        lines = [f"[{n.t_start:.0f}s–{n.t_end:.0f}s] id={n.id}  {n.label}" for n in eps]
        return "Video episodes:\n" + "\n".join(lines)

    def _search_graph(
        self,
        query: str,
        node_types: Optional[Any] = None,
        t_start: Optional[float] = None,
        t_end:   Optional[float] = None,
    ) -> str:
        # Normalise node_types
        if isinstance(node_types, str):
            node_types = [t.strip() for t in node_types.replace(",", " ").split()]
        allowed = set(node_types) if node_types else None

        candidates = []

        if self.index is not None and self.siglip is not None:
            # Semantic FAISS search
            q_emb = self.siglip.encode_text([query]).astype(np.float32)
            faiss.normalize_L2(q_emb)
            k = min(50, len(self.graph.node_id_list))
            sims, idxs = self.index.search(q_emb, k)
            for i, sim in zip(idxs[0], sims[0]):
                if int(i) < 0 or float(sim) < 0.25:
                    continue
                nid = self.graph.node_id_list[int(i)]
                node = self.graph.nodes.get(nid)
                if node:
                    candidates.append((float(sim), node))

        # Keyword fallback — match individual words (≥4 chars) against label + description
        keywords = [w for w in re.findall(r'\b\w{4,}\b', query.lower())]
        if keywords:
            for node in self.graph.nodes.values():
                text = (node.label + ' ' + (getattr(node, 'description', '') or '')).lower()
                matched = sum(1 for kw in keywords if kw in text)
                if matched > 0:
                    score = 0.5 + 0.4 * matched / len(keywords)
                    candidates.append((score, node))

        # Deduplicate, filter, sort
        seen = set()
        results = []
        for score, node in sorted(candidates, key=lambda x: -x[0]):
            if node.id in seen:
                continue
            seen.add(node.id)
            if allowed and node.node_type not in allowed:
                continue
            if t_start is not None and node.t_end < t_start:
                continue
            if t_end is not None and node.t_start > t_end:
                continue
            results.append(node)
            if len(results) >= self.MAX_NODES_RETURNED:
                break

        if not results:
            return f"No nodes found matching '{query}'."

        lines = [
            f"[{n.t_start:.0f}s–{n.t_end:.0f}s] [{n.node_type}] id={n.id}  {n.label}"
            for n in sorted(results, key=lambda n: n.t_start)
        ]
        return f"Found {len(results)} nodes matching '{query}':\n" + "\n".join(lines)

    def _get_scene_detail(self, t_start: float, t_end: float) -> str:
        buffer = 5.0
        window_span = t_end - t_start + 2 * buffer
        nodes = [
            n for n in self.graph.nodes.values()
            if n.t_start <= t_end + buffer and n.t_end >= t_start - buffer
            and n.node_type not in {"EpisodeNode", "VideoNode"}
            # Exclude CharacterNodes that span the entire video (global entities)
            and not (n.node_type == "CharacterNode" and (n.t_end - n.t_start) > window_span * 10)
        ]
        nodes.sort(key=lambda n: n.t_start)
        nodes = nodes[:self.MAX_NODES_RETURNED]

        if not nodes:
            return f"No nodes found in [{t_start:.0f}s–{t_end:.0f}s]."

        lines = []
        for n in nodes:
            meta_bits = []
            if n.node_type == "SpeechNode":
                # include SPOKEN_BY
                for e in self.graph.get_edges(n.id):
                    if e.relation_type == "SPOKEN_BY":
                        ch = self.graph.nodes.get(e.target_id)
                        if ch:
                            meta_bits.append(f"speaker={ch.label}")
            elif n.node_type == "StateChangeNode":
                if n.prev_state or n.next_state:
                    meta_bits.append(f"{n.prev_state}→{n.next_state}")
            elif n.node_type == "OCRNode":
                sem = n.metadata.get("semantic_type", "")
                if sem:
                    meta_bits.append(f"type={sem}")
            meta_str = f"  ({', '.join(meta_bits)})" if meta_bits else ""
            lines.append(
                f"[{n.t_start:.0f}s–{n.t_end:.0f}s] [{n.node_type}] {n.label}{meta_str}"
            )

        return (
            f"Nodes in [{t_start:.0f}s–{t_end:.0f}s]:\n" + "\n".join(lines)
        )

    def _extract_frames(
        self, t_start: float, t_end: float, max_frames: int = 4
    ) -> Tuple[str, List[str]]:
        if not self.video_path:
            return "No video file available for frame extraction.", []

        frames = extract_frames_for_window(
            self.video_path, t_start, t_end, max_frames=max_frames
        )
        if not frames:
            return f"Could not extract frames from [{t_start:.0f}s–{t_end:.0f}s].", []

        urls = frames_to_b64_urls(frames)
        ts_labels = ", ".join(f"{f.timestamp:.1f}s" for f in frames)
        text = (
            f"Extracted {len(frames)} frames from [{t_start:.0f}s–{t_end:.0f}s]. "
            f"Frame timestamps: {ts_labels}. "
            f"The frames are attached above."
        )
        return text, urls


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def agent_answer_question(
    question: str,
    graph: VKGraph,
    faiss_index: Optional[faiss.Index],
    llm,
    siglip_encoder,
    video_path: Optional[str] = None,
    question_type: Optional[List[str]] = None,
    time_reference: Optional[str] = None,
    mcq: bool = True,
    max_turns: int = 6,
) -> AnswerResult:
    """Answer a question by letting the VLM iteratively search the graph.

    First turn is seeded with:
      - All episode summaries (video table of contents — always cheap)
      - Frames from the explicitly provided time_reference window, if any

    The VLM then calls tools freely to search deeper, drill into scenes,
    or extract more frames until it has enough evidence to answer.
    """
    executor = GraphToolExecutor(graph, faiss_index, siglip_encoder, video_path)
    sampling = MCQ_REASONING_SAMPLING if mcq else QA_SAMPLING

    # Seed: episode summaries (always — video table of contents, cheap)
    episodes_text = executor._list_episodes()

    # Build plain-text first user message
    qt_str = ", ".join(question_type) if question_type else ""
    header_parts = []
    if qt_str:
        header_parts.append(f"[Question type: {qt_str}]")
    if time_reference:
        header_parts.append(f"[Time reference: {time_reference}]")
    header = "\n".join(header_parts)

    user_content = (
        f"{episodes_text}\n\n"
        + (f"{header}\n\n" if header else "")
        + f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    tool_calls_made: List[str] = []
    frames_used: List[float] = []
    nodes_seen: int = 0

    for turn in range(max_turns):
        output = llm.chat(
            messages=[messages],
            sampling_params=sampling,
            tools=GRAPH_TOOLS,
            chat_template_kwargs={"enable_thinking": mcq},
            use_tqdm=False,
        )[0]

        text = output.outputs[0].text.strip()
        tool_calls = _parse_tool_calls(text)

        if not tool_calls:
            # No tool calls — model is done, extract answer
            answer = extract_mcq_answer(text) if mcq else text
            return AnswerResult(
                answer=answer,
                intents=tool_calls_made,
                subgraph_size=nodes_seen,
                keyframes_used=frames_used,
                evidence_nodes=[],
            )

        # Record assistant turn
        messages.append({"role": "assistant", "content": text})

        # Execute each tool call
        for tc in tool_calls:
            tool_calls_made.append(tc["name"])
            result_text, frame_urls = executor.execute(tc["name"], tc["arguments"])

            if frame_urls:
                # Multimodal tool result: images + text
                content: Any = []
                for url in frame_urls:
                    content.append({"type": "image_url", "image_url": {"url": url}})
                content.append({"type": "text", "text": result_text})
                # Track frame timestamps from extract_frames args
                if "t_start" in tc["arguments"]:
                    frames_used.append(float(tc["arguments"]["t_start"]))
            else:
                content = result_text

            messages.append({
                "role": "tool",
                "name": tc["name"],
                "content": content,
            })
            nodes_seen += result_text.count("\n")

    # Max turns hit — ask for a final answer explicitly
    messages.append({
        "role": "user",
        "content": "Based on your research so far, give your final answer. Output exactly one letter: A, B, C, or D.",
    })
    output = llm.chat(
        messages=[messages],
        sampling_params=sampling,
        tools=GRAPH_TOOLS,
        chat_template_kwargs={"enable_thinking": mcq},
        use_tqdm=False,
    )[0]
    text = output.outputs[0].text.strip()
    answer = extract_mcq_answer(text) if mcq else text

    return AnswerResult(
        answer=answer,
        intents=tool_calls_made,
        subgraph_size=nodes_seen,
        keyframes_used=frames_used,
        evidence_nodes=[],
    )
