from __future__ import annotations

"""ReAct (Reasoning + Acting) agent for video QA.

The model writes explicit Thought: / Action: steps in plain text. We parse the
last Action: line, execute it, and inject the result back as an Observation:
user message. The loop continues until the model calls answer(X) or max_turns
is reached.

No tools= parameter passed to vLLM. No XML parsing. Observations go as
role: "user" messages, not role: "tool".
"""

import re
from typing import List, Optional, Tuple

import faiss

from ..schema import AnswerResult, VKGraph
from ..vllm_client import MCQ_REASONING_SAMPLING, QA_SAMPLING, extract_mcq_answer
from .agent import GraphToolExecutor

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = """\
You are a video QA agent. Use Thought / Action format to reason and retrieve evidence.

GOAL: Answer the multiple-choice question with exactly one letter (A, B, C, or D) \
by finding evidence in the knowledge graph or video frames.

Available actions:
  search_graph("query")              — semantic search over the video knowledge graph
  get_scene_detail(t_start, t_end)   — all nodes (speech, objects, OCR, actions) in a time window (seconds)
  extract_frames(t_start, t_end)     — extract real video frames from the raw video file
  answer(A|B|C|D)                    — provide your final answer when you are confident

Format (follow exactly):
  Thought: <your reasoning about what evidence you need next>
  Action: <action_name(args)>

Rules:
- Write one Thought: line, then one Action: line, nothing else.
- For questions about what is shown AT a specific timestamp → call extract_frames FIRST.
- For questions about ordering, sequences, or narrative → call search_graph FIRST, then extract_frames to verify timestamps found.
- After each Observation, ask yourself: "Do I now have enough evidence to answer confidently?"
- Only call answer(X) when you are confident. If uncertain, search more.\
"""

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_react_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (action_name, args_raw) for the LAST Action: line in text.

    Takes the last match so the model can recite context without confusing us.
    Returns (None, None) if no Action: line found.
    """
    matches = list(re.finditer(r'Action:\s*(\w+)\(([^)]*)\)', text, re.IGNORECASE))
    if not matches:
        return None, None
    m = matches[-1]
    return m.group(1).lower(), m.group(2).strip()


def _parse_args(args_raw: str) -> List[str]:
    """Split comma-separated args, strip surrounding quotes."""
    return [p.strip().strip("\"'") for p in args_raw.split(",") if p.strip()]


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

def _execute_action(
    action: str, args_raw: str, executor: GraphToolExecutor
) -> Tuple[str, List[str]]:
    """Execute a parsed action. Returns (obs_text, frame_url_list)."""
    try:
        parts = _parse_args(args_raw)
        if action == "search_graph":
            query = parts[0] if parts else args_raw.strip().strip("\"'")
            return executor._search_graph(query), []
        elif action == "get_scene_detail":
            if len(parts) < 2:
                return "get_scene_detail requires t_start and t_end arguments.", []
            t0, t1 = float(parts[0]), float(parts[1])
            return executor._get_scene_detail(t0, t1), []
        elif action == "extract_frames":
            if not parts:
                return "extract_frames requires t_start argument.", []
            t0 = float(parts[0])
            t1 = float(parts[1]) if len(parts) > 1 else t0 + 10.0
            max_f = int(parts[2]) if len(parts) > 2 else 6
            return executor._extract_frames(t0, t1, max_f)
        else:
            return f"Unknown action: '{action}'. Available: search_graph, get_scene_detail, extract_frames, answer.", []
    except Exception as e:
        return f"Action error ({action}): {e}", []


def _build_obs_content(obs_text: str, frame_urls: List[str]):
    """Build a user-message content value for the observation.

    Returns a list (multimodal) if frames are present, otherwise a plain string.
    """
    if frame_urls:
        content = [{"type": "image_url", "image_url": {"url": u}} for u in frame_urls]
        content.append({"type": "text", "text": f"Observation: {obs_text}"})
        return content
    return f"Observation: {obs_text}"


# ---------------------------------------------------------------------------
# Main ReAct loop
# ---------------------------------------------------------------------------

def react_answer_question(
    question: str,
    graph: VKGraph,
    faiss_index: Optional[faiss.Index],
    llm,
    siglip_encoder,
    video_path: Optional[str] = None,
    question_type: Optional[List[str]] = None,
    time_reference: Optional[str] = None,
    mcq: bool = True,
    max_turns: int = 8,
) -> AnswerResult:
    """Answer a question using a ReAct (Reason + Act) loop.

    Each turn:
      1. Model writes Thought: + Action:
      2. We parse the last Action: line and execute it
      3. We inject Observation: as a user message
      4. Repeat until model writes Action: answer(X) or max_turns exhausted
    """
    executor = GraphToolExecutor(graph, faiss_index, siglip_encoder, video_path)
    sampling = MCQ_REASONING_SAMPLING if mcq else QA_SAMPLING

    # Build seeded first user message
    qt_str = ", ".join(question_type) if question_type else ""
    header_parts: List[str] = []
    if qt_str:
        header_parts.append(f"[Question type: {qt_str}]")
    if time_reference:
        header_parts.append(f"[Time reference: {time_reference}]")

    user_msg = executor._list_episodes()
    if header_parts:
        user_msg += "\n\n" + "\n".join(header_parts)
    user_msg += f"\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    tool_calls_made: List[str] = []
    frames_used: List[str] = []
    nodes_seen: int = 0
    letter: str = "A"  # fallback
    trace: List[str] = []  # human-readable Thought/Action/Observation log

    for _turn in range(max_turns):
        output = llm.chat(
            messages=[messages],
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": mcq},
            use_tqdm=False,
        )[0]
        text = output.outputs[0].text.strip()
        messages.append({"role": "assistant", "content": text})
        trace.append(f"[Turn {_turn}] MODEL:\n{text}")

        action, args_raw = _parse_react_action(text)

        if action == "answer":
            # Extract the letter from answer(X)
            raw_letter = (args_raw or "").strip().upper()
            letter = raw_letter[:1] if raw_letter[:1] in "ABCD" else extract_mcq_answer(text)
            trace.append(f"[Turn {_turn}] ANSWER → {letter}")
            break

        if action is None:
            # Model didn't write an Action: line — treat output as final answer
            letter = extract_mcq_answer(text)
            trace.append(f"[Turn {_turn}] NO ACTION → extract_mcq_answer → {letter}")
            break

        # Execute the tool call
        tool_calls_made.append(action)
        obs_text, obs_frames = _execute_action(action, args_raw, executor)
        nodes_seen += obs_text.count("\n")
        if obs_frames:
            frames_used.append(args_raw)

        obs_summary = obs_text[:300] + ("..." if len(obs_text) > 300 else "")
        trace.append(
            f"[Turn {_turn}] ACTION: {action}({args_raw})\n"
            f"  OBS ({len(obs_frames)} frames): {obs_summary}"
        )

        messages.append({
            "role": "user",
            "content": _build_obs_content(obs_text, obs_frames),
        })

    else:
        # max_turns exhausted — force a final answer
        trace.append(f"[MaxTurns] forcing final answer")
        messages.append({
            "role": "user",
            "content": (
                "You have used all your turns. "
                "Based on all the evidence gathered, give your final answer. "
                "Output exactly one letter: A, B, C, or D."
            ),
        })
        output = llm.chat(
            messages=[messages],
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": mcq},
            use_tqdm=False,
        )[0]
        letter = extract_mcq_answer(output.outputs[0].text)
        trace.append(f"[MaxTurns] forced answer → {letter}")

    # Print trace to stderr for debugging
    import sys
    print("\n" + "="*60, file=sys.stderr)
    print(f"Q: {question[:100]}", file=sys.stderr)
    for t in trace:
        print(t, file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)

    return AnswerResult(
        answer=letter,
        intents=tool_calls_made,
        subgraph_size=nodes_seen,
        keyframes_used=frames_used,
        evidence_nodes=[],
    )
