from __future__ import annotations

"""Typed action space + deterministic executor for the inference-time walker.

The controller LLM emits exactly one action as schema-constrained JSON
(:data:`ACTION_SCHEMA`); :func:`execute_action` then applies it deterministically
to the :class:`WalkState`. No model output is ever trusted to mutate the graph
directly — every move is one of six bounded, typed operations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..schema import VKGNode, VKGraph
from . import graph_ops
from .frame_extractor import extract_frames_for_window, frames_to_b64_urls

MAX_IMAGE_FRAMES_PER_PROMPT: int = 10  # Qwen's 10-image-per-turn limit

# ---------------------------------------------------------------------------
# Action JSON schema (constrained decode)
# ---------------------------------------------------------------------------

ACTION_NAMES = ["EXPAND", "ZOOM", "DISCRIMINATE", "RECALL", "ANSWER", "STOP_REQUEST", "BUILD"]

ACTION_SCHEMA = {
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {"type": "string", "enum": ACTION_NAMES,
               "description": "BUILD: online KG construction for the window where evidence is missing. No params."},
        "relation": {
            "type": "string",
            "enum": list(graph_ops.VALID_RELATIONS),
            "description": "For EXPAND: which edge family to follow.",
        },
        "node_id": {
            "type": "string",
            "description": "For ZOOM: the node id to materialise fine evidence at.",
        },
        "option": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
            "description": "For DISCRIMINATE: which MCQ option to separate from rivals.",
        },
        "query": {
            "type": "string",
            "description": "For RECALL: a semantic search query.",
        },
        "letter": {
            "type": "string",
            "enum": ["A", "B", "C", "D"],
            "description": "For ANSWER: the chosen option letter.",
        },
        "cited_node_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "For ANSWER: the node ids that support the answer.",
        },
    },
}


# ---------------------------------------------------------------------------
# Walker state
# ---------------------------------------------------------------------------

@dataclass
class WalkState:
    qid: str
    question: str
    options: Dict[str, str]                  # {"A": text, ...}
    qtype: List[str]
    time_reference: Optional[Tuple[float, float]] = None

    node_ids: Set[str] = field(default_factory=set)
    frontier: Set[str] = field(default_factory=set)
    last_ring: Set[str] = field(default_factory=set)   # Δnodes from most recent action

    hop: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    # frames materialised by ZOOM / seeding: list of (b64_url, timestamp)
    frames: List[Tuple[str, float]] = field(default_factory=list)
    _frame_ts_buckets: Set[int] = field(default_factory=set)

    # text summaries of frames evicted from the image slot — preserves
    # analysis of older frame batches once they exceed MAX_IMAGE_FRAMES_PER_PROMPT.
    frame_archive: List[str] = field(default_factory=list)

    current_answer: Optional[str] = None
    prev_answer: Optional[str] = None        # previous hop's read-out (elasticity fallback)
    last_action: Optional[Dict[str, Any]] = None  # resolved action this hop (debug)
    cited_node_ids: List[str] = field(default_factory=list)
    discriminated: Set[str] = field(default_factory=set)  # MCQ options already probed

    warrant: Any = None                      # query.warrant.Warrant
    forced_action: Optional[Dict[str, Any]] = None  # backtrack target
    stuck: bool = False               # true when last EXPAND added no new nodes
    build_radius_multiplier: int = 1  # doubles each BUILD call
    build_density: int = 1            # doubles each BUILD call
    done: bool = False
    final_answer: Optional[str] = None

    MAX_NODES: int = 80

    # -- helpers --------------------------------------------------------

    def add_frames(self, urls_ts: List[Tuple[str, float]]) -> None:
        """Add frames, deduplicated by ~10s timestamp bucket. No hard cap."""
        for url, ts in urls_ts:
            bucket = int(ts // 10)
            if bucket in self._frame_ts_buckets:
                continue
            self._frame_ts_buckets.add(bucket)
            self.frames.append((url, ts))

    def archive_old_frames(self, graph: VKGraph,
                           keep: int = MAX_IMAGE_FRAMES_PER_PROMPT) -> None:
        """Archive frames beyond the latest ``keep``, converting to text summaries.

        Archived frames are removed from ``self.frames``; their text description
        (sourced from VKG metadata at that timestamp) is appended to
        ``self.frame_archive`` for inclusion in future answerer prompts.
        """
        if len(self.frames) <= keep:
            return
        old = self.frames[:-keep]
        self.frames = self.frames[-keep:]
        from .walker import frame_to_text_summary
        for _, ts in old:
            summary = frame_to_text_summary(graph, ts)
            self.frame_archive.append(f"[t={ts:.0f}s] {summary}")

    def absorb(self, new_ids: Set[str]) -> Set[str]:
        """Merge ``new_ids`` into node_ids; return the genuinely new ring."""
        ring = {nid for nid in new_ids if nid not in self.node_ids}
        # Budget guard — keep the most recent ring even if over budget, but cap.
        self.node_ids |= ring
        self.last_ring = ring
        return ring


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def execute_action(
    action: Dict[str, Any],
    state: WalkState,
    graph: VKGraph,
    faiss_index,
    siglip_encoder,
    frame_store,
    video_path: Optional[str],
    k: int = 5,
) -> WalkState:
    """Apply one typed action to ``state`` in place and return it.

    ANSWER / STOP_REQUEST are control actions handled by the walker; here they
    are no-ops on the graph. EXPAND / ZOOM / DISCRIMINATE / RECALL each grow the
    working subgraph and set ``state.last_ring`` to the nodes they added.
    """
    name = (action.get("action") or "").upper()
    state.last_ring = set()
    state.stuck = False

    if name == "EXPAND":
        relation = action.get("relation") or "causal"
        new_ids, _edges = graph_ops.expand(graph, state.frontier or state.node_ids,
                                            relation, k=k)
        ring = state.absorb(new_ids)
        # Newly reached nodes become the next frontier.
        state.frontier = ring or state.frontier
        state.stuck = not bool(ring)

    elif name == "ZOOM":
        node_id = action.get("node_id")
        node = graph.nodes.get(node_id) if node_id else None
        if node is not None:
            _zoom_node(state, graph, node, frame_store, video_path)

    elif name == "DISCRIMINATE":
        opt = action.get("option")
        _discriminate(state, graph, faiss_index, siglip_encoder, opt, k=k)

    elif name == "RECALL":
        query = action.get("query") or state.question
        new_ids = set(graph_ops.faiss_search(graph, faiss_index, siglip_encoder,
                                              query, k=k * 2))
        ring = state.absorb(new_ids)
        if ring:
            state.frontier |= ring

    # ANSWER / STOP_REQUEST → no graph mutation here.
    return state


def _zoom_node(
    state: WalkState,
    graph: VKGraph,
    node: VKGNode,
    frame_store,
    video_path: Optional[str],
) -> None:
    """Materialise fine evidence at ``node``: frames + fine nodes in its window."""
    # Fine nodes overlapping the node's window.
    window = graph.get_nodes_in_window(node.t_start, node.t_end, buffer_sec=5.0)
    state.absorb({n.id for n in window} | {node.id})

    # Frames: prefer stored keyframe, fall back to live extraction.
    urls_ts: List[Tuple[str, float]] = []
    got_stored = False
    if node.keyframe_id and frame_store is not None:
        try:
            if node.keyframe_id in frame_store:
                url = frame_store.get_b64_url(node.keyframe_id)
                urls_ts.append((url, node.t_start))
                got_stored = True
        except Exception:
            got_stored = False

    if not got_stored and video_path:
        frames = extract_frames_for_window(video_path, node.t_start, node.t_end,
                                            max_frames=3)
        for url, fr in zip(frames_to_b64_urls(frames), frames):
            urls_ts.append((url, fr.timestamp))

    state.add_frames(urls_ts)


def _discriminate(
    state: WalkState,
    graph: VKGraph,
    faiss_index,
    siglip_encoder,
    option: Optional[str],
    k: int = 5,
) -> None:
    """Targeted retrieval separating MCQ option ``option`` from its rivals."""
    if not option or option not in state.options:
        return
    state.discriminated.add(option)
    opt_text = state.options[option].strip()
    if not opt_text:
        return

    query = f"{opt_text} {state.question}"
    new_ids: Set[str] = set(graph_ops.faiss_search(
        graph, faiss_index, siglip_encoder, query, k=k * 2))

    # Entity-introduction expansion: if the option names a known character/entity,
    # pull in that entity's appearances (its introduction carries identity info).
    opt_lower = opt_text.lower()
    for char in graph.get_nodes_by_type("CharacterNode"):
        label = (char.label or "").lower()
        desc = (char.canonical_description or "").lower()
        if opt_lower in label or label in opt_lower or opt_lower in desc:
            if char.entity_id:
                for n in graph_ops.trace_entity(graph, char.entity_id)[:5]:
                    new_ids.add(n.id)
            new_ids.add(char.id)
            break

    ring = state.absorb(new_ids)
    if ring:
        state.frontier |= ring
