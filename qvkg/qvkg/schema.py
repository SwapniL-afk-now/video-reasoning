from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from sortedcontainers import SortedList

# ---------------------------------------------------------------------------
# Edge taxonomy
# ---------------------------------------------------------------------------

EDGE_TYPES: Dict[str, str] = {
    # Temporal backbone
    "PRECEDES":          "a ends before b starts",
    "OVERLAPS":          "a and b overlap in time",
    "DURING":            "a is contained within b's time span",
    # Hierarchical backbone
    "CONTAINS":          "parent contains child (episode→scene, scene→clip)",
    "INSTANCE_OF":       "event is an instance within a scene",
    # Entity continuity
    "SAME_ENTITY":       "same person/object at different times",
    "PERFORMS":          "character performs action",
    "INTERACTS_WITH":    "two entities interact",
    "LOCATED_IN":        "entity appears in scene",
    # Spatial
    "LEFT_OF":           "spatial: left",
    "RIGHT_OF":          "spatial: right",
    "ABOVE":             "spatial: above",
    "BELOW":             "spatial: below",
    "IN_FRONT_OF":       "spatial: closer to camera",
    "BEHIND":            "spatial: further from camera",
    "NEAR":              "spatial: in close proximity",
    "CONTAINS_SPATIAL":  "bounding box containment",
    # Causal
    "CAUSES":            "a directly causes b",
    "ENABLES":           "a creates condition for b",
    "PREVENTS":          "a prevents b from occurring",
    "MOTIVATES":         "a is character's motivation for b",
    # Semantic
    "SIMILAR_TO":        "high cosine similarity (FAISS)",
    "CONTRADICTS":       "conflicting observations",
    # Cross-modal
    "DESCRIBES":         "speech describes concurrent visual",
    "MENTIONS":          "speech mentions a visible entity",
    "LABELS":            "OCR text labels a visible object",
    "ACCOMPANIES":       "audio event tied to visual action",
}

CAUSAL_EDGE_TYPES: Set[str] = {"CAUSES", "ENABLES", "PREVENTS", "MOTIVATES"}

# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VKGNode:
    id:           str
    node_type:    str
    label:        str
    level:        int        # 0=clip, 1=scene, 2=episode, 3=video

    t_start:      float
    t_end:        float

    keyframe_id:  Optional[str]         = None
    bbox:         Optional[List[float]] = None   # [x1,y1,x2,y2] normalised
    confidence:   float                 = 1.0

    siglip_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    faiss_idx:        Optional[int]        = None

    entity_id:            Optional[str] = None
    canonical_description: Optional[str] = None

    prev_state:   Optional[str] = None
    next_state:   Optional[str] = None

    parent_id:    Optional[str] = None

    # Confidence-aware fields (TODO 1)
    temporal_precision:   float = 0.0   # seconds to nearest real sampled keyframe
    is_question_seeded:   bool  = False  # densely sampled for a benchmark question

    metadata:     Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if k != "siglip_embedding"}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VKGNode":
        d = dict(d)
        d.pop("siglip_embedding", None)
        return cls(**d)


@dataclass
class VKGEdge:
    source_id:     str
    target_id:     str
    relation_type: str
    weight:        float
    confidence:    float
    metadata:      Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "VKGEdge":
        return cls(**d)


# ---------------------------------------------------------------------------
# Helper dataclasses returned by sampler
# ---------------------------------------------------------------------------

@dataclass
class FrameInfo:
    id:        str
    timestamp: float
    path:      Optional[str] = None
    image:     Optional[object] = field(default=None, repr=False)  # PIL Image


@dataclass
class Scene:
    id:         str
    t_start:    float
    t_end:      float
    keyframes:  List[FrameInfo] = field(default_factory=list)
    scene_label: str            = ""


@dataclass
class Episode:
    id:              str
    label:           str
    t_start:         float
    t_end:           float
    narrative_role:  str
    summary:         str        = ""
    scenes:          List[Scene] = field(default_factory=list)

    def get_representative_frames(self, max_frames: int = 8) -> List[FrameInfo]:
        all_frames = [f for sc in self.scenes for f in sc.keyframes]
        if len(all_frames) <= max_frames:
            return all_frames
        step = len(all_frames) / max_frames
        return [all_frames[int(i * step)] for i in range(max_frames)]


@dataclass
class SampleResult:
    keyframes:         List[FrameInfo]
    scenes:            List[Scene]
    episodes:          List[Episode]
    siglip_embeddings: Optional[np.ndarray] = None


@dataclass
class AnswerResult:
    answer:         str
    intents:        List[str]
    subgraph_size:  int
    keyframes_used: List[float]
    evidence_nodes: List[str]


# ---------------------------------------------------------------------------
# SubGraph — a view over VKGraph
# ---------------------------------------------------------------------------

@dataclass
class CausalChain:
    source:    VKGNode
    target:    VKGNode
    relation:  str
    confidence: float
    metadata:  Dict = field(default_factory=dict)


class SubGraph:
    def __init__(self, nodes: Dict[str, VKGNode], edges: List[VKGEdge]):
        self.nodes = nodes
        self.edges = edges

    def get_sorted_events(self) -> List[VKGNode]:
        return sorted(self.nodes.values(), key=lambda n: n.t_start)

    def get_causal_chains(self) -> List[CausalChain]:
        chains = []
        for e in self.edges:
            if e.relation_type in CAUSAL_EDGE_TYPES:
                if e.source_id in self.nodes and e.target_id in self.nodes:
                    chains.append(CausalChain(
                        source=self.nodes[e.source_id],
                        target=self.nodes[e.target_id],
                        relation=e.relation_type,
                        confidence=e.confidence,
                        metadata=e.metadata,
                    ))
        return sorted(chains, key=lambda c: c.source.t_start)

    def get_characters(self) -> List[VKGNode]:
        return [n for n in self.nodes.values() if n.node_type == "CharacterNode"]

    def get_state_changes(self) -> List[VKGNode]:
        return sorted(
            [n for n in self.nodes.values() if n.node_type == "StateChangeNode"],
            key=lambda n: n.t_start,
        )

    def get_spatial_relations(self) -> List[VKGEdge]:
        spatial = {"LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW",
                   "IN_FRONT_OF", "BEHIND", "NEAR", "CONTAINS_SPATIAL"}
        return [e for e in self.edges if e.relation_type in spatial]

    def get_speech_nodes(self) -> List[VKGNode]:
        return sorted(
            [n for n in self.nodes.values() if n.node_type == "SpeechNode"],
            key=lambda n: n.t_start,
        )

    def get_visual_nodes(self) -> List[VKGNode]:
        return [n for n in self.nodes.values() if n.keyframe_id]


# ---------------------------------------------------------------------------
# VKGraph — main graph container
# ---------------------------------------------------------------------------

class VKGraph:
    def __init__(self):
        self.nodes:       Dict[str, VKGNode]       = {}
        self.edges:       Dict[str, List[VKGEdge]] = {}   # outgoing adjacency
        self.edges_in:    Dict[str, List[VKGEdge]] = {}   # incoming adjacency
        self.temporal_idx: SortedList              = SortedList(key=lambda n: n.t_start)
        self.entity_idx:  Dict[str, List[str]]    = {}   # entity_id → [node_ids]
        self.type_idx:    Dict[str, List[str]]    = {}   # node_type → [node_ids]
        self.node_id_list: List[str]              = []   # FAISS position → node_id

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(self, node: VKGNode) -> None:
        self.nodes[node.id] = node
        self.edges.setdefault(node.id, [])
        self.edges_in.setdefault(node.id, [])
        self.temporal_idx.add(node)
        if node.entity_id:
            self.entity_idx.setdefault(node.entity_id, []).append(node.id)
        self.type_idx.setdefault(node.node_type, []).append(node.id)

    def add_nodes(self, nodes: List[VKGNode]) -> None:
        for n in nodes:
            self.add_node(n)

    def add_edge(self, edge: VKGEdge) -> None:
        self.edges.setdefault(edge.source_id, []).append(edge)
        self.edges_in.setdefault(edge.target_id, []).append(edge)

    def add_edges(self, edges: List[VKGEdge]) -> None:
        for e in edges:
            self.add_edge(e)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[VKGNode]:
        return self.nodes.get(node_id)

    def get_edges(self, node_id: str) -> List[VKGEdge]:
        return self.edges.get(node_id, [])

    def get_incoming_edges(self, node_id: str) -> List[VKGEdge]:
        return self.edges_in.get(node_id, [])

    def get_neighbor(self, node: VKGNode, relation: str) -> Optional[VKGNode]:
        reverse = relation.endswith("_REV")
        rel = relation[:-4] if reverse else relation
        edge_list = self.edges_in[node.id] if reverse else self.edges.get(node.id, [])
        for e in edge_list:
            if e.relation_type == rel:
                nid = e.source_id if reverse else e.target_id
                return self.nodes.get(nid)
        return None

    def get_nodes_by_type(self, node_type: str) -> List[VKGNode]:
        return [self.nodes[nid] for nid in self.type_idx.get(node_type, [])
                if nid in self.nodes]

    def get_episodes(self) -> List[VKGNode]:
        return sorted(self.get_nodes_by_type("EpisodeNode"), key=lambda n: n.t_start)

    def get_nodes_in_window(
        self,
        t_start: float,
        t_end: float,
        buffer_sec: float = 30.0,
    ) -> List[VKGNode]:
        """O(log N) temporal range query — returns all nodes overlapping [t_start-buffer, t_end+buffer]."""
        lo = t_start - buffer_sec
        hi = t_end   + buffer_sec
        # SortedList is keyed on t_start; find nodes whose t_start <= hi
        # then filter by t_end >= lo for overlap
        result = []
        # bisect to the first node with t_start > hi
        idx = self.temporal_idx.bisect_right(
            _SentinelNode(hi)
        )
        for node in self.temporal_idx[:idx]:
            if node.t_end >= lo:
                result.append(node)
        return result

    def compute_temporal_precision(self) -> None:
        """Compute temporal_precision for every non-ClipNode based on nearest ClipNode."""
        clip_times = sorted(
            n.t_start for n in self.get_nodes_by_type("ClipNode")
        )
        if not clip_times:
            for node in self.nodes.values():
                if node.node_type != "ClipNode":
                    node.temporal_precision = 999.0
            return
        import bisect
        for node in self.nodes.values():
            if node.node_type == "ClipNode":
                node.temporal_precision = 0.0
                continue
            t = node.t_start
            pos = bisect.bisect_left(clip_times, t)
            candidates = []
            if pos < len(clip_times):
                candidates.append(abs(clip_times[pos] - t))
            if pos > 0:
                candidates.append(abs(clip_times[pos - 1] - t))
            node.temporal_precision = min(candidates) if candidates else 999.0

    def get_children(self, node: VKGNode, depth: int = 1) -> List[VKGNode]:
        result = []
        queue = [(node, 0)]
        while queue:
            curr, d = queue.pop(0)
            if d >= depth:
                continue
            for e in self.get_edges(curr.id):
                if e.relation_type == "CONTAINS":
                    child = self.nodes.get(e.target_id)
                    if child:
                        result.append(child)
                        queue.append((child, d + 1))
        return result

    def get_events_in_episode(self, episode: "Episode") -> List[VKGNode]:
        event_types = {"ActionNode", "InteractionNode", "StateChangeNode",
                       "SpeechNode", "OCRNode", "AudioEventNode"}
        return sorted(
            [n for n in self.nodes.values()
             if n.node_type in event_types
             and episode.t_start <= n.t_start <= episode.t_end],
            key=lambda n: n.t_start,
        )

    def find_event_by_description(
        self, desc: str, candidates: List[VKGNode]
    ) -> Optional[VKGNode]:
        desc_l = desc.lower()
        for n in candidates:
            if desc_l in n.label.lower() or n.label.lower() in desc_l:
                return n
        return None

    def find_entity_in_scene(
        self, entity_label: str, scene_id: str
    ) -> Optional[VKGNode]:
        label_l = entity_label.lower()
        for n in self.nodes.values():
            if n.parent_id == scene_id and label_l in n.label.lower():
                return n
        return None

    def get_all_character_mentions(self) -> List[VKGNode]:
        return self.get_nodes_by_type("CharacterNode")

    # ------------------------------------------------------------------
    # Subgraph extraction
    # ------------------------------------------------------------------

    def induced_subgraph(self, node_ids: Set[str]) -> SubGraph:
        sub_nodes = {nid: self.nodes[nid] for nid in node_ids if nid in self.nodes}
        sub_edges = [
            e for nid in node_ids
            for e in self.get_edges(nid)
            if e.target_id in node_ids
        ]
        return SubGraph(sub_nodes, sub_edges)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        all_edges: List[VKGEdge] = []
        for elist in self.edges.values():
            all_edges.extend(elist)
        # Deduplicate
        seen = set()
        unique_edges = []
        for e in all_edges:
            key = (e.source_id, e.target_id, e.relation_type)
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        data = {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in unique_edges],
            "node_id_list": self.node_id_list,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)

    @classmethod
    def load(cls, path: str) -> "VKGraph":
        with open(path) as f:
            data = json.load(f)
        g = cls()
        for nd in data["nodes"]:
            g.add_node(VKGNode.from_dict(nd))
        for ed in data["edges"]:
            g.add_edge(VKGEdge.from_dict(ed))
        g.node_id_list = data.get("node_id_list", [])
        return g


class _SentinelNode:
    """Lightweight sentinel for SortedList bisect comparisons on t_start."""
    def __init__(self, t_start: float):
        self.t_start = t_start


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")
