from __future__ import annotations

"""Free-function graph traversal layer for the inference-time walker.

These promote the edge-following logic that previously lived inside
``SubgraphActivator._follow_*`` so that both the walker (``walker.py``) and the
legacy QA path can share a single, typed expansion primitive.

The central entry point is :func:`expand`, which follows one *relation family*
from a frontier of node ids and returns the newly reached node ids plus the
edges traversed. Every move the walker makes on the graph goes through here.
"""

from typing import Dict, List, Set, Tuple

from ..schema import CAUSAL_EDGE_TYPES, SubGraph, VKGEdge, VKGNode, VKGraph

# ---------------------------------------------------------------------------
# Relation family → concrete edge-type sets
# ---------------------------------------------------------------------------

# The walker exposes a small, stable set of relation families. Each maps to one
# or more concrete edge types from the schema taxonomy.
RELATION_EDGE_TYPES: Dict[str, Set[str]] = {
    "CAUSAL":   set(CAUSAL_EDGE_TYPES),                 # CAUSES/ENABLES/PREVENTS/MOTIVATES
    "ENTITY":   {"SAME_ENTITY"},                        # + entity_idx threading (special-cased)
    "SPEAKER":  {"SPOKEN_BY"},
    "TEMPORAL": {"PRECEDES", "OVERLAPS", "DURING"},     # temporal backbone adjacency
    "EMOTION":  {"EMOTION_SHIFT"},
    "SIMILAR":  {"SIMILAR_TO", "DESCRIBES"},
    "CONTAINS": {"CONTAINS"},
}

VALID_RELATIONS = tuple(RELATION_EDGE_TYPES.keys())


# ---------------------------------------------------------------------------
# Core expansion
# ---------------------------------------------------------------------------

def expand(
    graph: VKGraph,
    frontier: Set[str],
    relation: str,
    k: int = 5,
) -> Tuple[Set[str], List[VKGEdge]]:
    """Follow ``relation`` from every node in ``frontier`` (both directions).

    Returns ``(new_node_ids, edges_traversed)`` where ``new_node_ids`` are the
    neighbours reached (excluding the frontier itself) and ``edges_traversed``
    are the edges that connect frontier → neighbour.

    Fan-out is bounded: at most ``k`` neighbours per frontier node, ranked by
    edge confidence (then source-node confidence) descending. This keeps the
    working subgraph small enough for the answerer to read while still following
    the highest-signal edges first.
    """
    rel = relation.upper().strip()
    edge_types = RELATION_EDGE_TYPES.get(rel)
    if not edge_types:
        return set(), []

    new_nodes: Set[str] = set()
    edges: List[VKGEdge] = []
    seen_edges: Set[Tuple[str, str, str]] = set()

    for nid in frontier:
        node = graph.nodes.get(nid)
        if node is None:
            continue

        # Gather candidate (neighbour_id, edge) pairs in both directions.
        candidates: List[Tuple[str, VKGEdge]] = []
        for e in graph.get_edges(nid):
            if e.relation_type in edge_types:
                candidates.append((e.target_id, e))
        for e in graph.get_incoming_edges(nid):
            if e.relation_type in edge_types:
                candidates.append((e.source_id, e))

        # ENTITY: also thread through the global entity index — a character's
        # appearances are linked by shared entity_id rather than dense edges.
        if rel == "ENTITY" and node.entity_id:
            for aid in graph.entity_idx.get(node.entity_id, []):
                if aid != nid and aid in graph.nodes:
                    synth = VKGEdge(
                        source_id=nid,
                        target_id=aid,
                        relation_type="SAME_ENTITY",
                        weight=1.0,
                        confidence=1.0,
                        metadata={"source": "entity_idx"},
                    )
                    candidates.append((aid, synth))

        # Rank by edge confidence, then neighbour-node confidence.
        def _rank(pair: Tuple[str, VKGEdge]) -> float:
            tid, e = pair
            tgt = graph.nodes.get(tid)
            return (e.confidence, tgt.confidence if tgt else 0.0)

        candidates.sort(key=_rank, reverse=True)

        taken = 0
        for tid, e in candidates:
            if tid not in graph.nodes:
                continue
            ekey = (e.source_id, e.target_id, e.relation_type)
            if ekey not in seen_edges:
                seen_edges.add(ekey)
                edges.append(e)
            if tid not in frontier:
                new_nodes.add(tid)
            taken += 1
            if taken >= k:
                break

    return new_nodes, edges


# ---------------------------------------------------------------------------
# Subgraph construction
# ---------------------------------------------------------------------------

def induced_edges(graph: VKGraph, node_ids: Set[str]) -> List[VKGEdge]:
    """All edges with both endpoints inside ``node_ids`` (deduplicated)."""
    edges: List[VKGEdge] = []
    seen: Set[Tuple[str, str, str]] = set()
    for nid in node_ids:
        for e in graph.get_edges(nid):
            if e.target_id in node_ids:
                key = (e.source_id, e.target_id, e.relation_type)
                if key not in seen:
                    seen.add(key)
                    edges.append(e)
    return edges


def build_subgraph(graph: VKGraph, node_ids: Set[str]) -> SubGraph:
    """Build a :class:`SubGraph` view from ``node_ids`` with induced edges."""
    nodes = {nid: graph.nodes[nid] for nid in node_ids if nid in graph.nodes}
    return SubGraph(nodes, induced_edges(graph, set(nodes.keys())))


# ---------------------------------------------------------------------------
# Entity / causal helpers
# ---------------------------------------------------------------------------

def trace_entity(graph: VKGraph, entity_id: str) -> List[VKGNode]:
    """All appearances of ``entity_id`` sorted by time."""
    nodes = [graph.nodes[nid] for nid in graph.entity_idx.get(entity_id, [])
             if nid in graph.nodes]
    return sorted(nodes, key=lambda n: n.t_start)


def causal_parents(graph: VKGraph, node_id: str) -> List[VKGNode]:
    """Nodes that causally lead into ``node_id`` (incoming causal edges)."""
    out: List[VKGNode] = []
    for e in graph.get_incoming_edges(node_id):
        if e.relation_type in CAUSAL_EDGE_TYPES:
            n = graph.nodes.get(e.source_id)
            if n:
                out.append(n)
    return out


def causal_children(graph: VKGraph, node_id: str) -> List[VKGNode]:
    """Nodes that ``node_id`` causally leads into (outgoing causal edges)."""
    out: List[VKGNode] = []
    for e in graph.get_edges(node_id):
        if e.relation_type in CAUSAL_EDGE_TYPES:
            n = graph.nodes.get(e.target_id)
            if n:
                out.append(n)
    return out


# ---------------------------------------------------------------------------
# FAISS semantic hop (used by RECALL / DISCRIMINATE)
# ---------------------------------------------------------------------------

def faiss_search(
    graph: VKGraph,
    faiss_index,
    siglip_encoder,
    query: str,
    k: int = 10,
    min_sim: float = 0.25,
) -> List[str]:
    """Semantic FAISS search → list of node ids (highest similarity first)."""
    if faiss_index is None or siglip_encoder is None or not graph.node_id_list:
        return []
    import faiss
    import numpy as np

    q_emb = siglip_encoder.encode_text([query]).astype(np.float32)
    faiss.normalize_L2(q_emb)
    kk = min(k, len(graph.node_id_list))
    if kk <= 0:
        return []
    sims, idx = faiss_index.search(q_emb, kk)
    out: List[str] = []
    for i, sim in zip(idx[0], sims[0]):
        if int(i) < 0 or float(sim) < min_sim:
            continue
        nid = graph.node_id_list[int(i)]
        if nid in graph.nodes:
            out.append(nid)
    return out
