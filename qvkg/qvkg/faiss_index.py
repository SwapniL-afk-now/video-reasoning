from __future__ import annotations

"""FAISS HNSW index construction and semantic edge building."""

import os
from typing import Optional

import faiss
import numpy as np

from .schema import VKGEdge, VKGraph


def _node_text(node) -> str:
    """Produce a rich text description of a node for SigLIP embedding."""
    t = f"{node.t_start:.0f}s"
    nt = node.node_type
    if nt == "ObjectNode":
        attrs = ", ".join(node.metadata.get("attributes", []))
        state = node.metadata.get("state", "")
        desc = node.label
        if attrs:
            desc += f" [{attrs}]"
        if state:
            desc += f" ({state})"
        return f"object: {desc} at {t}"
    if nt == "ActionNode":
        actor = node.metadata.get("actor", "")
        obj   = node.metadata.get("object", "")
        desc  = node.label
        if actor:
            desc += f" by {actor}"
        if obj:
            desc += f" on {obj}"
        return f"action: {desc} at {t}"
    if nt == "CharacterNode":
        canon = node.canonical_description or node.label
        return f"character: {canon} at {t}"
    if nt == "SpeechNode":
        # Speech text is already the best description; prepend source for narrator
        src = node.metadata.get("source", "")
        prefix = "[NARRATOR] " if src == "narrator" else ""
        return f"{prefix}{node.label}"
    if nt == "StateChangeNode":
        frm = node.prev_state or ""
        to  = node.next_state or ""
        return f"state change: {node.label} from '{frm}' to '{to}' at {t}"
    if nt == "OCRNode":
        sem = node.metadata.get("semantic_type", "")
        return f"text on screen: '{node.label}' ({sem}) at {t}"
    if nt == "LocationNode":
        return f"location: {node.label} from {node.t_start:.0f}s to {node.t_end:.0f}s"
    if nt == "SceneNode":
        loc = node.metadata.get("location", "")
        mood = node.metadata.get("mood", "")
        desc = node.label
        if loc and loc != "unknown":
            desc += f" at {loc}"
        if mood and mood != "neutral":
            desc += f" ({mood})"
        return f"scene: {desc} at {t}"
    if nt == "EpisodeNode":
        summary = node.metadata.get("summary", "")
        role    = node.metadata.get("narrative_role", "")
        desc = node.label
        if summary:
            desc += f" — {summary[:80]}"
        if role:
            desc += f" [{role}]"
        return f"episode: {desc} at {t}"
    if nt == "AudioEventNode":
        return f"audio event: {node.label}"
    # Default: keep type prefix for unknowns
    return f"{nt}: {node.label} at {t}"


def build_faiss_index(
    graph: VKGraph,
    siglip_encoder,
    index_path: str,
    frame_store=None,
) -> faiss.Index:
    nodes = list(graph.nodes.values())
    if not nodes:
        dim = siglip_encoder.embedding_dim
        index = faiss.IndexHNSWFlat(dim, 32)
        return index

    # Encode all nodes using type-specific rich text descriptions
    texts = [_node_text(n) for n in nodes]
    embeddings = siglip_encoder.encode_text(texts).astype(np.float32)

    # Fuse visual nodes with image embeddings
    if frame_store is not None:
        for i, node in enumerate(nodes):
            fid = node.keyframe_id
            if fid:
                img = frame_store.load_image(fid)
                if img is not None:
                    img_emb = siglip_encoder.encode_image(img).astype(np.float32)
                    embeddings[i] = (embeddings[i] + img_emb) / 2.0

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.add(embeddings)

    # Persist node id list on graph for FAISS ↔ node_id mapping
    graph.node_id_list = [n.id for n in nodes]

    faiss.write_index(index, index_path)
    return index


def load_faiss_index(index_path: str) -> Optional[faiss.Index]:
    if not os.path.exists(index_path):
        return None
    return faiss.read_index(index_path)


def build_semantic_edges_faiss(
    graph: VKGraph,
    faiss_index: faiss.Index,
    threshold: float = 0.78,
    k_neighbors: int = 10,
) -> int:
    n = len(graph.node_id_list)
    if n == 0:
        return 0

    embeddings = np.vstack([
        faiss_index.reconstruct(i) for i in range(n)
    ]).astype(np.float32)

    similarities, indices = faiss_index.search(embeddings, k_neighbors + 1)

    added = 0
    for i, (sims, nbrs) in enumerate(zip(similarities, indices)):
        node_a = graph.nodes.get(graph.node_id_list[i])
        if node_a is None:
            continue
        for sim, j in zip(sims[1:], nbrs[1:]):
            if int(j) < 0:
                break
            if float(sim) < threshold:
                break
            node_b = graph.nodes.get(graph.node_id_list[int(j)])
            if node_b is None:
                continue
            # Skip parent-child pairs
            if node_b.id == node_a.parent_id or node_a.id == node_b.parent_id:
                continue
            graph.add_edge(VKGEdge(
                source_id=node_a.id,
                target_id=node_b.id,
                relation_type="SIMILAR_TO",
                weight=float(sim),
                confidence=float(sim),
                metadata={"source": "faiss_hnsw"},
            ))
            added += 1

    return added
