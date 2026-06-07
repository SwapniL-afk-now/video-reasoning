from __future__ import annotations

"""Description-based character identity resolver using DBSCAN clustering."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .schema import VKGNode


@dataclass
class CharacterMention:
    scene_id:    str
    timestamp:   float
    description: str
    bbox:        Optional[List[float]] = None
    action:      Optional[str]        = None
    emotion:     Optional[str]        = None


class DescriptionBasedCharacterResolver:
    """
    Clusters character appearance descriptions with DBSCAN to resolve
    cross-scene identity. Works without audio (unlike HAVEN) and for
    any video domain (unlike egocentric-only MAGIC-Video/EgoGraph).
    """

    def resolve(
        self,
        mentions: List[CharacterMention],
        siglip_encoder,
        similarity_threshold: float = 0.80,
    ) -> Optional[List[VKGNode]]:
        from sklearn.cluster import DBSCAN

        if not mentions:
            return None

        descriptions = [m.description for m in mentions]

        try:
            embeddings = siglip_encoder.encode_text(descriptions)
        except Exception as e:
            print(f"  Character resolution embedding failed: {e}")
            return None

        import faiss
        faiss.normalize_L2(embeddings)

        labels = DBSCAN(
            eps=1.0 - similarity_threshold,
            min_samples=1,
            metric="cosine",
        ).fit_predict(embeddings)

        clusters: dict = {}
        for mention, label in zip(mentions, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(mention)

        char_nodes: List[VKGNode] = []
        for label, cluster_mentions in clusters.items():
            # Pick most detailed description as canonical
            canonical = max(cluster_mentions, key=lambda m: len(m.description))
            appearances = [
                {
                    "scene_id":  m.scene_id,
                    "timestamp": m.timestamp,
                    "bbox":      m.bbox,
                    "action":    m.action,
                    "emotion":   m.emotion,
                    "description": m.description,
                }
                for m in cluster_mentions
            ]
            node = VKGNode(
                id=f"char_{label}",
                node_type="CharacterNode",
                label=f"Person_{label}",
                level=0,
                t_start=min(m.timestamp for m in cluster_mentions),
                t_end=max(m.timestamp for m in cluster_mentions),
                entity_id=f"entity_char_{label}",
                canonical_description=canonical.description,
                metadata={"appearances": appearances},
            )
            char_nodes.append(node)

        return char_nodes
