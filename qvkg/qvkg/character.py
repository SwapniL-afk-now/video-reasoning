from __future__ import annotations

"""Character identity resolution: LLM-based (primary) with DBSCAN fallback."""

import json
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


class LLMEntityResolver:
    """
    Resolves cross-scene character identity using the VLM.

    The VLM receives all character descriptions with timestamps and returns
    a canonical entity map — handling any domain cue (jersey numbers, face
    descriptions, voice descriptions, on-screen name chyrons).

    Falls back to DescriptionBasedCharacterResolver (DBSCAN) on LLM failure.
    """

    # Max descriptions per LLM batch — keep small enough that output fits in 32k tokens
    BATCH_SIZE = 150

    def resolve(
        self,
        mentions: List[CharacterMention],
        llm,
        video_type: str = "",
    ) -> Optional[List[VKGNode]]:
        from .vllm_client import (
            ENTITY_RESOLUTION_SAMPLING,
            ENTITY_RESOLUTION_SYSTEM_PROMPT,
        )

        if not mentions:
            return None

        # Format all descriptions with timestamps for context
        desc_lines = [
            f"[{m.timestamp:.0f}s, scene {m.scene_id}] {m.description}"
            + (f" (doing: {m.action})" if m.action else "")
            + (f" (emotion: {m.emotion})" if m.emotion else "")
            for m in mentions
        ]

        # Batch if too many descriptions
        batches = [
            desc_lines[i:i + self.BATCH_SIZE]
            for i in range(0, len(desc_lines), self.BATCH_SIZE)
        ]

        # Submit every chunk in a single batched llm.chat so vLLM processes them
        # concurrently (continuous batching) rather than one blocking call each.
        batch_messages = [
            [
                {"role": "system", "content": ENTITY_RESOLUTION_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Video type: {video_type or 'unknown'}\n"
                    f"Total character mentions to resolve: {len(batch)}\n\n"
                    "Character descriptions (format: [timestamp, scene_id] description):\n"
                    + "\n".join(batch)
                    + "\n\nGroup these into canonical entities. "
                      "Return one entry per unique real-world person."
                )},
            ]
            for batch in batches
        ]

        all_entities: List[dict] = []
        try:
            outputs = llm.chat(
                messages=batch_messages,
                sampling_params=ENTITY_RESOLUTION_SAMPLING,
            )
            for out in outputs:
                all_entities.extend(json.loads(out.outputs[0].text))
        except Exception as e:
            print(f"  LLM entity resolution failed: {e}")
            return None  # triggers fallback in caller

        if not all_entities:
            return None

        # Build description → entity mapping
        desc_to_entity: dict = {}
        for ent in all_entities:
            for variant in ent.get("description_variants", []):
                desc_to_entity[variant] = ent

        # Build VKGNodes — one per canonical entity
        char_nodes: List[VKGNode] = []
        for ent in all_entities:
            canonical_id = ent.get("canonical_id", f"entity_{len(char_nodes)}")
            # Collect all mentions that match this entity's variants
            variants = set(ent.get("description_variants", []))
            entity_mentions = [m for m in mentions if m.description in variants]

            # Fallback: if no mentions matched (LLM invented a variant), match by
            # partial description overlap to avoid losing data
            if not entity_mentions:
                canon_desc = ent.get("canonical_description", "").lower()
                entity_mentions = [
                    m for m in mentions
                    if any(w in m.description.lower()
                           for w in canon_desc.split() if len(w) > 4)
                ]

            if not entity_mentions:
                continue

            appearances = [
                {
                    "scene_id":    m.scene_id,
                    "timestamp":   m.timestamp,
                    "bbox":        m.bbox,
                    "action":      m.action,
                    "emotion":     m.emotion,
                    "description": m.description,
                }
                for m in entity_mentions
            ]

            node = VKGNode(
                id=f"char_{canonical_id}",
                node_type="CharacterNode",
                label=ent.get("canonical_name", "") or ent.get("canonical_description", canonical_id)[:80],
                level=0,
                t_start=min(m.timestamp for m in entity_mentions),
                t_end=max(m.timestamp for m in entity_mentions),
                entity_id=f"entity_{canonical_id}",
                canonical_description=ent.get("canonical_description", ""),
                metadata={
                    "appearances":   appearances,
                    "entity_type":   ent.get("entity_type", "other"),
                    "canonical_id":  canonical_id,
                },
            )
            char_nodes.append(node)

        return char_nodes if char_nodes else None


class DescriptionBasedCharacterResolver:
    """
    DBSCAN-based character resolver — kept as fallback when LLM resolution fails.
    Uses SigLIP text embeddings + token-overlap second pass.
    """

    def resolve(
        self,
        mentions: List[CharacterMention],
        siglip_encoder,
        similarity_threshold: float = 0.65,
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
            clusters.setdefault(label, []).append(mention)

        # Second pass: merge clusters sharing >= 40% token Jaccard
        cluster_keys = list(clusters.keys())
        canonicals = {
            k: max(clusters[k], key=lambda m: len(m.description)).description
            for k in cluster_keys
        }
        parent = {k: k for k in cluster_keys}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _token_jaccard(a: str, b: str) -> float:
            wa = {w for w in a.lower().split() if len(w) > 3}
            wb = {w for w in b.lower().split() if len(w) > 3}
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / len(wa | wb)

        for i, ki in enumerate(cluster_keys):
            for kj in cluster_keys[i + 1:]:
                if _find(ki) != _find(kj):
                    if _token_jaccard(canonicals[ki], canonicals[kj]) >= 0.40:
                        parent[_find(kj)] = _find(ki)

        merged: dict = {}
        for k in cluster_keys:
            merged.setdefault(_find(k), []).extend(clusters[k])

        char_nodes: List[VKGNode] = []
        for label, cluster_mentions in merged.items():
            canonical = max(cluster_mentions, key=lambda m: len(m.description))
            appearances = [
                {
                    "scene_id":    m.scene_id,
                    "timestamp":   m.timestamp,
                    "bbox":        m.bbox,
                    "action":      m.action,
                    "emotion":     m.emotion,
                    "description": m.description,
                }
                for m in cluster_mentions
            ]
            node = VKGNode(
                id=f"char_{label}",
                node_type="CharacterNode",
                label=canonical.description[:80] or f"Person_{label}",
                level=0,
                t_start=min(m.timestamp for m in cluster_mentions),
                t_end=max(m.timestamp for m in cluster_mentions),
                entity_id=f"entity_char_{label}",
                canonical_description=canonical.description,
                metadata={"appearances": appearances},
            )
            char_nodes.append(node)

        return char_nodes


class ObjectIdentityResolver:
    """
    DBSCAN-based object identity resolver.

    Clusters ObjectNodes with similar label+attribute descriptions across scenes
    so that the same physical object (e.g. 'red sports car') is linked with
    SAME_ENTITY edges regardless of which scene it appears in.
    Uses the same SigLIP text embedding + token-Jaccard merge strategy as
    DescriptionBasedCharacterResolver.
    """

    def resolve(
        self,
        obj_nodes: List[VKGNode],
        siglip_encoder,
        similarity_threshold: float = 0.70,
    ) -> Optional[List[List[VKGNode]]]:
        """Return clusters of ObjectNodes that represent the same physical object.

        Returns a list of clusters (each cluster is a list of VKGNode). Returns
        None on failure. Singletons are excluded (need ≥2 occurrences).
        """
        from sklearn.cluster import DBSCAN

        if not obj_nodes:
            return None

        def _obj_desc(n: VKGNode) -> str:
            attrs = ", ".join(n.metadata.get("attributes", []))
            state = n.metadata.get("state", "")
            parts = [n.label]
            if attrs:
                parts.append(attrs)
            if state:
                parts.append(state)
            return " ".join(parts)

        descriptions = [_obj_desc(n) for n in obj_nodes]

        try:
            embeddings = siglip_encoder.encode_text(descriptions)
        except Exception as e:
            print(f"  Object identity resolution embedding failed: {e}")
            return None

        import faiss
        import numpy as np
        emb = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb)

        labels = DBSCAN(
            eps=1.0 - similarity_threshold,
            min_samples=2,
            metric="cosine",
        ).fit_predict(emb)

        clusters: dict = {}
        for node, label in zip(obj_nodes, labels):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(node)

        cluster_keys = list(clusters.keys())

        def _token_jaccard(a: str, b: str) -> float:
            wa = {w for w in a.lower().split() if len(w) > 3}
            wb = {w for w in b.lower().split() if len(w) > 3}
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / len(wa | wb)

        canonicals = {
            k: max(clusters[k], key=lambda n: len(_obj_desc(n)))
            for k in cluster_keys
        }
        parent = {k: k for k in cluster_keys}

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, ki in enumerate(cluster_keys):
            for kj in cluster_keys[i + 1:]:
                if _find(ki) != _find(kj):
                    if _token_jaccard(_obj_desc(canonicals[ki]),
                                      _obj_desc(canonicals[kj])) >= 0.40:
                        parent[_find(kj)] = _find(ki)

        merged: dict = {}
        for k in cluster_keys:
            merged.setdefault(_find(k), []).extend(clusters[k])

        return [group for group in merged.values() if len(group) >= 2]
