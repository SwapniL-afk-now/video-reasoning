from __future__ import annotations

"""SubgraphActivator: FAISS seed search + typed BFS/DFS graph expansion."""

from typing import List, Set, Tuple

import faiss
import numpy as np

from ..schema import SubGraph, VKGNode, VKGraph

CAUSAL_TYPES = {"CAUSES", "ENABLES", "PREVENTS", "MOTIVATES"}


class SubgraphActivator:
    def __init__(
        self,
        graph: VKGraph,
        faiss_index: faiss.Index,
        siglip_encoder,
        max_nodes: int = 60,
    ):
        self.graph = graph
        self.index = faiss_index
        self.siglip = siglip_encoder
        self.max_nodes = max_nodes

    def activate(self, question: str, intents: List[str]) -> SubGraph:
        # Step 1: Embed question → FAISS seed nodes
        q_emb = self.siglip.encode_text([question]).astype(np.float32)
        faiss.normalize_L2(q_emb)
        k = min(20, len(self.graph.node_id_list))
        if k == 0:
            return SubGraph({}, [])

        sims, idx = self.index.search(q_emb, k)
        seeds: List[VKGNode] = []
        for i, sim in zip(idx[0], sims[0]):
            if int(i) < 0:
                continue
            if float(sim) < 0.3:
                continue
            nid = self.graph.node_id_list[int(i)]
            node = self.graph.nodes.get(nid)
            if node:
                seeds.append(node)

        activated: Set[str] = {n.id for n in seeds}

        # Step 2: Intent-driven graph expansion
        for intent in intents:
            if intent == "TEMPORAL":
                for node in seeds:
                    activated |= self._walk_temporal_spine(node, hops=4)

            elif intent == "CAUSAL":
                for node in seeds:
                    activated |= self._follow_causal_edges(node, depth=3)

            elif intent == "IDENTITY":
                for node in seeds:
                    if node.entity_id:
                        activated |= set(
                            self.graph.entity_idx.get(node.entity_id, [])
                        )
                    # Also activate all CharacterNodes with same entity_id
                    for cnode in self.graph.get_nodes_by_type("CharacterNode"):
                        if cnode.entity_id == node.entity_id:
                            activated.add(cnode.id)
                    # Follow SAME_ENTITY edges in both directions
                    for edge in self.graph.get_edges(node.id):
                        if edge.relation_type == "SAME_ENTITY":
                            activated.add(edge.target_id)
                    for edge in self.graph.get_incoming_edges(node.id):
                        if edge.relation_type == "SAME_ENTITY":
                            activated.add(edge.source_id)
                # Narrator node: activate if question references voice-over
                q_lower = question.lower()
                if any(w in q_lower for w in
                       ("narrator", "voiceover", "voice over", "narration", "voice-over")):
                    narrator = self.graph.nodes.get("narrator")
                    if narrator:
                        activated.add("narrator")
                        activated |= set(
                            self.graph.entity_idx.get("entity_narrator", [])
                        )

            elif intent == "SPATIAL":
                for node in seeds:
                    activated |= self._expand_spatial(node)

            elif intent == "STATE":
                q_lower = question.lower()
                for sc_node in self.graph.get_nodes_by_type("StateChangeNode"):
                    label_l = sc_node.label.lower()
                    entity_l = sc_node.metadata.get("entity", "").lower()
                    if any(w in label_l or w in entity_l
                           for w in q_lower.split() if len(w) > 3):
                        activated.add(sc_node.id)

            elif intent == "SUMMARY":
                for ep in self.graph.get_episodes():
                    activated.add(ep.id)
                    for child in self.graph.get_children(ep, depth=1):
                        activated.add(child.id)

        # Step 3: Include parent context
        parents = set()
        for nid in list(activated):
            node = self.graph.nodes.get(nid)
            if node and node.parent_id:
                parents.add(node.parent_id)
        activated |= parents

        # Step 4: Prune to budget
        if len(activated) > self.max_nodes:
            activated = self._prune_to_budget(activated, seeds)

        sub_nodes = {nid: self.graph.nodes[nid]
                     for nid in activated if nid in self.graph.nodes}
        sub_edges = [
            e for nid in activated
            for e in self.graph.get_edges(nid)
            if e.target_id in activated
        ]
        return SubGraph(sub_nodes, sub_edges)

    def activate_by_time_reference(
        self,
        t_start: float,
        t_end: float,
        intents: List[str],
        buffer_sec: float = 30.0,
    ) -> Tuple[SubGraph, float]:
        """Temporal range query activation — replaces FAISS for time-ref questions.

        Returns (subgraph, min_temporal_precision).
        min_temporal_precision signals whether on-demand frame extraction is needed:
        if > gap_threshold (e.g. 8s), the graph has a hole at this window.
        """
        # Step 1: temporal range query (O(log N))
        window_nodes = self.graph.get_nodes_in_window(t_start, t_end, buffer_sec)
        activated: Set[str] = {n.id for n in window_nodes}

        if not activated:
            # Fallback: grab nearest nodes by time
            all_sorted = list(self.graph.temporal_idx)
            mid = (t_start + t_end) / 2
            nearest = sorted(all_sorted, key=lambda n: abs(n.t_start - mid))[:20]
            activated = {n.id for n in nearest}

        seeds = [self.graph.nodes[nid] for nid in activated if nid in self.graph.nodes]

        # Step 2: intent-driven expansion (same logic as activate())
        for intent in intents:
            if intent == "CAUSAL":
                for node in seeds:
                    activated |= self._follow_causal_edges(node, depth=3)

            elif intent == "IDENTITY":
                for node in seeds:
                    if node.entity_id:
                        activated |= set(
                            self.graph.entity_idx.get(node.entity_id, [])
                        )
                    # Follow SAME_ENTITY edges in both directions
                    for edge in self.graph.get_edges(node.id):
                        if edge.relation_type == "SAME_ENTITY":
                            activated.add(edge.target_id)
                    for edge in self.graph.get_incoming_edges(node.id):
                        if edge.relation_type == "SAME_ENTITY":
                            activated.add(edge.source_id)

            elif intent == "SPATIAL":
                for node in seeds:
                    activated |= self._expand_spatial(node)

            elif intent == "STATE":
                for sc_node in self.graph.get_nodes_by_type("StateChangeNode"):
                    if sc_node.t_start >= t_start - buffer_sec and sc_node.t_end <= t_end + buffer_sec:
                        activated.add(sc_node.id)

        # Step 3: include parent context
        parents: Set[str] = set()
        for nid in list(activated):
            node = self.graph.nodes.get(nid)
            if node and node.parent_id:
                parents.add(node.parent_id)
        activated |= parents

        # Step 4: budget prune
        if len(activated) > self.max_nodes:
            activated = self._prune_to_budget(activated, seeds)

        # Step 5: compute min temporal precision — gap detection signal
        min_prec = min(
            (self.graph.nodes[nid].temporal_precision
             for nid in activated if nid in self.graph.nodes),
            default=999.0,
        )

        sub_nodes = {nid: self.graph.nodes[nid] for nid in activated if nid in self.graph.nodes}
        sub_edges = [
            e for nid in activated
            for e in self.graph.get_edges(nid)
            if e.target_id in activated
        ]
        return SubGraph(sub_nodes, sub_edges), min_prec

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def _walk_temporal_spine(self, node: VKGNode, hops: int) -> Set[str]:
        activated: Set[str] = set()
        for direction in ["PRECEDES", "PRECEDES_REV"]:
            curr = node
            for _ in range(hops):
                neighbor = self.graph.get_neighbor(curr, direction)
                if not neighbor:
                    break
                activated.add(neighbor.id)
                curr = neighbor
        return activated

    def _follow_causal_edges(self, node: VKGNode, depth: int) -> Set[str]:
        activated: Set[str] = set()
        queue = [(node, 0)]
        while queue:
            curr, d = queue.pop(0)
            if d >= depth:
                continue
            for edge in self.graph.get_edges(curr.id):
                if edge.relation_type in CAUSAL_TYPES:
                    nb = self.graph.nodes.get(edge.target_id)
                    if nb and nb.id not in activated:
                        activated.add(nb.id)
                        queue.append((nb, d + 1))
            for edge in self.graph.get_incoming_edges(curr.id):
                if edge.relation_type in CAUSAL_TYPES:
                    nb = self.graph.nodes.get(edge.source_id)
                    if nb and nb.id not in activated:
                        activated.add(nb.id)
                        queue.append((nb, d + 1))
        return activated

    def _expand_spatial(self, node: VKGNode) -> Set[str]:
        spatial_types = {"LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW",
                         "IN_FRONT_OF", "BEHIND", "NEAR", "CONTAINS_SPATIAL"}
        activated: Set[str] = set()
        for edge in self.graph.get_edges(node.id):
            if edge.relation_type in spatial_types:
                activated.add(edge.target_id)
        for edge in self.graph.get_incoming_edges(node.id):
            if edge.relation_type in spatial_types:
                activated.add(edge.source_id)
        return activated

    def _prune_to_budget(
        self, activated: Set[str], seeds: List[VKGNode]
    ) -> Set[str]:
        seed_ids = {n.id for n in seeds}
        # Always keep seeds and their parents
        keep = seed_ids & activated
        # Fill remaining budget by confidence descending
        remaining = [
            self.graph.nodes[nid]
            for nid in activated
            if nid not in keep and nid in self.graph.nodes
        ]
        remaining.sort(key=lambda n: n.confidence, reverse=True)
        for n in remaining:
            if len(keep) >= self.max_nodes:
                break
            keep.add(n.id)
        return keep
