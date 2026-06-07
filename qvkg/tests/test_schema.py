"""Tests for VKGraph schema: node/edge CRUD, subgraph, serialisation."""
import json
import os
import tempfile

import numpy as np
import pytest

from qvkg.schema import (
    EDGE_TYPES, VKGEdge, VKGNode, VKGraph, SubGraph
)


def make_node(i: int, node_type="ClipNode", t=None) -> VKGNode:
    return VKGNode(
        id=f"n{i}",
        node_type=node_type,
        label=f"node {i}",
        level=0,
        t_start=float(t or i),
        t_end=float((t or i) + 1),
    )


def make_edge(src, tgt, rel="PRECEDES") -> VKGEdge:
    return VKGEdge(source_id=src, target_id=tgt,
                   relation_type=rel, weight=1.0, confidence=1.0)


class TestVKGNode:
    def test_to_from_dict_roundtrip(self):
        n = make_node(1)
        n.siglip_embedding = np.ones(8)
        d = n.to_dict()
        assert "siglip_embedding" not in d
        n2 = VKGNode.from_dict(d)
        assert n2.id == n.id
        assert n2.node_type == n.node_type


class TestVKGraph:
    def test_add_node(self):
        g = VKGraph()
        n = make_node(0)
        g.add_node(n)
        assert "n0" in g.nodes

    def test_add_edge(self):
        g = VKGraph()
        g.add_node(make_node(0))
        g.add_node(make_node(1))
        e = make_edge("n0", "n1")
        g.add_edge(e)
        assert len(g.get_edges("n0")) == 1
        assert len(g.get_incoming_edges("n1")) == 1

    def test_entity_index(self):
        g = VKGraph()
        n = make_node(0)
        n.entity_id = "entity_alice"
        g.add_node(n)
        assert "n0" in g.entity_idx["entity_alice"]

    def test_type_index(self):
        g = VKGraph()
        g.add_node(make_node(0, "ClipNode"))
        g.add_node(make_node(1, "SceneNode"))
        assert len(g.get_nodes_by_type("ClipNode")) == 1
        assert len(g.get_nodes_by_type("SceneNode")) == 1

    def test_get_children(self):
        g = VKGraph()
        parent = make_node(0, "SceneNode")
        child  = make_node(1, "ClipNode")
        g.add_node(parent)
        g.add_node(child)
        g.add_edge(make_edge("n0", "n1", "CONTAINS"))
        children = g.get_children(parent, depth=1)
        assert any(c.id == "n1" for c in children)

    def test_induced_subgraph(self):
        g = VKGraph()
        for i in range(4):
            g.add_node(make_node(i))
        g.add_edge(make_edge("n0", "n1"))
        g.add_edge(make_edge("n1", "n2"))
        sub = g.induced_subgraph({"n0", "n1"})
        assert "n0" in sub.nodes
        assert "n2" not in sub.nodes
        assert len(sub.edges) == 1

    def test_save_load(self):
        g = VKGraph()
        g.add_node(make_node(0))
        g.add_node(make_node(1))
        g.add_edge(make_edge("n0", "n1"))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            g.save(path)
            g2 = VKGraph.load(path)
            assert "n0" in g2.nodes
            assert len(g2.get_edges("n0")) == 1
        finally:
            os.unlink(path)

    def test_find_event_by_description(self):
        g = VKGraph()
        n = make_node(0)
        n.label = "person cooking soup"
        g.add_node(n)
        found = g.find_event_by_description("cooking soup", [n])
        assert found is not None
        assert found.id == "n0"

    def test_edge_types_complete(self):
        required = ["CAUSES", "ENABLES", "PREVENTS", "MOTIVATES",
                    "PRECEDES", "CONTAINS", "SIMILAR_TO"]
        for r in required:
            assert r in EDGE_TYPES


class TestSubGraph:
    def test_get_sorted_events(self):
        n0 = make_node(0, t=10)
        n1 = make_node(1, t=5)
        sg = SubGraph({"n0": n0, "n1": n1}, [])
        events = sg.get_sorted_events()
        assert events[0].id == "n1"

    def test_get_state_changes(self):
        n = make_node(0, "StateChangeNode")
        n.prev_state = "off"
        n.next_state = "on"
        sg = SubGraph({"n0": n}, [])
        states = sg.get_state_changes()
        assert len(states) == 1
        assert states[0].prev_state == "off"

    def test_get_causal_chains(self):
        n0 = make_node(0)
        n1 = make_node(1)
        e = make_edge("n0", "n1", "CAUSES")
        e.metadata = {"reasoning": "X causes Y"}
        sg = SubGraph({"n0": n0, "n1": n1}, [e])
        chains = sg.get_causal_chains()
        assert len(chains) == 1
        assert chains[0].relation == "CAUSES"
