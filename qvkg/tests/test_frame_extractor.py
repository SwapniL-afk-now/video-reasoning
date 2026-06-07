"""Tests for frame extractor, time parsing, gap detection, and temporal range query."""
import pytest
import numpy as np
from PIL import Image

from qvkg.query.intent import parse_time_reference
from qvkg.schema import VKGraph, VKGNode


# ---------------------------------------------------------------------------
# parse_time_reference
# ---------------------------------------------------------------------------

class TestParseTimeReference:
    def test_mm_ss_range(self):
        result = parse_time_reference("03:56-05:00")
        assert result is not None
        assert abs(result[0] - 236) < 1
        assert abs(result[1] - 300) < 1

    def test_pinpoint(self):
        result = parse_time_reference("40:20-40:20")
        assert result is not None
        assert result[0] == result[1]
        assert abs(result[0] - 2420) < 1

    def test_large_minutes(self):
        # LVBench uses 88:07 meaning 88 min 7 sec
        result = parse_time_reference("88:07-88:07")
        assert result is not None
        assert abs(result[0] - (88*60 + 7)) < 1

    def test_full_video_range(self):
        result = parse_time_reference("00:00-100:54")
        assert result is not None
        assert result[0] == 0.0
        assert abs(result[1] - (100*60 + 54)) < 1

    def test_reversed_range(self):
        # LVBench has some entries with end < start (85:50-85:35)
        result = parse_time_reference("85:50-85:35")
        assert result is not None
        assert result[0] <= result[1]

    def test_empty_returns_none(self):
        assert parse_time_reference("") is None
        assert parse_time_reference("   ") is None

    def test_invalid_returns_none(self):
        assert parse_time_reference("not-a-time") is None


# ---------------------------------------------------------------------------
# VKGraph.get_nodes_in_window
# ---------------------------------------------------------------------------

def make_graph_with_nodes():
    g = VKGraph()
    for i in range(10):
        n = VKGNode(
            id=f"n{i}", node_type="ClipNode", label=f"clip {i}",
            level=0, t_start=float(i * 60), t_end=float(i * 60 + 30),
        )
        g.add_node(n)
    return g


class TestGetNodesInWindow:
    def test_returns_overlapping_nodes(self):
        g = make_graph_with_nodes()
        # Nodes span [t, t+30]. Window [100, 140]:
        # node@60: 60-90 → 90 < 100 → no overlap
        # node@120: 120-150 → overlaps [100,140]
        nodes = g.get_nodes_in_window(100, 140, buffer_sec=0)
        ts = [n.t_start for n in nodes]
        assert 60 not in ts  # 60-90 does NOT overlap [100,140]
        assert 120 in ts     # 120-150 overlaps [100,140]

    def test_buffer_expands_window(self):
        g = make_graph_with_nodes()
        nodes = g.get_nodes_in_window(100, 140, buffer_sec=30)
        ts = [n.t_start for n in nodes]
        assert 60 in ts  # now 60-90 overlaps [70, 170]

    def test_empty_window_returns_nothing(self):
        g = make_graph_with_nodes()
        nodes = g.get_nodes_in_window(10000, 20000, buffer_sec=0)
        assert len(nodes) == 0


# ---------------------------------------------------------------------------
# VKGraph.compute_temporal_precision
# ---------------------------------------------------------------------------

class TestComputeTemporalPrecision:
    def test_clip_nodes_have_zero_precision(self):
        g = VKGraph()
        clip = VKGNode(id="c0", node_type="ClipNode", label="clip",
                       level=0, t_start=0, t_end=1)
        g.add_node(clip)
        g.compute_temporal_precision()
        assert g.nodes["c0"].temporal_precision == 0.0

    def test_scene_node_precision_is_distance_to_nearest_clip(self):
        g = VKGraph()
        clip = VKGNode(id="c0", node_type="ClipNode", label="clip",
                       level=0, t_start=10, t_end=11)
        scene = VKGNode(id="s0", node_type="SceneNode", label="scene",
                        level=1, t_start=15, t_end=30)
        g.add_node(clip)
        g.add_node(scene)
        g.compute_temporal_precision()
        assert abs(g.nodes["s0"].temporal_precision - 5.0) < 0.1

    def test_no_clips_gives_high_precision_value(self):
        g = VKGraph()
        scene = VKGNode(id="s0", node_type="SceneNode", label="scene",
                        level=1, t_start=15, t_end=30)
        g.add_node(scene)
        g.compute_temporal_precision()
        assert g.nodes["s0"].temporal_precision == 999.0


# ---------------------------------------------------------------------------
# frame_extractor helpers (no real video — test logic only)
# ---------------------------------------------------------------------------

class TestFrameExtractorLogic:
    def test_rank_by_motion_returns_top_k(self):
        from qvkg.query.frame_extractor import _rank_by_motion
        from qvkg.schema import FrameInfo

        frames = []
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            frames.append(FrameInfo(id=f"f{i}", timestamp=float(i), image=img))

        ranked = _rank_by_motion(frames, top_k=4)
        assert len(ranked) == 4
        # Should be in chronological order
        ts = [f.timestamp for f in ranked]
        assert ts == sorted(ts)

    def test_extract_returns_empty_for_bad_path(self):
        from qvkg.query.frame_extractor import extract_frames_for_window
        frames = extract_frames_for_window("/nonexistent/video.mp4", 0, 10)
        assert frames == []
