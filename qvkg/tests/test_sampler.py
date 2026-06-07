"""Unit tests for sampler helpers (no video file needed)."""
import numpy as np
import pytest

from qvkg.schema import FrameInfo
from qvkg.sampler import HierarchicalSampler


def make_frames(n: int, start: float = 0.0, step: float = 1.0):
    from PIL import Image
    frames = []
    for i in range(n):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        frames.append(FrameInfo(id=f"f{i:03d}", timestamp=start + i * step, image=img))
    return frames


class TestHierarchicalSampler:
    def setup_method(self):
        self.sampler = HierarchicalSampler(siglip_encoder=None)

    def test_fill_coverage_gaps_adds_midpoint(self):
        frames = make_frames(60)  # 60 frames, 1s apart
        selected = [frames[0], frames[59]]  # only first and last
        filled = self.sampler._fill_coverage_gaps(frames, selected, max_gap_sec=30.0)
        # Gap is 59s > 30s, should add a midpoint
        assert len(filled) > 2

    def test_select_keyframes_respects_budget(self):
        frames = make_frames(30)
        scores = np.random.rand(30)
        kf = self.sampler._select_keyframes(frames, scores, k=5, max_gap_sec=100.0,
                                            min_gap_sec=0.0)
        assert len(kf) <= 10  # k + possible coverage extras

    def test_select_keyframes_sorted(self):
        frames = make_frames(20)
        scores = np.random.rand(20)
        kf = self.sampler._select_keyframes(frames, scores, k=5, max_gap_sec=100.0)
        timestamps = [f.timestamp for f in kf]
        assert timestamps == sorted(timestamps)

    def test_histogram_diff_identical_images(self):
        import cv2
        from PIL import Image
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        diff = self.sampler._histogram_diff(img, img)
        assert diff < 0.01  # identical images → near-zero difference

    def test_rebalance_trims_to_budget(self):
        frames = make_frames(200)
        scores = np.random.rand(200)
        trimmed = self.sampler._rebalance(frames, scores, frames, budget=50)
        assert len(trimmed) == 50

    def test_rebalance_no_trim_when_under_budget(self):
        frames = make_frames(30)
        scores = np.random.rand(30)
        result = self.sampler._rebalance(frames, scores, frames, budget=500)
        assert len(result) == 30
