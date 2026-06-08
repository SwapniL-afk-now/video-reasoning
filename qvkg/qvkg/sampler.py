from __future__ import annotations

"""Hierarchical frame sampler: 4-level importance-scored selection."""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np
from PIL import Image

from .schema import Episode, FrameInfo, SampleResult, Scene


class HierarchicalSampler:
    def __init__(self, siglip_encoder=None, yolo_model=None):
        self.siglip = siglip_encoder
        self.yolo = yolo_model  # optional YOLOv8n for object count delta scoring

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def sample(self, video_path: str, budget: int = 500,
               hard_thresh: float = 0.75, soft_thresh: float = 0.5) -> SampleResult:
        coarse, fps, audio_rms = self._extract_coarse_frames(video_path)
        if not coarse:
            return SampleResult([], [], [])

        print(f"  Extracted {len(coarse)} coarse frames at ~3fps")
        siglip_embs = self._siglip_batch(coarse)
        scores = self._score_all(coarse, audio_rms, siglip_embs)
        boundaries = self._detect_boundaries(coarse, scores, hard_thresh, soft_thresh,
                                              embeddings=siglip_embs)
        print(f"  Detected {len(boundaries)} scenes")

        keyframes: List[FrameInfo] = []
        for scene in boundaries:
            sc_frames = coarse[scene._start_idx:scene._end_idx + 1]
            sc_scores = scores[scene._start_idx:scene._end_idx + 1]
            duration = scene.t_end - scene.t_start
            k = max(2, int(duration / 15))
            kf = self._select_keyframes(sc_frames, sc_scores, k,
                                        max_gap_sec=30.0)
            scene.keyframes = kf
            keyframes.extend(kf)

        keyframes = self._rebalance(keyframes, scores, coarse, budget)
        print(f"  Selected {len(keyframes)} keyframes (budget={budget})")
        return SampleResult(
            keyframes=keyframes, scenes=boundaries, episodes=[],
            siglip_embeddings=siglip_embs,
        )

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def sample_question_windows(
        self,
        video_path: str,
        time_references: List[Tuple[float, float]],
        fps: float = 5.0,
        motion_rank: bool = False,
        max_frames_per_window: int = 16,
    ) -> List[FrameInfo]:
        """Dense sampling at benchmark question windows.

        For each (t_start, t_end):
        - Pinpoint (span==0): expand ±2s
        - ≤30s: sample at fps (default 5fps)
        - ≤300s: sample at 2fps
        - >300s: 8 evenly-spaced frames

        If motion_rank: rank frames within each window by optical flow magnitude.
        All returned FrameInfo objects have is_question_seeded attribute set via metadata.
        """
        from .query.frame_extractor import _extract_av, _rank_by_motion

        all_frames: List[FrameInfo] = []
        seen_ts: set = set()  # dedup by rounded timestamp

        for t_start, t_end in time_references:
            span = t_end - t_start

            # Expand pinpoint
            if span < 1.0:
                t_start = max(0.0, t_start - 2.0)
                t_end   = t_end + 2.0
                span    = t_end - t_start

            if span <= 30:
                target_fps   = fps
                max_f        = max_frames_per_window
            elif span <= 300:
                target_fps   = 2.0
                max_f        = max_frames_per_window
            else:
                target_fps   = 8.0 / max(span, 1)
                max_f        = 8

            window_frames = _extract_av(video_path, t_start, t_end, target_fps, max_f * 3)

            if motion_rank and len(window_frames) > max_f:
                window_frames = _rank_by_motion(window_frames, max_f)
            elif len(window_frames) > max_f:
                step = len(window_frames) / max_f
                window_frames = [window_frames[int(i * step)] for i in range(max_f)]

            for f in window_frames:
                key = round(f.timestamp, 1)
                if key not in seen_ts:
                    seen_ts.add(key)
                    f.metadata = getattr(f, "metadata", {})  # type: ignore
                    # Tag frame so builder can mark is_question_seeded
                    f.id = f"qs_{f.id}"
                    all_frames.append(f)

        return sorted(all_frames, key=lambda f: f.timestamp)

    def _extract_coarse_frames(
        self, video_path: str, target_fps: float = 3.0
    ) -> Tuple[List[FrameInfo], float, np.ndarray]:
        frames: List[FrameInfo] = []
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0

        # Extract audio RMS per second
        audio_rms = self._extract_audio_rms(video_path)

        step_sec = 1.0 / target_fps
        time_base = stream.time_base
        target_ts = 0.0
        while target_ts <= duration:
            pts = int(target_ts / float(time_base))
            container.seek(pts, stream=stream)
            for av_frame in container.decode(video=0):
                img = av_frame.to_image().convert("RGB")
                fid = f"f_{len(frames):06d}"
                frames.append(FrameInfo(id=fid, timestamp=target_ts, image=img))
                break
            target_ts += step_sec

        container.close()
        return frames, fps, audio_rms

    def _extract_audio_rms(self, video_path: str) -> np.ndarray:
        """Return per-second audio RMS array (zeros if no audio track)."""
        try:
            container = av.open(video_path)
            if not container.streams.audio:
                container.close()
                return np.zeros(10000)
            stream = container.streams.audio[0]
            samples = []
            for packet in container.demux(stream):
                for frame in packet.decode():
                    arr = frame.to_ndarray().astype(np.float32)
                    samples.append(arr.flatten())
            container.close()
            if not samples:
                return np.zeros(10000)
            audio = np.concatenate(samples)
            sr = stream.sample_rate or 44100
            sec = max(1, int(len(audio) / sr))
            rms = np.zeros(sec + 1)
            for i in range(sec):
                chunk = audio[i * sr:(i + 1) * sr]
                if len(chunk) > 0:
                    rms[i] = float(np.sqrt(np.mean(chunk ** 2)))
            if rms.max() > 0:
                rms = rms / rms.max()
            return rms
        except Exception:
            return np.zeros(10000)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_frame(self, i: int, frames: List[FrameInfo],
                     embeddings: Optional[np.ndarray],
                     audio_rms: np.ndarray) -> float:
        prev_img = np.array(frames[i - 1].image) if i > 0 else np.array(frames[i].image)
        curr_img = np.array(frames[i].image)

        hist_d = self._histogram_diff(curr_img, prev_img)
        sem_d = (1.0 - float(np.dot(embeddings[i], embeddings[i - 1]))) if i > 0 and embeddings is not None else 0.0
        motion = self._optical_flow_mag(prev_img, curr_img)
        ts = int(frames[i].timestamp)
        audio_e = float(audio_rms[min(ts, len(audio_rms) - 1)])
        text_s = self._text_region_score(curr_img)

        # Text presence is weighted so frames bearing on-screen captions / name
        # lower-thirds are retained as keyframes and reach the OCR/VLM stage —
        # otherwise short captions (e.g. a guest's name) never enter the graph.
        return (0.25 * hist_d + 0.25 * sem_d + 0.15 * motion
                + 0.15 * audio_e + 0.20 * text_s)

    def _score_all(
        self, frames: List[FrameInfo], audio_rms: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = len(frames)
        scores = np.zeros(n)
        n_workers = min(os.cpu_count() or 4, n)
        chunk_size = max(1, n // n_workers)
        chunks = [range(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = []
            for c in chunks:
                futures.append(pool.submit(
                    self._score_chunk, c, frames, embeddings, audio_rms
                ))
            for f in futures:
                indices, chunk_scores = f.result()
                scores[list(indices)] = chunk_scores
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def _score_chunk(self, idx_range, frames, embeddings, audio_rms):
        chunk_scores = np.zeros(len(idx_range))
        for local_i, global_i in enumerate(idx_range):
            chunk_scores[local_i] = self._score_frame(global_i, frames, embeddings, audio_rms)
        return idx_range, chunk_scores

    def _siglip_batch(self, frames: List[FrameInfo]) -> Optional[np.ndarray]:
        if self.siglip is None:
            return None
        try:
            images = [f.image for f in frames]
            batch_size = 32
            all_embs = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                embs = self.siglip.encode_images_batch(batch)
                all_embs.append(embs)
            return np.concatenate(all_embs, axis=0)
        except Exception:
            return None

    def _histogram_diff(self, a: np.ndarray, b: np.ndarray) -> float:
        def hist(img):
            h = cv2.calcHist([img], [0, 1, 2], None,
                             [8, 8, 8], [0, 256, 0, 256, 0, 256])
            return cv2.normalize(h, h).flatten()
        return float(cv2.compareHist(hist(a), hist(b), cv2.HISTCMP_BHATTACHARYYA))

    def _text_region_score(self, img: np.ndarray) -> float:
        """Heuristic on-screen-text density (0–1), biased to the lower third.

        Gradient → Otsu → horizontal morphological close groups character strokes
        into line-shaped blobs; wide/short blobs of moderate size are counted as
        text. Captions/lower-thirds (bottom 40%) are up-weighted. Cheap enough to
        run per coarse (~1fps) frame; returns 0.0 on any failure."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            scale = 320.0 / max(1, w)
            if scale < 1.0:
                gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
            H, W = gray.shape

            grad = np.absolute(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
            grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            text_area = 0.0
            for c in contours:
                x, y, cw, ch = cv2.boundingRect(c)
                aspect = cw / max(1, ch)
                area = cw * ch
                if (aspect >= 2.5 and 6 <= ch <= 0.20 * H
                        and area >= 0.0008 * H * W):
                    weight = 1.5 if y >= 0.60 * H else 1.0  # favour lower-thirds
                    text_area += area * weight
            return min(1.0, (text_area / float(H * W)) * 6.0)
        except Exception:
            return 0.0

    def _optical_flow_mag(self, prev: np.ndarray, curr: np.ndarray) -> float:
        try:
            p = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            c = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                p, c, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
            return min(1.0, float(mag) / 20.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Scene boundary detection
    # ------------------------------------------------------------------

    def _detect_boundaries(
        self,
        frames: List[FrameInfo],
        scores: np.ndarray,
    hard_thresh: float = 0.75,
    soft_thresh: float = 0.5,
    embeddings: Optional[np.ndarray] = None,
    ) -> List[Scene]:
        if len(frames) == 0:
            return []

        if embeddings is None:
            embeddings = self._siglip_batch(frames)
        scene_start = 0
        boundaries: List[Scene] = []

        for i in range(1, len(frames)):
            curr_img = np.array(frames[i].image)
            prev_img = np.array(frames[i - 1].image)
            hist_dist = self._histogram_diff(curr_img, prev_img)

            emb_dist = 0.0
            if embeddings is not None:
                emb_dist = float(1.0 - np.dot(embeddings[i], embeddings[i - 1]))

            is_hard = hist_dist > hard_thresh
            is_soft = emb_dist > soft_thresh and hist_dist > 0.3

            if is_hard or is_soft:
                sc = self._make_scene(frames, scene_start, i - 1)
                boundaries.append(sc)
                scene_start = i

        # Final scene
        boundaries.append(self._make_scene(frames, scene_start, len(frames) - 1))
        return boundaries

    def _make_scene(
        self, frames: List[FrameInfo], start_idx: int, end_idx: int
    ) -> Scene:
        sc = Scene(
            id=f"scene_{start_idx:04d}",
            t_start=frames[start_idx].timestamp,
            t_end=frames[end_idx].timestamp,
        )
        sc._start_idx = start_idx  # type: ignore[attr-defined]
        sc._end_idx = end_idx      # type: ignore[attr-defined]
        return sc

    # ------------------------------------------------------------------
    # Keyframe selection per scene
    # ------------------------------------------------------------------

    def _select_keyframes(
        self,
        frames: List[FrameInfo],
        scores: np.ndarray,
        k: int,
        max_gap_sec: float = 30.0,
        min_gap_sec: float = 5.0,
    ) -> List[FrameInfo]:
        if not frames:
            return []

        selected: List[FrameInfo] = []
        last_t = -9999.0
        sorted_idx = np.argsort(scores)[::-1]

        for idx in sorted_idx:
            if len(selected) >= k:
                break
            t = frames[idx].timestamp
            if t - last_t < min_gap_sec:
                continue
            selected.append(frames[idx])
            last_t = t

        # Coverage pass: fill gaps > max_gap_sec
        selected = self._fill_coverage_gaps(frames, selected, max_gap_sec)
        return sorted(selected, key=lambda f: f.timestamp)

    def _fill_coverage_gaps(
        self,
        all_frames: List[FrameInfo],
        selected: List[FrameInfo],
        max_gap_sec: float,
    ) -> List[FrameInfo]:
        sel = sorted(selected, key=lambda f: f.timestamp)
        extra: List[FrameInfo] = []

        if not sel and all_frames:
            extra.append(all_frames[len(all_frames) // 2])

        for i in range(len(sel) - 1):
            gap = sel[i + 1].timestamp - sel[i].timestamp
            if gap > max_gap_sec:
                mid_t = (sel[i].timestamp + sel[i + 1].timestamp) / 2
                best = min(all_frames, key=lambda f: abs(f.timestamp - mid_t))
                extra.append(best)

        return sel + extra

    # ------------------------------------------------------------------
    # Budget rebalancing
    # ------------------------------------------------------------------

    def _rebalance(
        self,
        keyframes: List[FrameInfo],
        scores: np.ndarray,
        coarse: List[FrameInfo],
        budget: int,
    ) -> List[FrameInfo]:
        if len(keyframes) <= budget:
            return keyframes
        # Build score map by frame id
        score_map = {f.id: float(scores[i]) for i, f in enumerate(coarse)}
        sorted_kf = sorted(keyframes,
                           key=lambda f: score_map.get(f.id, 0.0),
                           reverse=True)
        return sorted(sorted_kf[:budget], key=lambda f: f.timestamp)
