from __future__ import annotations

"""Hierarchical frame sampler: 4-level importance-scored selection."""

import io
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import av
import cv2
import numpy as np
from PIL import Image

from .schema import Episode, FrameInfo, SampleResult, Scene


# ---------------------------------------------------------------------------
# Module-level SigLIP worker — must be at module scope for multiprocessing
# ---------------------------------------------------------------------------

def _siglip_chunk_worker(jpeg_bytes_list: list, model_name: str, device: str) -> np.ndarray:
    """Worker process: load SigLIP, encode chunk, exit (GPU memory freed on exit)."""
    import io
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoProcessor, SiglipModel

    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device).eval()

    images = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpeg_bytes_list]

    all_embs: list = []
    batch_size = 64  # smaller per-worker since GPU is shared across workers
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output[1]
            features = features / features.norm(dim=-1, keepdim=True)
            all_embs.append(features.cpu().float().numpy())

    # Explicit cleanup before process exits
    del model
    torch.cuda.empty_cache()

    return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 1152))


# ---------------------------------------------------------------------------
# Module-level Whisper worker
# ---------------------------------------------------------------------------

def _whisper_transcribe_worker(
    video_path: str,
    model_size: str,
    compute_type: str = "int8_float16",
    initial_prompt: Optional[str] = None,
) -> list:
    """Worker process: load Whisper, transcribe, return segments, exit.

    Tuned for throughput (§5): VAD skips silence, greedy (beam_size=1) decoding,
    int8_float16 compute, and condition_on_previous_text=False to avoid
    drift-induced re-decodes. The transcript feeds graph nodes, not a leaderboard."""
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
    except Exception:
        # Some GPUs/builds reject int8_float16 — fall back to float16.
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(
        video_path,
        word_timestamps=True,
        vad_filter=True,
        beam_size=1,
        condition_on_previous_text=False,
        # Proper-noun vocabulary hint (names from question metadata / OCR):
        # Whisper hallucinates hardest on names, and graph entity linking
        # depends on getting them right.
        initial_prompt=initial_prompt,
    )
    result = list(segments)
    del model
    return result


class HierarchicalSampler:
    def __init__(self, siglip_encoder=None, yolo_model=None):
        self.siglip = siglip_encoder
        self.yolo = yolo_model  # optional YOLOv8n for object count delta scoring
        # Scoring-stage config + per-frame caches (set in sample()).
        self.flow_max_dim: int = 256
        self.use_optical_flow: bool = True
        self._gray_small: Optional[list] = None   # downscaled grayscale per frame idx
        self._hist: Optional[list] = None         # normalized 8x8x8 histogram per frame idx

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def sample(self, video_path: str, budget: int = 500,
               hard_thresh: float = 0.75, soft_thresh: float = 0.5,
                n_workers: int = 8, coarse_fps: float = 1.0,
                flow_max_dim: int = 256, use_optical_flow: bool = True,
                coarse_frame_cap: int = 0) -> SampleResult:
        import time as _time

        # Config for the CPU scoring stage (read by _score_frame / caches).
        self.flow_max_dim = int(flow_max_dim) if flow_max_dim else 256
        self.use_optical_flow = bool(use_optical_flow)

        t0 = _time.time()
        coarse, fps, audio_rms = self._extract_coarse_frames(
            video_path, target_fps=coarse_fps, n_workers=n_workers,
            coarse_frame_cap=coarse_frame_cap,
        )
        print(f"  [timing] step=decode+audio t={_time.time()-t0:.1f}s")
        if not coarse:
            return SampleResult([], [], [])

        print(f"  Extracted {len(coarse)} coarse frames at ~{coarse_fps}fps")

        # Cache per-frame grayscale (downscaled) + colour histogram ONCE so the
        # scoring + boundary passes never recompute cvtColor / calcHist (§2).
        t1 = _time.time()
        self._gray_small, self._hist = self._precompute_caches(coarse)
        print(f"  [timing] step=cache(gray+hist) t={_time.time()-t1:.1f}s")

        t2 = _time.time()
        siglip_embs = self._siglip_batch(coarse)
        print(f"  [timing] step=siglip t={_time.time()-t2:.1f}s")

        t3 = _time.time()
        scores = self._score_all(coarse, audio_rms, siglip_embs)
        print(f"  [timing] step=score t={_time.time()-t3:.1f}s")

        t4 = _time.time()
        boundaries = self._detect_boundaries(coarse, scores, hard_thresh, soft_thresh,
                                              embeddings=siglip_embs)
        print(f"  [timing] step=boundaries t={_time.time()-t4:.1f}s")
        print(f"  Detected {len(boundaries)} scenes")

        keyframes: List[FrameInfo] = []
        for scene in boundaries:
            sc_frames = coarse[scene._start_idx:scene._end_idx + 1]
            sc_scores = scores[scene._start_idx:scene._end_idx + 1]
            duration = scene.t_end - scene.t_start
            # 1 keyframe per 5s — dense enough to catch 2-3s chyrons/dialogue moments
            k = max(2, int(duration / 5))
            kf = self._select_keyframes(sc_frames, sc_scores, k,
                                        max_gap_sec=10.0)
            scene.keyframes = kf
            keyframes.extend(kf)

        keyframes = self._rebalance(keyframes, scores, coarse, budget)
        print(f"  Selected {len(keyframes)} keyframes (budget={budget})")
        print(f"  [timing] step=sample_total t={_time.time()-t0:.1f}s")
        # Release per-frame caches.
        self._gray_small = None
        self._hist = None
        return SampleResult(
            keyframes=keyframes, scenes=boundaries, episodes=[],
            siglip_embeddings=siglip_embs,
            audio_rms=audio_rms,
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

    def _extract_chunk_frames(
        self, video_path: str, t_start: float, t_end: float,
        target_fps: float, chunk_idx: int,
    ) -> List[FrameInfo]:
        """Extract frames from [t_start, t_end] at target_fps. Thread-safe."""
        frames: List[FrameInfo] = []
        container = av.open(video_path)
        stream = container.streams.video[0]
        time_base = stream.time_base

        step_sec = 1.0 / target_fps
        target_ts = t_start
        frame_num = 0
        while target_ts <= t_end:
            pts = int(target_ts / float(time_base))
            container.seek(pts, stream=stream)
            for av_frame in container.decode(video=0):
                # Retry reformat on EAGAIN (Errno 11) — can occur under heavy
                # parallel load when libswscale's internal buffers are briefly
                # unavailable.
                for _attempt in range(5):
                    try:
                        img = av_frame.to_image().convert("RGB")
                        break
                    except BlockingIOError:
                        import time as _time
                        _time.sleep(0.05 * (2 ** _attempt))
                else:
                    img = av_frame.to_image().convert("RGB")  # final raise
                frames.append(FrameInfo(
                    id=f"f_c{chunk_idx}_{frame_num:05d}",
                    timestamp=target_ts,
                    image=img,
                ))
                frame_num += 1
                break
            target_ts += step_sec

        container.close()
        return frames

    def _extract_coarse_frames(
        self, video_path: str, target_fps: float = 3.0, n_workers: int = 8,
        coarse_frame_cap: int = 0,
    ) -> Tuple[List[FrameInfo], float, np.ndarray]:
        # Probe duration and fps first (cheap, no frame decode)
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 25)
        duration = float(stream.duration * stream.time_base) if stream.duration else 0
        container.close()

        # Adaptive coarse sampling for very long videos (§6): cap the total
        # coarse frame count so every downstream CPU op stays bounded. The
        # effective fps tapers; question-aware densification (run later) keeps
        # referenced windows dense regardless.
        if coarse_frame_cap and duration > 0:
            max_fps = coarse_frame_cap / duration
            if max_fps < target_fps:
                print(f"  [sampler] coarse cap {coarse_frame_cap} frames over "
                      f"{duration:.0f}s → fps {target_fps:.2f}→{max_fps:.3f}")
                target_fps = max_fps

        audio_rms = self._extract_audio_rms(video_path)

        import time as _time
        if n_workers <= 1:
            print(f"  [sampler] sequential extraction (n_workers=1)")
            frames = self._extract_chunk_frames(video_path, 0.0, duration, target_fps, 0)
        else:
            chunk_dur = duration / n_workers
            chunks = [
                (i * chunk_dur, min((i + 1) * chunk_dur, duration), i)
                for i in range(n_workers)
            ]
            print(f"  [sampler] parallel extraction: {n_workers} workers × "
                  f"{chunk_dur:.0f}s chunks (video={duration:.0f}s)")
            t0 = _time.time()
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [
                    pool.submit(self._extract_chunk_frames,
                                video_path, t_s, t_e, target_fps, idx)
                    for t_s, t_e, idx in chunks
                ]
                chunk_results = [f.result() for f in futures]
            elapsed = _time.time() - t0
            per_chunk = [len(r) for r in chunk_results]
            print(f"  [sampler] workers done in {elapsed:.1f}s — "
                  f"frames per chunk: {per_chunk} (total={sum(per_chunk)})")
            frames = sorted(
                [fr for chunk in chunk_results for fr in chunk],
                key=lambda f: f.timestamp,
            )

        # Re-assign sequential IDs after merge
        for i, f in enumerate(frames):
            f.id = f"f_{i:06d}"

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
            sr = int(stream.sample_rate or 44100)
            sec = max(1, int(len(audio) / sr))
            rms = np.zeros(sec + 1)
            # Vectorized per-second RMS: reshape full seconds at once (§7).
            n_full = (len(audio) // sr)
            if n_full > 0:
                block = audio[:n_full * sr].reshape(n_full, sr)
                rms[:n_full] = np.sqrt((block.astype(np.float32) ** 2).mean(axis=1))
            # Trailing partial second.
            tail = audio[n_full * sr:]
            if n_full < sec and len(tail) > 0:
                rms[n_full] = float(np.sqrt(np.mean(tail.astype(np.float32) ** 2)))
            if rms.max() > 0:
                rms = rms / rms.max()
            return rms
        except Exception:
            return np.zeros(10000)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _precompute_caches(self, frames: List[FrameInfo]):
        """Compute downscaled grayscale + colour histogram once per frame.

        Reused by both scoring and boundary detection so cvtColor / calcHist
        run exactly once per frame instead of 2–3× (§2). Grayscale is downscaled
        to ``flow_max_dim`` on the long side so optical flow / text scoring are
        cheap (§1)."""
        max_dim = max(32, int(self.flow_max_dim))
        grays: list = []
        hists: list = []
        for f in frames:
            rgb = np.array(f.image)
            # Full-res colour histogram (8x8x8, normalized).
            h = cv2.calcHist([rgb], [0, 1, 2], None, [8, 8, 8],
                             [0, 256, 0, 256, 0, 256])
            hists.append(cv2.normalize(h, h).flatten())
            # Downscaled grayscale (long side ≤ max_dim) for flow + text.
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gh, gw = gray.shape
            scale = max_dim / float(max(gh, gw))
            if scale < 1.0:
                gray = cv2.resize(gray, (max(1, int(gw * scale)),
                                         max(1, int(gh * scale))))
            grays.append(gray)
        return grays, hists

    def _score_frame(self, i: int, frames: List[FrameInfo],
                     embeddings: Optional[np.ndarray],
                     audio_rms: np.ndarray) -> float:
        hist_d = self._histogram_diff_cached(i, i - 1) if i > 0 else 0.0
        sem_d = (1.0 - float(np.dot(embeddings[i], embeddings[i - 1]))) if i > 0 and embeddings is not None else 0.0
        ts = int(frames[i].timestamp)
        audio_e = float(audio_rms[min(ts, len(audio_rms) - 1)])
        text_s = self._text_region_score_cached(i)

        if self.use_optical_flow and i > 0:
            motion = self._optical_flow_mag_cached(i)
            # Text presence is weighted so frames bearing on-screen captions /
            # name lower-thirds are retained as keyframes.
            return (0.25 * hist_d + 0.25 * sem_d + 0.15 * motion
                    + 0.15 * audio_e + 0.20 * text_s)
        # Motion term dropped → renormalize remaining weights (§1).
        return (0.30 * hist_d + 0.30 * sem_d
                + 0.18 * audio_e + 0.22 * text_s)

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

    def _siglip_batch(self, frames: List[FrameInfo], gpu_batch: int = 384) -> Optional[np.ndarray]:
        """Encode all frames in one main-process SigLIP pass, in large GPU batches.

        SigLIP is GPU-bound and batches near-linearly; a single already-loaded
        model encoding hundreds of images per batch beats the previous N-process
        fan-out (which paid the model-load cost up to 8× and contended for one
        device). Runs under ``inference_mode`` + fp16 autocast.
        """
        if self.siglip is None:
            return None

        import torch

        try:
            images = [f.image for f in frames]
            n = len(images)
            dim = self.siglip.model.config.vision_config.hidden_size
            if n == 0:
                return np.zeros((0, dim))

            device = self.siglip.device
            use_autocast = device == "cuda"
            out: List[np.ndarray] = []
            print(f"  [siglip] single-pass encode: {n} frames "
                  f"in batches of {gpu_batch}")
            with torch.inference_mode():
                for i in range(0, n, gpu_batch):
                    batch = images[i:i + gpu_batch]
                    inputs = self.siglip.processor(
                        images=batch, return_tensors="pt"
                    ).to(device)
                    if use_autocast:
                        with torch.autocast("cuda", dtype=torch.float16):
                            output = self.siglip.model.get_image_features(**inputs)
                    else:
                        output = self.siglip.model.get_image_features(**inputs)
                    feats = (output.pooler_output
                             if hasattr(output, "pooler_output") else output[1])
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    out.append(feats.cpu().float().numpy())
            return np.concatenate(out, axis=0) if out else np.zeros((0, dim))

        except Exception:
            import traceback
            traceback.print_exc()
            return None

    # -- Cached scoring helpers (use the per-frame caches from _precompute_caches) --

    def _histogram_diff_cached(self, i: int, j: int) -> float:
        """Bhattacharyya distance between two cached histograms."""
        if self._hist is None:
            return 0.0
        return float(cv2.compareHist(self._hist[i], self._hist[j],
                                     cv2.HISTCMP_BHATTACHARYYA))

    def _optical_flow_mag_cached(self, i: int) -> float:
        """Farneback magnitude on the cached, downscaled grayscale pair."""
        if self._gray_small is None or i <= 0:
            return 0.0
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._gray_small[i - 1], self._gray_small[i], None,
                0.5, 3, 15, 3, 5, 1.2, 0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean()
            return min(1.0, float(mag) / 20.0)
        except Exception:
            return 0.0

    def _text_region_score_cached(self, i: int) -> float:
        if self._gray_small is None:
            return 0.0
        return self._text_region_score_gray(self._gray_small[i])

    def _histogram_diff(self, a: np.ndarray, b: np.ndarray) -> float:
        def hist(img):
            h = cv2.calcHist([img], [0, 1, 2], None,
                             [8, 8, 8], [0, 256, 0, 256, 0, 256])
            return cv2.normalize(h, h).flatten()
        return float(cv2.compareHist(hist(a), hist(b), cv2.HISTCMP_BHATTACHARYYA))

    def _text_region_score(self, img: np.ndarray) -> float:
        """RGB entry point — kept for callers; delegates to the grayscale core."""
        try:
            return self._text_region_score_gray(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        except Exception:
            return 0.0

    def _text_region_score_gray(self, gray: np.ndarray) -> float:
        """Heuristic on-screen-text density (0–1), biased to the lower third.

        Gradient → Otsu → horizontal morphological close groups character strokes
        into line-shaped blobs; wide/short blobs of moderate size are counted as
        text. Captions/lower-thirds (bottom 40%) are up-weighted. Cheap enough to
        run per coarse (~1fps) frame; returns 0.0 on any failure."""
        try:
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

        have_cache = self._hist is not None and len(self._hist) == len(frames)
        for i in range(1, len(frames)):
            if have_cache:
                hist_dist = self._histogram_diff_cached(i, i - 1)
            else:
                hist_dist = self._histogram_diff(
                    np.array(frames[i].image), np.array(frames[i - 1].image))

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
        selected_times: List[float] = []
        sorted_idx = np.argsort(scores)[::-1]

        for idx in sorted_idx:
            if len(selected) >= k:
                break
            t = frames[idx].timestamp
            # abs() gap: don't skip frames earlier in time than the last pick
            if selected_times and min(abs(t - st) for st in selected_times) < min_gap_sec:
                continue
            selected.append(frames[idx])
            selected_times.append(t)

        # Coverage pass: fill gaps > max_gap_sec (including leading/trailing)
        selected = self._fill_coverage_gaps(frames, selected, max_gap_sec)
        return sorted(selected, key=lambda f: f.timestamp)

    def _fill_coverage_gaps(
        self,
        all_frames: List[FrameInfo],
        selected: List[FrameInfo],
        max_gap_sec: float,
    ) -> List[FrameInfo]:
        if not all_frames:
            return selected

        sel = sorted(selected, key=lambda f: f.timestamp)
        extra: List[FrameInfo] = []

        if not sel:
            extra.append(all_frames[len(all_frames) // 2])
            return extra

        # Build the boundary list: scene_start, all selected, scene_end
        # then iteratively fill any gap > max_gap_sec until none remain.
        boundary_start = all_frames[0].timestamp
        boundary_end   = all_frames[-1].timestamp
        frame_by_t = {f.timestamp: f for f in all_frames}
        all_ts_sorted = sorted(frame_by_t)

        result = {f.timestamp: f for f in sel}

        changed = True
        while changed:
            changed = False
            sorted_ts = sorted(result)
            # Check leading gap
            checkpoints = (
                [(boundary_start, sorted_ts[0])]
                + [(sorted_ts[i], sorted_ts[i + 1]) for i in range(len(sorted_ts) - 1)]
                + [(sorted_ts[-1], boundary_end)]
            )
            for lo, hi in checkpoints:
                if hi - lo > max_gap_sec:
                    mid_t = (lo + hi) / 2
                    best_t = min(all_ts_sorted, key=lambda t: abs(t - mid_t))
                    if best_t not in result:
                        result[best_t] = frame_by_t[best_t]
                        changed = True

        return list(result.values())

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
