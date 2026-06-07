from __future__ import annotations

"""On-demand frame extraction from raw video at arbitrary timestamp windows."""

import base64
import io
from typing import List, Optional

import av
import cv2
import numpy as np
from PIL import Image

from ..schema import FrameInfo


def extract_frames_for_window(
    video_path: str,
    t_start: float,
    t_end: float,
    max_frames: int = 8,
    motion_rank: bool = False,
) -> List[FrameInfo]:
    """Extract frames from video within [t_start, t_end].

    Window strategy:
    - Pinpoint (span == 0): expand to [t-2s, t+2s], extract at 3fps
    - Short (span ≤ 30s): 3fps, cap at max_frames
    - Medium (30s–300s): subsample evenly to max_frames
    - Long (>300s): subsample evenly to max_frames (coarse)

    If motion_rank=True: score frames by optical flow, return top-k
    highest-motion frames instead of uniform/sequential selection.
    """
    span = t_end - t_start

    # Expand pinpoint windows
    if span < 1.0:
        t_start = max(0.0, t_start - 2.0)
        t_end   = t_end + 2.0
        span    = t_end - t_start

    # Determine target fps / frame count
    if span <= 30:
        target_fps = 3.0
    elif span <= 300:
        target_fps = max_frames / span   # evenly spaced
    else:
        target_fps = max_frames / span   # coarse

    frames = _extract_av(video_path, t_start, t_end, target_fps, max_frames)

    if motion_rank and len(frames) > max_frames:
        frames = _rank_by_motion(frames, max_frames)
    elif len(frames) > max_frames:
        # Uniform subsample
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]

    return frames


def frames_to_b64_urls(frames: List[FrameInfo]) -> List[str]:
    """Convert FrameInfo list to base64 data URIs for vLLM."""
    urls = []
    for f in frames:
        if f.image is None:
            continue
        img = f.image if isinstance(f.image, Image.Image) else Image.fromarray(f.image)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_av(
    video_path: str,
    t_start: float,
    t_end: float,
    target_fps: float,
    max_frames: int,
) -> List[FrameInfo]:
    frames: List[FrameInfo] = []
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONREF"  # faster seek

        # Seek to t_start
        seek_ts = int(t_start / stream.time_base)
        container.seek(seek_ts, stream=stream)

        frame_interval = 1.0 / max(target_fps, 0.1)
        last_accepted_t = -999.0
        fid = 0

        for packet in container.demux(stream):
            for av_frame in packet.decode():
                ts = float(av_frame.pts * stream.time_base) if av_frame.pts else 0.0
                if ts < t_start:
                    continue
                if ts > t_end:
                    break
                if ts - last_accepted_t < frame_interval * 0.9:
                    continue
                img = av_frame.to_image().convert("RGB")
                frames.append(FrameInfo(
                    id=f"live_{int(ts*1000):010d}",
                    timestamp=ts,
                    image=img,
                ))
                last_accepted_t = ts
                fid += 1
                if len(frames) >= max_frames * 3:  # safety cap before motion ranking
                    break
            if ts > t_end:
                break

        container.close()
    except Exception as e:
        # Return empty on any video read failure
        pass

    return frames


def _rank_by_motion(frames: List[FrameInfo], top_k: int) -> List[FrameInfo]:
    """Return top_k highest-motion frames using optical flow magnitude."""
    if len(frames) <= top_k:
        return frames

    scores = []
    grays = [cv2.cvtColor(np.array(f.image), cv2.COLOR_RGB2GRAY) for f in frames]

    for i, frame in enumerate(frames):
        if i == 0:
            scores.append(0.0)
            continue
        try:
            flow = cv2.calcOpticalFlowFarneback(
                grays[i - 1], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = float(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
        except Exception:
            mag = 0.0
        scores.append(mag)

    ranked_idx = sorted(range(len(frames)), key=lambda i: scores[i], reverse=True)
    top_idx = sorted(ranked_idx[:top_k])   # restore chronological order
    return [frames[i] for i in top_idx]
