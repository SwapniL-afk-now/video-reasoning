from __future__ import annotations

import base64
import io
import os
from typing import List, Optional

import h5py
import numpy as np
from PIL import Image

from .schema import FrameInfo


class FrameStore:
    """HDF5-backed keyframe storage.

    Stores frames as JPEG-compressed bytes under dataset key = frame_id.
    """

    def __init__(self, output_dir: str, mode: str = "a"):
        os.makedirs(output_dir, exist_ok=True)
        self.h5_path = os.path.join(output_dir, "frames.h5")
        self._mode = mode

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_keyframes(self, frames: List[FrameInfo], jpeg_quality: int = 85) -> None:
        with h5py.File(self.h5_path, "a") as f:
            for frame in frames:
                if frame.id in f:
                    continue
                img = frame.image
                if img is None:
                    continue
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=jpeg_quality)
                encoded = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                ds = f.create_dataset(frame.id, data=encoded)
                ds.attrs["timestamp"] = frame.timestamp

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, frame_id: str) -> Optional[FrameInfo]:
        if not os.path.exists(self.h5_path):
            return None
        with h5py.File(self.h5_path, "r") as f:
            if frame_id not in f:
                return None
            ds = f[frame_id]
            raw = bytes(np.array(ds))
            timestamp = float(ds.attrs.get("timestamp", 0.0))
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return FrameInfo(id=frame_id, timestamp=timestamp, image=img)

    def load_image(self, frame_id: str) -> Optional[Image.Image]:
        fi = self.load(frame_id)
        return fi.image if fi else None

    def get_b64_url(self, frame_id: str) -> str:
        """Return a data: URI suitable for vLLM image_url content."""
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(self.h5_path)
        with h5py.File(self.h5_path, "r") as f:
            if frame_id not in f:
                raise KeyError(frame_id)
            raw = bytes(np.array(f[frame_id]))
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def list_frame_ids(self) -> List[str]:
        if not os.path.exists(self.h5_path):
            return []
        with h5py.File(self.h5_path, "r") as f:
            return list(f.keys())

    def __contains__(self, frame_id: str) -> bool:
        if not os.path.exists(self.h5_path):
            return False
        with h5py.File(self.h5_path, "r") as f:
            return frame_id in f
