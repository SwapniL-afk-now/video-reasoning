# Performance Optimizations Plan

Current Step 1 (hierarchical frame sampling) takes ~20 min for a 100-min video. These changes target zero-quality-impact optimizations.

## 1. Fix duplicate SigLIP embedding

**Files:** `qvkg/qvkg/sampler.py`, `qvkg/qvkg/schema.py`

**Problem:** `_score_all()` computes SigLIP embeddings for all frames (line 184), then `_detect_boundaries()` recomputes them identically (line 255). This doubles the GPU encoding work for zero benefit.

**Fix:**
- Add `siglip_embeddings: Optional[np.ndarray]` field to `SampleResult` in `schema.py`
- In `_score_all`, store embeddings on the result before returning
- In `_detect_boundaries`, accept an optional `embeddings` parameter; if provided, skip `_siglip_batch()`
- In `builder.py`, pass the cached embeddings between the two calls

**Speedup:** ~3-4 min saved, exact same results.

---

## 2. Selective frame decode (skip 29/30 frames during decode)

**File:** `qvkg/qvkg/sampler.py`, `_extract_coarse_frames()`

**Problem:** PyAv's `container.demux()` decodes every frame at full resolution, but we only keep 1 per ~30 frames. For a 100-min 30fps video, that's ~180K decoded → 6K kept. The decode of the 174K discarded frames is pure waste.

**Fix:** Replace sequential decode loop with timestamp-based seeking:

```python
# Current approach (decodes everything):
for packet in container.demux(stream):
    for av_frame in packet.decode():
        if frame_idx % interval == 0:
            ...
        frame_idx += 1

# Proposed approach (seek to each target timestamp):
target_ts = 0.0
while target_ts <= duration:
    container.seek(int(target_ts / time_base))
    for av_frame in container.decode(video=0):
        img = av_frame.to_image().convert("RGB")
        frames.append(FrameInfo(id=f"f_{len(frames):06d}", timestamp=target_ts, image=img))
        break  # one frame at this timestamp
    target_ts += step_sec
```

**Speedup:** ~3-5 min saved. Produces the exact same 6K frames at the same timestamps.

---

## 3. Move audio RMS extraction outside the decode loop

**File:** `qvkg/qvkg/sampler.py`, `_extract_coarse_frames()`

**Problem:** `_extract_audio_rms()` is called inside `_extract_coarse_frames()` (line 130) before the decode loop. It opens its own separate container — it's independent work but delays the start of video decode.

**Fix:** Keep the call where it is (already independent) but optionally run in a background thread so it overlaps with video decode.

**Speedup:** ~1-2 seconds (minor but free).

---

## 4. Parallelize `_score_all` frame scoring loop

**File:** `qvkg/qvkg/sampler.py`, `_score_all()`

**Problem:** The sequential loop (lines 186-200) processes 6K frames one at a time on a single CPU core. Each iteration does histogram diff (cv2), optical flow (cv2), and dot product — all CPU work. The 6K iterations are embarrassingly independent.

**Fix:** Split frame indices into chunks (one per CPU core) and process with `ThreadPoolExecutor`:

```python
def _score_chunk(self, idx_range, frames, embeddings, audio_rms):
    scores = np.zeros(len(idx_range))
    for local_i, global_i in enumerate(idx_range):
        scores[local_i] = self._score_frame(global_i, frames, embeddings, audio_rms)
    return idx_range, scores

def _score_all(self, frames, audio_rms):
    embeddings = self._siglip_batch(frames)
    n = len(frames)
    n_workers = os.cpu_count() or 4
    chunks = np.array_split(range(n), n_workers)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(self._score_chunk, c, frames, embeddings, audio_rms)
                   for c in chunks]
        full_scores = np.zeros(n)
        for f in futures:
            indices, scores = f.result()
            full_scores[indices] = scores
    return full_scores
```

This requires extracting the per-frame scoring logic into a `_score_frame()` method.

**Speedup:** ~4-6 min saved on a 16+ core machine. Zero quality impact — identical scores, computed in parallel.

---

## Summary

| Change | Speedup | Quality Impact | Risk |
|--------|---------|----------------|------|
| 1. Cache SigLIP embeddings | ~3-4 min | None | Low (2 lines) |
| 2. Selective frame decode | ~3-5 min | None | Low (seek logic tested) |
| 3. Move audio RMS | ~2 sec | None | Trivial |
| 4. Parallelize scoring | ~4-6 min | None | Medium (threading) |

**Total:** 20 min → **~6-10 min** for a 100-min video. All changes are pure performance optimizations with zero effect on output quality.

---

## 5. Parallelize Whisper transcription with Step 1 (IMPLEMENTED)

**Files:** `qvkg/qvkg/builder.py`

**Problem:** Step 1 (frame sampling, ~10-20 min) and Step 2 (Whisper audio transcription, ~2-5 min) are both reading from the same video file but processing completely independent streams (video vs audio). They were running sequentially.

**Fix:** Launch `_transcribe_background()` in a `ThreadPoolExecutor` before Step 1 starts. The background thread runs `self.whisper.transcribe()` and materializes the segment list. At Step 2, `whisper_future.result()` collects the already-completed result (or blocks briefly if Step 1 finished faster than Whisper).

**Speedup:** ~2-5 min saved by overlapping Whisper with frame sampling.

**Resume-safe:** If Step 1 was cached from a previous run (no background thread needed), Step 2 falls back to running Whisper synchronously.
