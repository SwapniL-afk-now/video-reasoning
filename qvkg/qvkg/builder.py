from __future__ import annotations

"""VKGBuilder: 11-step offline VKG construction orchestrator."""

import json
import os
import pickle
import re
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .causal import build_causal_request, parse_causal_edges
from .character import (
    CharacterMention, DescriptionBasedCharacterResolver, LLMEntityResolver,
    ObjectIdentityResolver,
)
from .episode import segment_episodes
from .extraction import SceneData, run_scene_extraction
from .faiss_index import build_faiss_index, build_semantic_edges_faiss
from .frame_store import FrameStore
from .sampler import HierarchicalSampler
from .schema import (
    Episode, FrameInfo, Scene, VKGEdge, VKGNode, VKGraph,
)
from .subtitle import discover_subtitle_path, parse_subtitle_file
from .vllm_client import CAUSAL_SAMPLING, VIDEO_TYPE_SAMPLING, VIDEO_TYPE_SYSTEM_PROMPT


def _checkpoint_path(output_dir: str, step: str) -> str:
    return os.path.join(output_dir, f".ckpt_{step}")


def _is_checkpoint(output_dir: str, step: str) -> bool:
    return os.path.exists(_checkpoint_path(output_dir, step))


def _write_checkpoint(output_dir: str, step: str):
    open(_checkpoint_path(output_dir, step), "w").close()


def _read_pickle(output_dir: str, name: str):
    path = os.path.join(output_dir, name)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _write_pickle(output_dir: str, name: str, obj):
    path = os.path.join(output_dir, name)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class VKGBuilder:
    def __init__(self, llm, whisper_model, siglip_encoder, config: dict):
        self.llm = llm
        self.whisper = whisper_model
        self.siglip = siglip_encoder
        self.config = config

    # ------------------------------------------------------------------
    # Background transcription: runs in a separate process, overlaps with Step 1
    # ------------------------------------------------------------------

    def _transcribe_background(self, video_path: str) -> Optional[List]:
        """Launch Whisper in an isolated worker process.

        Running in a separate process (not just a thread) means Whisper's
        model weights and CUDA allocations are fully released when the process
        exits, and GIL contention with the SigLIP step is eliminated.
        """
        import torch.multiprocessing as tmp
        from .sampler import _whisper_transcribe_worker

        whisper_model_size = getattr(self.whisper, "_model_size", "large-v3-turbo")
        whisper_compute_type = (getattr(self.whisper, "_compute_type", None)
                                or self.config.get("whisper_compute_type", "int8_float16"))
        try:
            ctx = tmp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as executor:
                asr_vocab = self.config.get("asr_vocab") or []
                initial_prompt = ("Names mentioned: " + ", ".join(asr_vocab) + ".") \
                    if asr_vocab else None
                future = executor.submit(
                    _whisper_transcribe_worker, video_path, whisper_model_size,
                    whisper_compute_type, initial_prompt,
                )
                return future.result()
        except Exception as e:
            raise RuntimeError(f"Whisper transcription failed: {e}")

    def build(self, video_path: str, output_dir: str, phase: str = "all"):
        """Build the VKG.

        phase='all'    — full pipeline (default)
        phase='siglip' — Steps 1+2+2b only (SigLIP + Whisper, no vLLM); returns None
        phase='llm'    — Steps 0+3-10 only (vLLM); assumes siglip phase already ran
        """
        os.makedirs(output_dir, exist_ok=True)
        graph = VKGraph()
        frame_store = FrameStore(output_dir)

        # ------------------------------------------------------------------
        # Steps 1 + 2 + 2b  (SigLIP + Whisper — no vLLM required)
        # Runs in 'all' and 'siglip' phases.
        # ------------------------------------------------------------------
        if phase in ("all", "siglip"):
            # Pre-launch Whisper in background thread — overlaps with Step 1
            whisper_future = None
            step1_cached = _is_checkpoint(output_dir, "step1_sampled")
            step2_cached = _is_checkpoint(output_dir, "step2_transcribed")
            if not step2_cached and self.whisper is not None:
                pool = ThreadPoolExecutor(max_workers=1)
                whisper_future = pool.submit(self._transcribe_background, video_path)

            # Step 1: Hierarchical frame sampling
            sample = _read_pickle(output_dir, "sample.pkl")
            if step1_cached and sample is not None:
                print("Step 1: Hierarchical frame sampling [CACHED]")
                print(f"  Loaded {len(sample.keyframes)} keyframes, {len(sample.scenes)} scenes")
            else:
                print("Step 1: Hierarchical frame sampling...")
                sampler = HierarchicalSampler(self.siglip)
                sample = sampler.sample(
                    video_path,
                    budget=self.config.get("frame_budget", 500),
                    hard_thresh=self.config.get("hard_boundary_thresh", 0.75),
                    soft_thresh=self.config.get("soft_boundary_thresh", 0.5),
                    n_workers=self.config.get("n_chunk_workers", 8),
                    coarse_fps=self.config.get("coarse_fps", 1.0),
                    flow_max_dim=self.config.get("flow_max_dim", 256),
                    use_optical_flow=self.config.get("use_optical_flow", True),
                    coarse_frame_cap=self.config.get("coarse_frame_cap", 0),
                )

                question_time_refs = self.config.get("question_time_refs", [])
                if question_time_refs:
                    print(f"  Question-aware pre-sampling: {len(question_time_refs)} windows...")
                    motion_rank = self.config.get("video_type", "") in ("sport", "live")
                    qs_frames = sampler.sample_question_windows(
                        video_path, question_time_refs, fps=5.0, motion_rank=motion_rank
                    )
                    existing_ts = {round(f.timestamp, 1) for f in sample.keyframes}
                    new_frames = [f for f in qs_frames
                                  if round(f.timestamp, 1) not in existing_ts]
                    sample.keyframes.extend(new_frames)
                    sample.keyframes.sort(key=lambda f: f.timestamp)
                    print(f"  Added {len(new_frames)} question-seeded keyframes "
                          f"→ total {len(sample.keyframes)}")

                frame_store.save_keyframes(sample.keyframes)
                _write_pickle(output_dir, "sample.pkl", sample)
                _write_checkpoint(output_dir, "step1_sampled")
                print(f"  Saved {len(sample.keyframes)} keyframes, {len(sample.scenes)} scenes")

            # Step 2: Audio transcription
            speech_nodes = _read_pickle(output_dir, "speech_nodes.pkl")
            if _is_checkpoint(output_dir, "step2_transcribed") and speech_nodes is not None:
                print(f"Step 2: Audio transcription [CACHED] ({len(speech_nodes)} segments)")
                graph.add_nodes(speech_nodes)
            else:
                print("Step 2: Audio transcription (Whisper)...")
                speech_nodes = []
                try:
                    if whisper_future is not None:
                        segments = whisper_future.result()
                    elif self.whisper is not None:
                        segments, _ = self.whisper.transcribe(
                            video_path, word_timestamps=True
                        )
                        segments = list(segments)
                    else:
                        segments = None

                    if segments:
                        speech_nodes = self._build_speech_nodes(segments)
                        graph.add_nodes(speech_nodes)
                        _write_pickle(output_dir, "speech_nodes.pkl", speech_nodes)
                        _write_checkpoint(output_dir, "step2_transcribed")
                        print(f"  Transcribed {len(speech_nodes)} speech segments")
                except Exception as e:
                    print(f"  Whisper failed: {e} — skipping audio")

            # Step 2b: Subtitle-track ingestion
            subtitle_nodes = _read_pickle(output_dir, "subtitle_nodes.pkl")
            if _is_checkpoint(output_dir, "step2b_subtitles") and subtitle_nodes is not None:
                print(f"Step 2b: Subtitle ingestion [CACHED] ({len(subtitle_nodes)} cues)")
                graph.add_nodes(subtitle_nodes)
            else:
                subtitle_nodes = []
                sub_path = (self.config.get("subtitle_path")
                            or discover_subtitle_path(video_path))
                if sub_path:
                    print(f"Step 2b: Subtitle ingestion ({os.path.basename(sub_path)})...")
                    try:
                        segments = parse_subtitle_file(sub_path)
                        subtitle_nodes = self._build_subtitle_nodes(segments)
                        graph.add_nodes(subtitle_nodes)
                        _write_pickle(output_dir, "subtitle_nodes.pkl", subtitle_nodes)
                        _write_checkpoint(output_dir, "step2b_subtitles")
                        print(f"  Ingested {len(subtitle_nodes)} subtitle cues")
                    except Exception as e:
                        print(f"  Subtitle ingestion failed: {e} — skipping")
                else:
                    print("Step 2b: Subtitle ingestion — no subtitle track found, skipping")
            speech_nodes = (speech_nodes or []) + (subtitle_nodes or [])

            if phase == "siglip":
                print("SigLIP phase complete — checkpoints saved.")
                return None

        # ------------------------------------------------------------------
        # Step 0: Video type detection (skip if already set in config)
        # Runs in 'all' and 'llm' phases.
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Step 0: Video type detection
        # Runs in 'all' and 'llm' phases (needs vLLM).
        # ------------------------------------------------------------------
        if "video_type" not in self.config or not self.config["video_type"]:
            if _is_checkpoint(output_dir, "step0_videotype"):
                vt_meta = _read_pickle(output_dir, "video_type.pkl")
                if vt_meta:
                    self.config.update(vt_meta)
                    print(f"Step 0: Video type detection [CACHED] → {self.config.get('video_type')}")
            else:
                print("Step 0: Video type detection...")
                vt_meta = self._detect_video_type(video_path)
                self.config.update(vt_meta)
                _write_pickle(output_dir, "video_type.pkl", vt_meta)
                _write_checkpoint(output_dir, "step0_videotype")
                print(f"  Detected: {vt_meta.get('video_type')} "
                      f"(narrator={vt_meta.get('has_narrator_voiceover')}, "
                      f"multi-speaker={vt_meta.get('has_multiple_speakers')})")
        else:
            print(f"Step 0: Video type set by config → {self.config['video_type']}")

        # In 'llm' phase, load step1/2 results from checkpoints written by siglip phase.
        if phase == "llm":
            sample = _read_pickle(output_dir, "sample.pkl")
            if sample is None:
                raise RuntimeError(
                    f"llm phase requires step1 checkpoint — run siglip phase first for {video_path}"
                )
            print(f"Step 1: Hierarchical frame sampling [CACHED from siglip phase] "
                  f"({len(sample.keyframes)} keyframes)")
            speech_nodes = _read_pickle(output_dir, "speech_nodes.pkl") or []
            subtitle_nodes = _read_pickle(output_dir, "subtitle_nodes.pkl") or []
            speech_nodes = speech_nodes + subtitle_nodes
            graph.add_nodes(speech_nodes)

        # ------------------------------------------------------------------
        # Step 3: Scene extraction
        # ------------------------------------------------------------------
        scene_data = _read_pickle(output_dir, "scene_data.pkl")
        if _is_checkpoint(output_dir, "step3_extracted") and scene_data is not None:
            print(f"Step 3: Scene extraction [CACHED] ({len(scene_data)} scenes)")
        else:
            print("Step 3: Scene extraction (Qwen batch)...")
            scene_data = run_scene_extraction(
                sample.scenes, frame_store, self.llm,
                video_type=self.config.get("video_type", ""),
                has_narrator=self.config.get("has_narrator_voiceover", False),
            )
            _write_pickle(output_dir, "scene_data.pkl", scene_data)
            _write_checkpoint(output_dir, "step3_extracted")
            print(f"  Extracted data for {len(scene_data)} scenes")

        # ------------------------------------------------------------------
        # Episode segmentation (populates sample.episodes)
        # ------------------------------------------------------------------
        if _is_checkpoint(output_dir, "step_episodes") and sample.episodes:
            print(f"Episode segmentation [CACHED] ({len(sample.episodes)} episodes)")
        elif not sample.episodes:
            print("Episode segmentation (Qwen)...")
            try:
                episodes = segment_episodes(sample.scenes, scene_data, self.llm)
                sample.episodes = episodes
                _write_pickle(output_dir, "sample.pkl", sample)
                _write_checkpoint(output_dir, "step_episodes")
                print(f"  Segmented {len(episodes)} episodes")
            except Exception as e:
                print(f"  Episode segmentation failed: {e} — using flat structure")
                sample.episodes = []

        # Steps 4-8: Node creation, edges, character resolution
        checkpoint_graph = os.path.join(output_dir, "graph_ckpt.json")
        if _is_checkpoint(output_dir, "step8_graph_built") and os.path.exists(checkpoint_graph):
            print("Steps 4-8: Node/edge creation [CACHED]")
            graph = VKGraph.load(checkpoint_graph)
            # Re-add speech nodes that were added to graph in Step 2 but not persisted in graph checkpoint
            for sn in speech_nodes:
                if sn.id not in graph.nodes:
                    graph.add_node(sn)
            print(f"  Graph has {len(graph.nodes)} nodes, "
                  f"{sum(len(v) for v in graph.edges.values())} edges")
        else:
            # Step 4: Node creation
            print("Step 4: Node creation...")
            self._create_temporal_backbone(graph, sample, scene_data)
            self._create_entity_nodes(graph, scene_data, frame_store)
            self._create_event_nodes(graph, scene_data)
            self._create_perception_nodes(graph, scene_data, speech_nodes)
            print(f"  Graph has {len(graph.nodes)} nodes")

            # Step 5: Temporal + hierarchical edges
            print("Step 5: Temporal + hierarchical edges...")
            self._build_temporal_edges(graph)
            self._build_hierarchical_edges(graph, sample)

            # Step 6: Spatial edges
            print("Step 6: Spatial edges (from Qwen output)...")
            self._build_spatial_edges_from_extraction(graph, scene_data)

            # Step 7: Cross-modal edges
            print("Step 7: Cross-modal edges...")
            self._build_crossmodal_edges(graph)
            self._build_speaker_edges_from_extraction(graph, scene_data)
            self._build_ocr_semantic_edges(graph, scene_data)

            # Step 8: Character resolution (LLM-native, DBSCAN fallback)
            print("Step 8: Character resolution...")
            char_mentions = self._collect_character_mentions(graph, scene_data)
            llm_resolver = LLMEntityResolver()
            characters = llm_resolver.resolve(
                char_mentions, self.llm,
                video_type=self.config.get("video_type", ""),
            )
            if characters is None:
                print("  LLM resolution failed — falling back to DBSCAN")
                fallback = DescriptionBasedCharacterResolver()
                characters = fallback.resolve(char_mentions, self.siglip)
            if characters is not None:
                # Remove raw char_raw_* nodes — resolved chars replace them
                raw_ids = [nid for nid in list(graph.nodes.keys())
                           if nid.startswith("char_raw_")]
                for rid in raw_ids:
                    graph.nodes.pop(rid, None)
                    graph.type_idx["CharacterNode"] = [
                        nid for nid in graph.type_idx.get("CharacterNode", [])
                        if nid != rid
                    ]
                for char in characters:
                    if char.id in graph.nodes:
                        graph.nodes[char.id] = char
                    else:
                        graph.add_node(char)
                self._link_characters_to_events(graph, characters)
                self._build_same_entity_edges(graph, characters)
                # Re-run speaker edges now that resolved chars are in graph
                self._build_speaker_edges_from_extraction(graph, scene_data)
                print(f"  Resolved {len(characters)} characters, "
                      f"removed {len(raw_ids)} raw mentions")
            else:
                print("  Character resolution failed — keeping raw character mentions")

            # Step 8.0b: ground on-screen name banners into character identity.
            self._link_ocr_names_to_characters(graph)

            # Step 8.1: Object identity resolution
            print("Step 8.1: Object identity resolution...")
            obj_nodes = graph.get_nodes_by_type("ObjectNode")
            obj_resolver = ObjectIdentityResolver()
            obj_clusters = obj_resolver.resolve(obj_nodes, self.siglip)
            if obj_clusters:
                n_obj_edges = 0
                for cluster_idx, cluster in enumerate(obj_clusters):
                    entity_id = f"entity_obj_{cluster_idx}"
                    for node in cluster:
                        node.entity_id = entity_id
                    for i, a in enumerate(cluster):
                        for b in cluster[i + 1:]:
                            graph.add_edge(VKGEdge(
                                source_id=a.id, target_id=b.id,
                                relation_type="SAME_ENTITY", weight=1.0, confidence=0.85,
                                metadata={"source": "object_identity_resolver"},
                            ))
                            n_obj_edges += 1
                print(f"  Resolved {len(obj_clusters)} object identity groups, "
                      f"{n_obj_edges} SAME_ENTITY edges")
            else:
                print("  Object identity resolution skipped or no cross-scene objects found")

            # Step 8.2: Location nodes + TAKES_PLACE_IN edges
            print("Step 8.2: Location nodes...")
            self._build_location_nodes(graph, scene_data)

            # Step 8.3: Emotional arc edges (EMOTION_SHIFT)
            print("Step 8.3: Emotional arc edges...")
            n_emotion = self._build_emotion_shift_edges(graph)
            print(f"  Added {n_emotion} EMOTION_SHIFT edges")

            # Step 8.4: MENTIONS edges (speech → entity)
            print("Step 8.4: MENTIONS edges...")
            n_mentions = self._build_mentions_edges(graph, scene_data)
            print(f"  Added {n_mentions} MENTIONS edges")

            # Step 8.5: AudioEventNodes from RMS peaks
            if sample.audio_rms is not None:
                print("Step 8.5: Audio event nodes...")
                n_audio = self._build_audio_event_nodes(graph, sample.audio_rms)
                print(f"  Added {n_audio} AudioEventNodes")

            # Step 8.6: EpisodeNode participant lists
            self._build_episode_participant_lists(graph)

            graph.save(checkpoint_graph)
            _write_checkpoint(output_dir, "step8_graph_built")
            print(f"  Graph checkpoint saved ({len(graph.nodes)} nodes)")

        # Step 9: FAISS index
        index_path = os.path.join(output_dir, "vkg.index")
        if _is_checkpoint(output_dir, "step9_indexed") and os.path.exists(index_path):
            print("Step 9: FAISS index [CACHED]")
            from .faiss_index import load_faiss_index
            faiss_index = load_faiss_index(index_path)
            # Rebuild node_id_list if not persisted (matches build_faiss_index ordering)
            if not graph.node_id_list:
                graph.node_id_list = [n.id for n in graph.nodes.values()]
            print(f"  Loaded FAISS HNSW index over {len(graph.node_id_list)} nodes")
        else:
            print("Step 9: FAISS index...")
            faiss_index = build_faiss_index(graph, self.siglip, index_path, frame_store)
            _write_checkpoint(output_dir, "step9_indexed")
            print(f"  Built FAISS HNSW index over {len(graph.node_id_list)} nodes")

        # Step 10: Causal chain inference
        if _is_checkpoint(output_dir, "step10_causal"):
            print("Step 10: Causal chain inference [CACHED]")
            # Reload graph from checkpoint that includes causal edges
            if os.path.exists(checkpoint_graph):
                graph = VKGraph.load(checkpoint_graph)
        else:
            print("Step 10: Causal chain inference (Qwen batch)...")
            episodes = graph.get_episodes()
            if episodes:
                try:
                    causal_requests = [
                        build_causal_request(
                            _episode_node_to_episode(ep, graph),
                            graph, frame_store
                        )
                        for ep in episodes
                    ]
                    causal_outputs = self.llm.chat(
                        messages=[r["messages"] for r in causal_requests],
                        sampling_params=CAUSAL_SAMPLING,
                        use_tqdm=True,
                    )
                    total_causal = 0
                    for req, ep_node, out in zip(causal_requests, episodes, causal_outputs):
                        ep_obj = _episode_node_to_episode(ep_node, graph)
                        edges = parse_causal_edges(
                            out.outputs[0].text, graph, ep_obj
                        )
                        graph.add_edges(edges)
                        total_causal += len(edges)
                    print(f"  Added {total_causal} causal edges")
                except Exception as e:
                    print(f"  Step 10 skipped ({type(e).__name__}: {e})")
            graph.save(checkpoint_graph)
            _write_checkpoint(output_dir, "step10_causal")

        # Step 10.5: NARRATIVE_ARC edges between consecutive episodes
        episodes_sorted = graph.get_episodes()
        for i in range(len(episodes_sorted) - 1):
            graph.add_edge(VKGEdge(
                source_id=episodes_sorted[i].id,
                target_id=episodes_sorted[i + 1].id,
                relation_type="NARRATIVE_ARC",
                weight=1.0, confidence=1.0,
            ))

        # Step 11: Semantic edges
        if _is_checkpoint(output_dir, "step11_semantic"):
            print("Step 11: Semantic edges [CACHED]")
            graph = VKGraph.load(checkpoint_graph)
        else:
            print("Step 11: Semantic edges (FAISS ANN)...")
            n_sem = build_semantic_edges_faiss(
                graph, faiss_index,
                threshold=self.config.get("semantic_threshold", 0.78),
                k_neighbors=self.config.get("semantic_k_neighbors", 10),
            )
            print(f"  Added {n_sem} semantic SIMILAR_TO edges")
            graph.save(checkpoint_graph)
            _write_checkpoint(output_dir, "step11_semantic")

        # Compute temporal precision for all nodes (gap detection signal)
        graph.compute_temporal_precision()
        vkg_path = os.path.join(output_dir, "vkg.json")
        graph.save(vkg_path)

        # Save meta
        meta = {
            "video_path": video_path,
            "n_nodes": len(graph.nodes),
            "n_edges": sum(len(v) for v in graph.edges.values()),
            "n_keyframes": len(sample.keyframes),
            "n_scenes": len(sample.scenes),
            "config": self.config,
        }
        with open(os.path.join(output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"VKG built: {len(graph.nodes)} nodes, "
              f"{sum(len(v) for v in graph.edges.values())} edges")
        return graph

    # ------------------------------------------------------------------
    # Step 2 helper: speech nodes
    # ------------------------------------------------------------------

    def _build_speech_nodes(self, segments) -> List[VKGNode]:
        nodes = []
        for seg in segments:
            text = getattr(seg, "text", "").strip()
            if not text:
                continue
            node = VKGNode(
                id=f"speech_{len(nodes):05d}",
                node_type="SpeechNode",
                label=text,
                level=0,
                t_start=float(getattr(seg, "start", 0)),
                t_end=float(getattr(seg, "end", 0)),
                confidence=float(getattr(seg, "avg_logprob", 0.8) + 1.0),
                metadata={"source": "whisper"},
            )
            nodes.append(node)
        return nodes

    def _build_subtitle_nodes(self, segments) -> List[VKGNode]:
        """Turn parsed subtitle cues into SpeechNodes tagged source='subtitle'.

        These carry exact subtitle wording and timing — authoritative for
        text-referred questions — and are typed as SpeechNodes so existing
        scene-attribution and cross-modal edge logic picks them up.
        """
        nodes = []
        for seg in segments:
            text = getattr(seg, "text", "").strip()
            if not text:
                continue
            nodes.append(VKGNode(
                id=f"sub_{len(nodes):05d}",
                node_type="SpeechNode",
                label=text,
                level=0,
                t_start=float(getattr(seg, "start", 0)),
                t_end=float(getattr(seg, "end", 0)),
                confidence=1.0,
                metadata={"source": "subtitle"},
            ))
        return nodes

    # ------------------------------------------------------------------
    # Step 4 helpers: node creation
    # ------------------------------------------------------------------

    def _create_temporal_backbone(
        self,
        graph: VKGraph,
        sample,
        scene_data: Dict[str, SceneData],
    ) -> None:
        # Video root node
        all_kf = sample.keyframes
        video_node = VKGNode(
            id="video_root",
            node_type="VideoNode",
            label="Video",
            level=3,
            t_start=all_kf[0].timestamp if all_kf else 0.0,
            t_end=all_kf[-1].timestamp if all_kf else 0.0,
        )
        graph.add_node(video_node)

        # Episode nodes (from sample — populated by episode segmentation in builder.build)
        import re as _re
        for ep in sample.episodes:
            label = ep.label
            if _re.match(r'^Episode \d+$', label) and ep.summary:
                role_str = ep.narrative_role.replace('_', ' ').title()
                label = f"{role_str}: {ep.summary[:50]}"
            ep_node = VKGNode(
                id=ep.id,
                node_type="EpisodeNode",
                label=label,
                level=2,
                t_start=ep.t_start,
                t_end=ep.t_end,
                parent_id="video_root",
                metadata={
                    "narrative_role": ep.narrative_role,
                    "summary": ep.summary,
                },
            )
            graph.add_node(ep_node)

        # Scene nodes — one per extraction window (preserves temporal attribution).
        # scene_data keys are window IDs like "scene_0007_w03"; each has its own
        # precise time_range so the activator can find exactly the right window
        # for a given question time reference.
        for win_id, sd in sorted(scene_data.items(), key=lambda x: x[1].time_range[0]):
            scene_node = VKGNode(
                id=win_id,
                node_type="SceneNode",
                label=sd.scene_label or win_id,
                level=1,
                t_start=sd.time_range[0],
                t_end=sd.time_range[1],
                metadata={
                    "mood":               sd.scene_mood,
                    "narrative_function": sd.narrative_function,
                    "current_speaker":    sd.current_speaker,
                    "parent_scene_id":    sd.parent_scene_id,
                },
            )
            graph.add_node(scene_node)

        # Clip nodes (individual keyframes)
        for kf in sample.keyframes:
            clip_node = VKGNode(
                id=f"clip_{kf.id}",
                node_type="ClipNode",
                label=f"frame at {kf.timestamp:.1f}s",
                level=0,
                t_start=kf.timestamp,
                t_end=kf.timestamp,
                keyframe_id=kf.id,
            )
            graph.add_node(clip_node)

    def _create_entity_nodes(
        self,
        graph: VKGraph,
        scene_data: Dict[str, SceneData],
        frame_store: FrameStore,
    ) -> None:
        obj_counter = 0
        for scene_id, sd in scene_data.items():
            for obj in sd.objects:
                label = obj.get("label", "object")
                node = VKGNode(
                    id=f"obj_{obj_counter:05d}",
                    node_type="ObjectNode",
                    label=label,
                    level=0,
                    t_start=sd.time_range[0],
                    t_end=sd.time_range[1],
                    bbox=obj.get("bbox_norm"),
                    confidence=float(obj.get("confidence", 1.0)),
                    parent_id=scene_id,
                    metadata={
                        "attributes": obj.get("attributes", []),
                        "state": obj.get("state", ""),
                    },
                )
                graph.add_node(node)
                obj_counter += 1

            # Character nodes (initial — will be resolved later)
            for char in sd.characters:
                cid = f"char_raw_{scene_id}_{obj_counter:05d}"
                node = VKGNode(
                    id=cid,
                    node_type="CharacterNode",
                    label=char.get("description", "unknown person"),
                    level=0,
                    t_start=sd.time_range[0],
                    t_end=sd.time_range[1],
                    bbox=char.get("bbox_norm"),
                    parent_id=scene_id,
                    canonical_description=char.get("description", ""),
                    metadata={
                        "emotion": char.get("emotion", ""),
                        "action":  char.get("action", ""),
                    },
                )
                graph.add_node(node)
                obj_counter += 1

    def _create_event_nodes(
        self,
        graph: VKGraph,
        scene_data: Dict[str, SceneData],
    ) -> None:
        evt_counter = 0
        for scene_id, sd in scene_data.items():
            t_mid = (sd.time_range[0] + sd.time_range[1]) / 2

            for action in sd.actions:
                node = VKGNode(
                    id=f"act_{evt_counter:05d}",
                    node_type="ActionNode",
                    label=action.get("description", "action"),
                    level=0,
                    t_start=sd.time_range[0],
                    t_end=sd.time_range[1],
                    confidence=float(action.get("confidence", 1.0)),
                    parent_id=scene_id,
                    metadata={
                        "actor":  action.get("actor", ""),
                        "object": action.get("object", ""),
                    },
                )
                graph.add_node(node)
                evt_counter += 1

            for sc in sd.state_changes:
                node = VKGNode(
                    id=f"state_{evt_counter:05d}",
                    node_type="StateChangeNode",
                    label=f"{sc.get('entity','')} changes state",
                    level=0,
                    t_start=float(sc.get("approx_timestamp", t_mid)),
                    t_end=float(sc.get("approx_timestamp", t_mid)),
                    prev_state=sc.get("from_state", ""),
                    next_state=sc.get("to_state", ""),
                    parent_id=scene_id,
                    metadata={"entity": sc.get("entity", "")},
                )
                graph.add_node(node)
                evt_counter += 1

    def _create_perception_nodes(
        self,
        graph: VKGraph,
        scene_data: Dict[str, SceneData],
        speech_nodes: List[VKGNode],
    ) -> None:
        ocr_counter = 0
        for scene_id, sd in scene_data.items():
            for ocr in sd.ocr_text:
                text = ocr.get("text", "").strip()
                if not text:
                    continue
                node = VKGNode(
                    id=f"ocr_{ocr_counter:05d}",
                    node_type="OCRNode",
                    label=text,
                    level=0,
                    t_start=sd.time_range[0],
                    t_end=sd.time_range[1],
                    bbox=ocr.get("bbox_norm"),
                    confidence=float(ocr.get("confidence", 1.0)),
                    parent_id=scene_id,
                )
                graph.add_node(node)
                ocr_counter += 1

    # ------------------------------------------------------------------
    # Step 5: Edge builders
    # ------------------------------------------------------------------

    def _build_temporal_edges(self, graph: VKGraph) -> None:
        # Level-0: fine-grained clip/event/speech ordering
        events = sorted(
            [n for n in graph.nodes.values() if n.level == 0],
            key=lambda n: n.t_start,
        )
        for i in range(len(events) - 1):
            a, b = events[i], events[i + 1]
            rel = "PRECEDES" if a.t_end <= b.t_start else "OVERLAPS"
            graph.add_edge(VKGEdge(
                source_id=a.id, target_id=b.id,
                relation_type=rel, weight=1.0, confidence=1.0,
            ))

        # Level-1: scene-to-scene ordering (enables efficient coarse traversal)
        scenes = sorted(
            graph.get_nodes_by_type("SceneNode"), key=lambda n: n.t_start
        )
        for i in range(len(scenes) - 1):
            graph.add_edge(VKGEdge(
                source_id=scenes[i].id, target_id=scenes[i + 1].id,
                relation_type="PRECEDES", weight=1.0, confidence=1.0,
            ))

    def _build_hierarchical_edges(self, graph: VKGraph, sample) -> None:
        # Scene window → episode: link by time-range overlap, not by scene.id
        # (window SceneNodes have IDs like "scene_0007_w03", not scene.id)
        for ep in sample.episodes:
            ep_node = graph.nodes.get(ep.id)
            if ep_node is None:
                continue
            for sc_node in graph.get_nodes_by_type("SceneNode"):
                mid = (sc_node.t_start + sc_node.t_end) / 2
                if ep_node.t_start <= mid <= ep_node.t_end:
                    sc_node.parent_id = ep.id
                    graph.add_edge(VKGEdge(
                        source_id=ep.id, target_id=sc_node.id,
                        relation_type="CONTAINS", weight=1.0, confidence=1.0,
                    ))

        # Episode → video root
        for ep_node in graph.get_episodes():
            graph.add_edge(VKGEdge(
                source_id="video_root", target_id=ep_node.id,
                relation_type="CONTAINS", weight=1.0, confidence=1.0,
            ))

        # Clip → scene (by timestamp containment)
        scene_nodes = sorted(
            graph.get_nodes_by_type("SceneNode"), key=lambda n: n.t_start
        )
        for clip in graph.get_nodes_by_type("ClipNode"):
            for sc in scene_nodes:
                if sc.t_start <= clip.t_start <= sc.t_end:
                    clip.parent_id = sc.id
                    graph.add_edge(VKGEdge(
                        source_id=sc.id, target_id=clip.id,
                        relation_type="CONTAINS", weight=1.0, confidence=1.0,
                    ))
                    break

    def _build_spatial_edges_from_extraction(
        self,
        graph: VKGraph,
        scene_data: Dict[str, SceneData],
    ) -> None:
        for scene_id, sd in scene_data.items():
            for rel in sd.spatial_relations:
                subj = graph.find_entity_in_scene(rel.get("subject", ""), scene_id)
                obj  = graph.find_entity_in_scene(rel.get("object", ""), scene_id)
                if subj and obj:
                    rel_type = rel.get("relation", "near").upper()
                    if rel_type == "CONTAINS":
                        rel_type = "CONTAINS_SPATIAL"
                    graph.add_edge(VKGEdge(
                        source_id=subj.id,
                        target_id=obj.id,
                        relation_type=rel_type,
                        weight=0.85,
                        confidence=0.85,
                        metadata={"source": "qwen3vl", "scene": scene_id},
                    ))

    @staticmethod
    def _bbox_iou(a: List[float], b: List[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _build_crossmodal_edges(self, graph: VKGraph) -> None:
        speech_nodes = graph.get_nodes_by_type("SpeechNode")
        ocr_nodes    = graph.get_nodes_by_type("OCRNode")
        scene_nodes  = graph.get_nodes_by_type("SceneNode")
        obj_nodes    = graph.get_nodes_by_type("ObjectNode")

        # Build scene_id → list of ObjectNodes for fast lookup
        scene_objs: Dict[str, List[VKGNode]] = {}
        for obj in obj_nodes:
            if obj.parent_id:
                scene_objs.setdefault(obj.parent_id, []).append(obj)

        # Speech → concurrent scene (DESCRIBES)
        for speech in speech_nodes:
            for scene in scene_nodes:
                if scene.t_start <= speech.t_start <= scene.t_end:
                    graph.add_edge(VKGEdge(
                        source_id=speech.id, target_id=scene.id,
                        relation_type="DESCRIBES", weight=0.9, confidence=0.9,
                        metadata={"source": "temporal_overlap"},
                    ))
                    break

        # OCR → concurrent scene (LABELS) + overlapping ObjectNode (LABELS)
        for ocr in ocr_nodes:
            matched_scene_id = None
            for scene in scene_nodes:
                if scene.t_start <= ocr.t_start <= scene.t_end:
                    graph.add_edge(VKGEdge(
                        source_id=ocr.id, target_id=scene.id,
                        relation_type="LABELS", weight=0.9, confidence=0.9,
                        metadata={"source": "temporal_overlap"},
                    ))
                    matched_scene_id = scene.id
                    break
            if ocr.bbox and matched_scene_id:
                for obj in scene_objs.get(matched_scene_id, []):
                    if obj.bbox and self._bbox_iou(ocr.bbox, obj.bbox) >= 0.10:
                        graph.add_edge(VKGEdge(
                            source_id=ocr.id, target_id=obj.id,
                            relation_type="LABELS", weight=0.85, confidence=0.85,
                            metadata={"source": "bbox_overlap"},
                        ))

    # ------------------------------------------------------------------
    # Step 8 helpers
    # ------------------------------------------------------------------

    def _collect_character_mentions(
        self,
        graph: VKGraph,
        scene_data: Dict[str, SceneData],
    ) -> List[CharacterMention]:
        mentions = []
        for scene_id, sd in scene_data.items():
            t_mid = (sd.time_range[0] + sd.time_range[1]) / 2
            for char in sd.characters:
                mentions.append(CharacterMention(
                    scene_id=scene_id,
                    timestamp=t_mid,
                    description=char.get("description", "unknown person"),
                    bbox=char.get("bbox_norm"),
                    action=char.get("action"),
                    emotion=char.get("emotion"),
                ))
        return mentions


    _NAME_BANNER_RE = re.compile(
        r"^[A-Z][A-Za-z'\u2019\-]{1,15}(?: [A-Z][A-Za-z'\u2019\-]{1,15}){0,2}$")

    def _link_ocr_names_to_characters(self, graph: VKGraph) -> None:
        """Link on-screen name banners (lower-thirds) to the characters shown.

        TV/show content names people via OCR text (e.g. "SERINAH") displayed
        while the person is on screen, but the extractor mints the OCRNode and
        CharacterNode independently — the graph knows the name appeared and
        that a person appeared, never that they are the same identity.

        A banner usually recurs every time its person is on screen, while
        bystanders change — so the character whose appearance timestamps
        co-occur with the MOST occurrences of a given banner is its referent.
        That character gets the name folded into its label (visible to every
        serialization and keyword match); all co-occurring characters get a
        DESCRIBES edge as weaker evidence.
        """
        chars = graph.get_nodes_by_type("CharacterNode")
        ocrs = graph.get_nodes_by_type("OCRNode")
        if not chars or not ocrs:
            return

        def _app_ts(c) -> list:
            apps = (c.metadata or {}).get("appearances", [])
            ts = [a.get("timestamp") for a in apps if a.get("timestamp") is not None]
            return ts or ([c.t_start] if c.t_start is not None else [])

        # name -> list of banner OCR nodes
        by_name: Dict[str, list] = {}
        for o in ocrs:
            text = (o.label or "").strip()
            if not text or any(ch_.isdigit() for ch_ in text):
                continue
            if not self._NAME_BANNER_RE.match(text.title()):
                continue
            by_name.setdefault(text.title(), []).append(o)

        n_edges = n_named = 0
        for name, banners in by_name.items():
            times = sorted({round(o.t_start, 1) for o in banners
                            if o.t_start is not None})
            if not times:
                continue
            # Count, per character, how many banner occurrences it co-occurs with.
            scores: Dict[str, int] = {}
            for c in chars:
                ts = _app_ts(c)
                scores[c.id] = sum(1 for bt in times
                                   if any(abs(t - bt) <= 5.0 for t in ts))
            hits = {cid: sc for cid, sc in scores.items() if sc > 0}
            if not hits:
                continue
            for o in banners:
                for cid in hits:
                    graph.add_edge(VKGEdge(
                        source_id=o.id, target_id=cid,
                        relation_type="DESCRIBES", weight=1.0,
                        confidence=0.6,
                        metadata={"source": "ocr_name_banner", "name": name},
                    ))
                    n_edges += 1
            # Name the character only when one candidate clearly dominates.
            ranked = sorted(hits.items(), key=lambda kv: -kv[1])
            if len(ranked) == 1 or ranked[0][1] >= 2 * max(1, ranked[1][1]):
                c = graph.nodes.get(ranked[0][0])
                if c is not None and name.lower() not in (c.label or "").lower():
                    c.label = f"{name} ({c.label})" if c.label else name
                    n_named += 1
        print(f"  OCR name banners: {n_edges} DESCRIBES edges, "
              f"{n_named} characters named")

    def _link_characters_to_events(
        self,
        graph: VKGraph,
        characters: List[VKGNode],
    ) -> None:
        for char in characters:
            appearances = char.metadata.get("appearances", [])
            for app in appearances:
                scene_id = app.get("scene_id", "")
                if not scene_id:
                    continue
                # Link character → action nodes in same scene
                for act in graph.get_nodes_by_type("ActionNode"):
                    if act.parent_id == scene_id:
                        actor_desc = act.metadata.get("actor", "").lower()
                        confidence = 0.8
                        relation_type = "PERFORMS"

                        if actor_desc and any(
                            w in char.canonical_description.lower()
                            for w in actor_desc.split()
                            if len(w) > 3
                        ):
                            confidence = 0.9  # Higher confidence for text-matched
                        else:
                            relation_type = "CO_OCCURS_WITH"

                        graph.add_edge(VKGEdge(
                            source_id=char.id, target_id=act.id,
                            relation_type=relation_type,
                            weight=confidence, confidence=confidence,
                        ))

    def _build_same_entity_edges(
        self, graph: VKGraph, characters: List[VKGNode]
    ) -> None:
        by_entity: Dict[str, List[VKGNode]] = {}
        for ch in characters:
            if ch.entity_id:
                by_entity.setdefault(ch.entity_id, []).append(ch)
        for group in by_entity.values():
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    graph.add_edge(VKGEdge(
                        source_id=a.id, target_id=b.id,
                        relation_type="SAME_ENTITY", weight=1.0, confidence=1.0,
                        metadata={"source": "character_resolution"},
                    ))

    def _detect_video_type(self, video_path: str) -> dict:
        """Sample 12 frames from the first 3 minutes and ask the VLM to classify."""
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame = min(total_frames - 1, int(fps * 180))  # first 3 min
        n_frames = 8  # keep well under the 10-image-per-prompt limit
        sample_frames = [int(max_frame * i / (n_frames - 1)) for i in range(n_frames)]

        import base64
        content = []
        for fidx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if not ok:
                continue
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok2:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
            content.append({
                "type": "text",
                "text": f"[t≈{fidx/fps:.0f}s]",
            })
        cap.release()

        content.append({
            "type": "text",
            "text": "Classify this video type based on these frames. Return JSON only.",
        })

        try:
            outputs = self.llm.chat(
                messages=[[
                    {"role": "system", "content": VIDEO_TYPE_SYSTEM_PROMPT},
                    {"role": "user",   "content": content},
                ]],
                sampling_params=VIDEO_TYPE_SAMPLING,
            )
            result = json.loads(outputs[0].outputs[0].text)
            return {
                "video_type":             result.get("video_type", "other"),
                "has_narrator_voiceover": bool(result.get("has_narrator_voiceover", False)),
                "has_multiple_speakers":  bool(result.get("has_multiple_speakers", True)),
                "dominant_language":      result.get("dominant_language", "en"),
                "key_characteristics":    result.get("key_characteristics", []),
            }
        except Exception as e:
            print(f"  Video type detection failed: {e} — defaulting to 'other'")
            return {
                "video_type": "other",
                "has_narrator_voiceover": False,
                "has_multiple_speakers": True,
                "dominant_language": "en",
                "key_characteristics": [],
            }

    def _build_speaker_edges_from_extraction(
        self,
        graph: VKGraph,
        scene_data: Dict[str, SceneData],
    ) -> None:
        """Build SPOKEN_BY edges using the LLM-extracted current_speaker per scene."""
        for scene_id, sd in scene_data.items():
            if not sd.current_speaker or sd.current_speaker == "unknown":
                continue

            speech_in_scene = [
                n for n in graph.get_nodes_by_type("SpeechNode")
                if sd.time_range[0] <= n.t_start <= sd.time_range[1]
            ]
            if not speech_in_scene:
                continue

            if sd.current_speaker == "narrator":
                speaker_node = self._get_or_create_narrator_node(graph)
            else:
                speaker_node = self._find_character_by_description(
                    graph, sd.current_speaker
                )

            if speaker_node is None:
                continue

            conf = 0.9 if sd.speaker_on_screen else 0.75
            for speech in speech_in_scene:
                graph.add_edge(VKGEdge(
                    source_id=speech.id, target_id=speaker_node.id,
                    relation_type="SPOKEN_BY", weight=conf, confidence=conf,
                    metadata={
                        "source":    "llm_extraction",
                        "on_screen": sd.speaker_on_screen,
                    },
                ))

    def _build_ocr_semantic_edges(
        self,
        graph: VKGraph,
        scene_data: Dict[str, SceneData],
    ) -> None:
        """Link OCRNodes to entities using LLM-extracted ocr_semantics."""
        for scene_id, sd in scene_data.items():
            if not sd.ocr_semantics:
                continue
            ocr_nodes_in_scene = [
                n for n in graph.get_nodes_by_type("OCRNode")
                if n.parent_id == scene_id
            ]
            scene_candidates = [
                n for n in graph.nodes.values() if n.parent_id == scene_id
            ]
            for ocr_node in ocr_nodes_in_scene:
                sem = next(
                    (s for s in sd.ocr_semantics
                     if s.get("text", "") == ocr_node.label),
                    None,
                )
                if not sem:
                    continue
                ocr_node.metadata["semantic_type"] = sem.get("semantic_type", "other")
                ocr_node.metadata["refers_to"]     = sem.get("refers_to", "")
                refers_to = sem.get("refers_to", "")
                if not refers_to:
                    continue
                target = graph.find_event_by_description(refers_to, scene_candidates)
                if target and target.id != ocr_node.id:
                    graph.add_edge(VKGEdge(
                        source_id=ocr_node.id, target_id=target.id,
                        relation_type="LABELS", weight=0.9, confidence=0.9,
                        metadata={
                            "source":        "llm_semantic",
                            "semantic_type": sem.get("semantic_type"),
                        },
                    ))

    def _get_or_create_narrator_node(self, graph: VKGraph) -> VKGNode:
        """Return (or create once) a canonical Narrator CharacterNode."""
        if "narrator" in graph.nodes:
            return graph.nodes["narrator"]
        node = VKGNode(
            id="narrator",
            node_type="CharacterNode",
            label="Narrator",
            level=0,
            t_start=0.0,
            t_end=1e9,
            entity_id="entity_narrator",
            canonical_description="narrator / voice-over",
            metadata={"appearances": [], "entity_type": "narrator"},
        )
        graph.add_node(node)
        return node

    def _find_character_by_description(
        self, graph: VKGraph, description: str
    ) -> Optional[VKGNode]:
        """Find the CharacterNode whose canonical_description best matches description."""
        if not description:
            return None
        desc_lower = description.lower()
        desc_words = {w for w in desc_lower.split() if len(w) > 3}

        best_node, best_score = None, 0.0
        for node in graph.get_nodes_by_type("CharacterNode"):
            canon = (node.canonical_description or node.label or "").lower()
            # Exact or near-exact match
            if desc_lower in canon or canon in desc_lower:
                return node
            # Token Jaccard
            canon_words = {w for w in canon.split() if len(w) > 3}
            if not desc_words or not canon_words:
                continue
            score = len(desc_words & canon_words) / len(desc_words | canon_words)
            if score > best_score:
                best_score, best_node = score, node

        return best_node if best_score >= 0.25 else None

    def _build_speech_attribution_edges(self, graph: VKGraph) -> None:
        """Heuristic speaker attribution — kept for backward compatibility; prefer _build_speaker_edges_from_extraction."""
        speech_nodes = graph.get_nodes_by_type("SpeechNode")
        char_nodes = graph.get_nodes_by_type("CharacterNode")
        if not char_nodes:
            return

        # Build scene_id → list of CharacterNodes that appear in that scene
        scene_chars: Dict[str, List[VKGNode]] = {}
        for char in char_nodes:
            for app in char.metadata.get("appearances", []):
                sid = app.get("scene_id")
                if sid:
                    scene_chars.setdefault(sid, []).append(char)

        # Build speech_id → parent scene_id via DESCRIBES edges
        speech_scene: Dict[str, str] = {}
        for edges in graph.edges.values():
            for edge in edges:
                if edge.relation_type == "DESCRIBES":
                    src = graph.nodes.get(edge.source_id)
                    tgt = graph.nodes.get(edge.target_id)
                    if src and tgt and src.node_type == "SpeechNode" and tgt.node_type == "SceneNode":
                        speech_scene[edge.source_id] = edge.target_id

        for speech in speech_nodes:
            best_char: Optional[VKGNode] = None

            best_overlap = 0.0
            for char in char_nodes:
                overlap = max(0.0, min(speech.t_end, char.t_end) - max(speech.t_start, char.t_start))
                if overlap > best_overlap:
                    best_overlap, best_char = overlap, char

            if best_overlap == 0.0:
                scene_id = speech_scene.get(speech.id)
                if scene_id and scene_id in scene_chars:
                    best_char = scene_chars[scene_id][0]

            if best_char is None:
                speech_mid = (speech.t_start + speech.t_end) / 2
                nearest = min(char_nodes, key=lambda c: abs((c.t_start + c.t_end) / 2 - speech_mid))
                if abs((nearest.t_start + nearest.t_end) / 2 - speech_mid) <= 60.0:
                    best_char = nearest

            if best_char is not None:
                confidence = 0.8 if best_overlap > 0.0 else 0.6
                graph.add_edge(VKGEdge(
                    source_id=speech.id, target_id=best_char.id,
                    relation_type="SPOKEN_BY", weight=confidence, confidence=confidence,
                    metadata={"source": "temporal_heuristic"},
                ))


    # ------------------------------------------------------------------
    # New helper methods (Steps 8.2–8.6)
    # ------------------------------------------------------------------

    def _build_location_nodes(
        self, graph: VKGraph, scene_data: Dict[str, SceneData]
    ) -> None:
        """Create LocationNodes and TAKES_PLACE_IN edges from scene extraction output."""
        location_map: Dict[str, VKGNode] = {}

        def _norm(loc: str) -> str:
            return loc.lower().strip().rstrip("s")  # simple normalisation

        for scene_id, sd in scene_data.items():
            raw_loc = (sd.location or "").strip()
            if not raw_loc or raw_loc.lower() == "unknown":
                continue
            key = _norm(raw_loc)
            if key not in location_map:
                loc_node = VKGNode(
                    id=f"loc_{len(location_map):04d}",
                    node_type="LocationNode",
                    label=raw_loc,
                    level=1,
                    t_start=sd.time_range[0],
                    t_end=sd.time_range[1],
                )
                graph.add_node(loc_node)
                location_map[key] = loc_node
            else:
                # Extend the location node's time span
                loc_node = location_map[key]
                loc_node.t_start = min(loc_node.t_start, sd.time_range[0])
                loc_node.t_end   = max(loc_node.t_end,   sd.time_range[1])

            scene_node = graph.nodes.get(scene_id)
            if scene_node:
                scene_node.metadata["location"] = raw_loc
                graph.add_edge(VKGEdge(
                    source_id=scene_id,
                    target_id=location_map[key].id,
                    relation_type="TAKES_PLACE_IN",
                    weight=1.0, confidence=1.0,
                    metadata={"source": "scene_extraction"},
                ))

        if location_map:
            print(f"  Created {len(location_map)} LocationNodes")

    def _build_emotion_shift_edges(self, graph: VKGraph) -> int:
        """Add EMOTION_SHIFT edges between consecutive character appearances with changing emotion."""
        n_edges = 0
        for char_node in graph.get_nodes_by_type("CharacterNode"):
            appearances = sorted(
                char_node.metadata.get("appearances", []),
                key=lambda a: a.get("timestamp", 0),
            )
            prev_emotion = None
            prev_app_node = char_node  # attach edge to the character node itself
            for app in appearances:
                emotion = (app.get("emotion") or "").strip().lower()
                if not emotion or emotion == "unknown":
                    continue
                if prev_emotion and emotion != prev_emotion:
                    graph.add_edge(VKGEdge(
                        source_id=prev_app_node.id,
                        target_id=char_node.id,
                        relation_type="EMOTION_SHIFT",
                        weight=0.9, confidence=0.9,
                        metadata={
                            "from": prev_emotion,
                            "to":   emotion,
                            "t":    app.get("timestamp", 0),
                        },
                    ))
                    n_edges += 1
                prev_emotion = emotion
        return n_edges

    def _build_mentions_edges(
        self, graph: VKGraph, scene_data: Dict[str, SceneData]
    ) -> int:
        """Build MENTIONS edges: SpeechNode → entity, using VLM-extracted mentioned_entities.

        Falls back to token-overlap heuristic when mentioned_entities is empty.
        """
        n_edges = 0
        speech_nodes = graph.get_nodes_by_type("SpeechNode")
        char_nodes   = graph.get_nodes_by_type("CharacterNode")
        obj_nodes    = graph.get_nodes_by_type("ObjectNode")
        entity_nodes = char_nodes + obj_nodes

        # Index entities by label tokens for fast lookup
        entity_label_idx: Dict[str, List[VKGNode]] = {}
        for en in entity_nodes:
            for tok in en.label.lower().split():
                if len(tok) > 3:
                    entity_label_idx.setdefault(tok, []).append(en)

        # Pass 1: use VLM-extracted mentioned_entities (high confidence)
        for scene_id, sd in scene_data.items():
            for me in sd.mentioned_entities:
                refers_to = me.get("refers_to_label", "").strip().lower()
                if not refers_to:
                    continue
                target = graph.find_entity_in_scene(refers_to, scene_id)
                if not target:
                    # Broader search using find_event_by_description
                    candidates = [n for n in entity_nodes
                                  if sd.time_range[0] <= n.t_start <= sd.time_range[1]]
                    target = graph.find_event_by_description(refers_to, candidates)
                if not target:
                    continue
                # Find the SpeechNode covering this scene
                for speech in speech_nodes:
                    if sd.time_range[0] <= speech.t_start <= sd.time_range[1]:
                        graph.add_edge(VKGEdge(
                            source_id=speech.id, target_id=target.id,
                            relation_type="MENTIONS", weight=0.9, confidence=0.9,
                            metadata={"source": "llm_extraction",
                                      "text":   me.get("text", "")},
                        ))
                        n_edges += 1

        # Pass 2: token-overlap heuristic for SpeechNodes (subtitle source gets boost)
        existing_mentions: set = set()
        for elist in graph.edges.values():
            for e in elist:
                if e.relation_type == "MENTIONS":
                    existing_mentions.add((e.source_id, e.target_id))

        for speech in speech_nodes:
            text_lower = speech.label.lower()
            text_tokens = {w for w in text_lower.split() if len(w) > 3}
            conf = 0.9 if speech.metadata.get("source") == "subtitle" else 0.75
            for tok in text_tokens:
                for en in entity_label_idx.get(tok, []):
                    key = (speech.id, en.id)
                    if key not in existing_mentions:
                        # Verify the entity is temporally plausible (within 120s)
                        if abs(en.t_start - speech.t_start) <= 120.0:
                            graph.add_edge(VKGEdge(
                                source_id=speech.id, target_id=en.id,
                                relation_type="MENTIONS", weight=conf, confidence=conf,
                                metadata={"source": "token_overlap"},
                            ))
                            existing_mentions.add(key)
                            n_edges += 1
        return n_edges

    def _build_audio_event_nodes(
        self, graph: VKGraph, audio_rms: "np.ndarray"
    ) -> int:
        """Create AudioEventNodes at RMS peaks ≥ mean + 2σ."""
        import numpy as np
        if audio_rms is None or len(audio_rms) == 0:
            return 0
        mean = float(np.mean(audio_rms))
        std  = float(np.std(audio_rms))
        threshold = mean + 2.0 * std
        if threshold <= 0:
            return 0

        scene_nodes = sorted(graph.get_nodes_by_type("SceneNode"),
                             key=lambda n: n.t_start)
        n_created = 0
        for sec, rms_val in enumerate(audio_rms):
            if float(rms_val) < threshold:
                continue
            t = float(sec)
            node = VKGNode(
                id=f"audio_{sec:06d}",
                node_type="AudioEventNode",
                label=f"audio peak at {t:.0f}s",
                level=0,
                t_start=t,
                t_end=t + 1.0,
                confidence=min(1.0, float(rms_val) / max(threshold, 1e-6)),
                metadata={"rms": float(rms_val), "threshold": threshold},
            )
            graph.add_node(node)

            # Link to concurrent scene with ACCOMPANIES
            for sc in scene_nodes:
                if sc.t_start <= t <= sc.t_end:
                    graph.add_edge(VKGEdge(
                        source_id=node.id, target_id=sc.id,
                        relation_type="ACCOMPANIES", weight=0.8, confidence=0.8,
                    ))
                    break
            n_created += 1
        return n_created

    def _build_episode_participant_lists(self, graph: VKGraph) -> None:
        """Annotate each EpisodeNode with character entity_ids who appear in it."""
        for ep_node in graph.get_episodes():
            participant_ids = []
            seen: set = set()
            for char in graph.get_nodes_by_type("CharacterNode"):
                if char.entity_id and char.entity_id not in seen:
                    # Character overlaps episode if any appearance falls within it
                    for app in char.metadata.get("appearances", []):
                        t = app.get("timestamp", 0)
                        if ep_node.t_start <= t <= ep_node.t_end:
                            participant_ids.append(char.entity_id)
                            seen.add(char.entity_id)
                            break
            ep_node.metadata["participant_entity_ids"] = participant_ids


# ------------------------------------------------------------------
# Helper: convert EpisodeNode to Episode object for causal inference
# ------------------------------------------------------------------

def _episode_node_to_episode(ep_node: VKGNode, graph: VKGraph) -> Episode:
    from .schema import Episode as Ep
    return Ep(
        id=ep_node.id,
        label=ep_node.label,
        t_start=ep_node.t_start,
        t_end=ep_node.t_end,
        narrative_role=ep_node.metadata.get("narrative_role", "other"),
        summary=ep_node.metadata.get("summary", ""),
        scenes=[],
    )
