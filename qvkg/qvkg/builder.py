from __future__ import annotations

"""VKGBuilder: 11-step offline VKG construction orchestrator."""

import json
import os
import pickle
import uuid
from typing import Dict, List, Optional

from .causal import build_causal_request, parse_causal_edges
from .character import CharacterMention, DescriptionBasedCharacterResolver
from .episode import segment_episodes
from .extraction import SceneData, run_scene_extraction
from .faiss_index import build_faiss_index, build_semantic_edges_faiss
from .frame_store import FrameStore
from .sampler import HierarchicalSampler
from .schema import (
    Episode, FrameInfo, Scene, VKGEdge, VKGNode, VKGraph,
)
from .vllm_client import CAUSAL_SAMPLING


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

    def build(self, video_path: str, output_dir: str) -> VKGraph:
        os.makedirs(output_dir, exist_ok=True)
        graph = VKGraph()
        frame_store = FrameStore(output_dir)

        # ------------------------------------------------------------------
        # Step 1: Hierarchical frame sampling
        # ------------------------------------------------------------------
        sample = _read_pickle(output_dir, "sample.pkl")
        if _is_checkpoint(output_dir, "step1_sampled") and sample is not None:
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
            )

            # Question-aware dense pre-sampling (merges into keyframe list)
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

        # ------------------------------------------------------------------
        # Step 2: Audio transcription
        # ------------------------------------------------------------------
        speech_nodes = _read_pickle(output_dir, "speech_nodes.pkl")
        if _is_checkpoint(output_dir, "step2_transcribed") and speech_nodes is not None:
            print(f"Step 2: Audio transcription [CACHED] ({len(speech_nodes)} segments)")
            graph.add_nodes(speech_nodes)
        else:
            print("Step 2: Audio transcription (Whisper)...")
            speech_nodes = []
            if self.whisper is not None:
                try:
                    segments, info = self.whisper.transcribe(
                        video_path, word_timestamps=False
                    )
                    speech_nodes = self._build_speech_nodes(list(segments))
                    graph.add_nodes(speech_nodes)
                    _write_pickle(output_dir, "speech_nodes.pkl", speech_nodes)
                    _write_checkpoint(output_dir, "step2_transcribed")
                    print(f"  Transcribed {len(speech_nodes)} speech segments")
                except Exception as e:
                    print(f"  Whisper failed: {e} — skipping audio")

        # ------------------------------------------------------------------
        # Step 3: Scene extraction
        # ------------------------------------------------------------------
        scene_data = _read_pickle(output_dir, "scene_data.pkl")
        if _is_checkpoint(output_dir, "step3_extracted") and scene_data is not None:
            print(f"Step 3: Scene extraction [CACHED] ({len(scene_data)} scenes)")
        else:
            print("Step 3: Scene extraction (Qwen batch)...")
            scene_data = run_scene_extraction(sample.scenes, frame_store, self.llm)
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

            # Step 8: Character resolution
            print("Step 8: Character resolution...")
            resolver = DescriptionBasedCharacterResolver()
            char_mentions = self._collect_character_mentions(graph, scene_data)
            characters = resolver.resolve(char_mentions, self.siglip)
            for char in characters:
                if char.id in graph.nodes:
                    graph.nodes[char.id] = char
                else:
                    graph.add_node(char)
            self._link_characters_to_events(graph, characters)
            print(f"  Resolved {len(characters)} characters")

            graph.save(checkpoint_graph)
            _write_checkpoint(output_dir, "step8_graph_built")
            print(f"  Graph checkpoint saved ({len(graph.nodes)} nodes)")

        # Step 9: FAISS index
        index_path = os.path.join(output_dir, "vkg.index")
        if _is_checkpoint(output_dir, "step9_indexed") and os.path.exists(index_path):
            print("Step 9: FAISS index [CACHED]")
            from .faiss_index import load_faiss_index
            faiss_index = load_faiss_index(index_path)
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
            graph.save(checkpoint_graph)
            _write_checkpoint(output_dir, "step10_causal")

        # Step 11: Semantic edges
        if _is_checkpoint(output_dir, "step11_semantic"):
            print("Step 11: Semantic edges [CACHED]")
            graph = VKGraph.load(checkpoint_graph)
        else:
            print("Step 11: Semantic edges (FAISS ANN)...")
            n_sem = build_semantic_edges_faiss(
                graph, faiss_index,
                threshold=self.config.get("semantic_threshold", 0.78),
                k_neighbors=10,
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
        for ep in sample.episodes:
            ep_node = VKGNode(
                id=ep.id,
                node_type="EpisodeNode",
                label=ep.label,
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

        # Scene nodes
        for scene in sample.scenes:
            sd = scene_data.get(scene.id, SceneData(scene.id, (scene.t_start, scene.t_end)))
            scene_node = VKGNode(
                id=scene.id,
                node_type="SceneNode",
                label=sd.scene_label or f"scene_{scene.id}",
                level=1,
                t_start=scene.t_start,
                t_end=scene.t_end,
                keyframe_id=scene.keyframes[0].id if scene.keyframes else None,
                metadata={"mood": sd.scene_mood},
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
        events = sorted(
            [n for n in graph.nodes.values() if n.level == 0],
            key=lambda n: n.t_start,
        )
        for i in range(len(events) - 1):
            a, b = events[i], events[i + 1]
            if a.t_end <= b.t_start:
                rel = "PRECEDES"
            else:
                rel = "OVERLAPS"
            graph.add_edge(VKGEdge(
                source_id=a.id, target_id=b.id,
                relation_type=rel, weight=1.0, confidence=1.0,
            ))

    def _build_hierarchical_edges(self, graph: VKGraph, sample) -> None:
        # Scene → episode
        for ep in sample.episodes:
            ep_node = graph.nodes.get(ep.id)
            if ep_node is None:
                continue
            for scene in ep.scenes:
                sc_node = graph.nodes.get(scene.id)
                if sc_node:
                    sc_node.parent_id = ep.id
                    graph.add_edge(VKGEdge(
                        source_id=ep.id, target_id=scene.id,
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

    def _build_crossmodal_edges(self, graph: VKGraph) -> None:
        speech_nodes = graph.get_nodes_by_type("SpeechNode")
        ocr_nodes    = graph.get_nodes_by_type("OCRNode")
        scene_nodes  = graph.get_nodes_by_type("SceneNode")

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

        # OCR → concurrent scene (LABELS)
        for ocr in ocr_nodes:
            for scene in scene_nodes:
                if scene.t_start <= ocr.t_start <= scene.t_end:
                    graph.add_edge(VKGEdge(
                        source_id=ocr.id, target_id=scene.id,
                        relation_type="LABELS", weight=0.9, confidence=0.9,
                        metadata={"source": "temporal_overlap"},
                    ))
                    break

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

    def _link_characters_to_events(
        self,
        graph: VKGraph,
        characters: List[VKGNode],
    ) -> None:
        for char in characters:
            appearances = char.metadata.get("appearances", [])
            for app in appearances:
                scene_id = app.get("scene_id", "")
                # Link character → action nodes in same scene
                for act in graph.get_nodes_by_type("ActionNode"):
                    if act.parent_id == scene_id:
                        actor_desc = act.metadata.get("actor", "").lower()
                        if actor_desc and any(
                            w in char.canonical_description.lower()
                            for w in actor_desc.split()
                            if len(w) > 3
                        ):
                            graph.add_edge(VKGEdge(
                                source_id=char.id, target_id=act.id,
                                relation_type="PERFORMS",
                                weight=0.8, confidence=0.8,
                            ))


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
