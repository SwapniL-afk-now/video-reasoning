from __future__ import annotations

"""Inference-time agentic graph traversal — the batched-by-hop walker.

A training-free reasoning loop: the controller LLM picks one typed action over a
visible sub-graph, a deterministic executor applies it, and a coverage–elasticity
warrant decides when to stop. The loop is iterative *per question* but
synchronous *across questions*: each hop fires at most three batched ``llm.chat``
calls (1 controller + 2 answerer read-outs) for the whole question set.

See AGENTIC_TRAVERSAL.md for the design rationale.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from ..schema import AnswerResult, VKGraph
from ..vllm_client import (
    WALKER_ANSWER_SAMPLING,
    WALKER_ANSWER_SYSTEM,
    WALKER_CONTROLLER_SYSTEM,
    extract_mcq_answer,
    walker_controller_sampling,
)
from . import graph_ops
from . import warrant as warrant_mod
from .actions import ACTION_SCHEMA, WalkState, execute_action
from .frame_extractor import extract_frames_for_window, frames_to_b64_urls
from .intent import parse_time_reference
from .serializer import ContextSerializer
from .verifier import verify

_SERIALIZER = ContextSerializer()
_CONTROLLER_SAMPLING = walker_controller_sampling(ACTION_SCHEMA)


# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------

_OPTION_RE = re.compile(r"\(([A-D])\)\s*([^()]*?)(?=\s*\([A-D]\)|$)", re.DOTALL)


def parse_options(question: str) -> Dict[str, str]:
    """Extract ``{"A": text, ...}`` from an MCQ question string."""
    opts: Dict[str, str] = {}
    for m in _OPTION_RE.finditer(question):
        opts[m.group(1)] = " ".join(m.group(2).split())
    return opts


# ---------------------------------------------------------------------------
# Seeding S_0 (deterministic — no LLM call)
# ---------------------------------------------------------------------------

def _resolve_entities(graph: VKGraph, texts: List[str]) -> Set[str]:
    """Fuzzy-match stem/option strings to CharacterNode appearances."""
    seeds: Set[str] = set()
    chars = graph.get_nodes_by_type("CharacterNode")
    for raw in texts:
        low = raw.lower()
        toks = {w for w in re.findall(r"\b\w{4,}\b", low)}
        if not toks:
            continue
        for ch in chars:
            label = (ch.label or "").lower()
            desc = (ch.canonical_description or "").lower()
            cid = (ch.entity_id or "").lower()
            desc_toks = {w for w in re.findall(r"\b\w{4,}\b", desc)}
            if (label and (label in low or low in label)) or \
               (cid and cid in low) or \
               (desc_toks and (toks & desc_toks)):
                if ch.entity_id:
                    seeds |= set(graph.entity_idx.get(ch.entity_id, []))
                seeds.add(ch.id)
    return seeds


def seed_state(
    q: dict,
    graph: VKGraph,
    faiss_index,
    siglip_encoder,
    frame_store,
    video_path: Optional[str],
    mcq: bool,
    seed_faiss_k: int = 12,
) -> WalkState:
    """Build S_0 deterministically: entity nodes + time-window nodes + FAISS seeds."""
    question = q["question"]
    options = parse_options(question) if mcq else {}
    qtype = q.get("question_type") or []
    tr = parse_time_reference(q["time_reference"]) if q.get("time_reference") else None

    node_ids: Set[str] = set()

    # 1) Entities named in the stem + options.
    texts = [question] + list(options.values())
    node_ids |= _resolve_entities(graph, texts)

    # 2) Time-reference window nodes.
    if tr is not None:
        t0, t1 = tr
        node_ids |= {n.id for n in graph.get_nodes_in_window(t0, t1, buffer_sec=15.0)}

    # 3) Top-k FAISS hits for the stem.
    node_ids |= set(graph_ops.faiss_search(
        graph, faiss_index, siglip_encoder, question, k=seed_faiss_k))

    # Fallback: if nothing seeded, take episodes / first nodes.
    if not node_ids:
        eps = graph.get_episodes()
        node_ids = {e.id for e in eps[:6]} or {n.id for n in list(graph.nodes.values())[:10]}

    state = WalkState(
        qid=q.get("uid", question[:24]),
        question=question,
        options=options,
        qtype=qtype,
        time_reference=tr,
        node_ids=set(node_ids),
        frontier=set(node_ids),
    )

    # Seed frames from the time-reference window so entity/OCR questions start
    # with visual evidence.
    if tr is not None and video_path:
        t0, t1 = tr
        frames = extract_frames_for_window(video_path, t0, t1, max_frames=6)
        state.add_frames(list(zip(frames_to_b64_urls(frames),
                                  [f.timestamp for f in frames])))

    return state


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _subgraph_of(graph: VKGraph, node_ids: Set[str]):
    return graph_ops.build_subgraph(graph, node_ids)


def _controller_messages(state: WalkState, graph: VKGraph) -> list:
    """Text-only controller prompt: serialized sub-graph + gaps + last answer."""
    sub = _subgraph_of(graph, state.node_ids)
    ctx = _SERIALIZER.serialize(
        sub, state.question, intents=[], question_types=state.qtype,
        t_start=state.time_reference[0] if state.time_reference else None,
        t_end=state.time_reference[1] if state.time_reference else None,
        visual_provided=False,
        include_edges=True,
    )

    gap_text = "(none — every required slot is filled)"
    coverage = 0.0
    if state.warrant:
        coverage = state.warrant.coverage
        if state.warrant.gaps:
            gap_text = "\n".join(f"  - {g}" for g in state.warrant.gaps)

    frontier_preview = ", ".join(sorted(state.frontier)[:12]) or "(empty)"
    ans = state.current_answer or "(none yet)"

    user = (
        f"{ctx}\n\n"
        f"## Frontier node ids\n{frontier_preview}\n\n"
        f"## Missing evidence (gaps)\n{gap_text}\n\n"
        f"## Coverage so far\n{coverage:.2f}\n\n"
        f"## Your last tentative answer\n{ans}\n\n"
        f"Choose ONE action (JSON)."
    )
    return [
        {"role": "system", "content": WALKER_CONTROLLER_SYSTEM},
        {"role": "user", "content": user},
    ]


def _answerer_messages(state: WalkState, graph: VKGraph, node_ids: Set[str]) -> list:
    """Multimodal answerer prompt over a specific node set (full or ablated)."""
    sub = _subgraph_of(graph, node_ids)
    ctx = _SERIALIZER.serialize(
        sub, state.question, intents=[], question_types=state.qtype,
        t_start=state.time_reference[0] if state.time_reference else None,
        t_end=state.time_reference[1] if state.time_reference else None,
        visual_provided=bool(state.frames),
        include_edges=True,
    )
    content: List[dict] = []
    for url, ts in state.frames:
        content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": f"[t={ts:.0f}s]"})
    content.append({"type": "text", "text": ctx})
    return [
        {"role": "system", "content": WALKER_ANSWER_SYSTEM},
        {"role": "user", "content": content},
    ]


# ---------------------------------------------------------------------------
# Forced backtrack (rejected ANSWER / unsatisfied STOP_REQUEST)
# ---------------------------------------------------------------------------

def _forced_action(state: WalkState, graph: VKGraph) -> Dict[str, Any]:
    """Pick a deterministic action targeting the first unfilled slot."""
    w = state.warrant
    slots = list(w.gap_slots) if w else []
    if "mcq_option_coverage" in slots and w and w.missing_options:
        return {"action": "DISCRIMINATE", "option": w.missing_options[0]}
    if "causal_edge" in slots:
        return {"action": "EXPAND", "relation": "CAUSAL"}
    if "temporal_occurrences" in slots:
        return {"action": "EXPAND", "relation": "ENTITY"}
    if "episode_summary" in slots:
        return {"action": "EXPAND", "relation": "CONTAINS"}
    if "entity_or_frame" in slots:
        if state.frontier:
            nid = sorted(
                state.frontier,
                key=lambda x: graph.nodes[x].t_start if x in graph.nodes else 0.0,
            )[0]
            return {"action": "ZOOM", "node_id": nid}
        return {"action": "EXPAND", "relation": "ENTITY"}
    return {"action": "RECALL", "query": state.question}


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _parse_action(text: str) -> Dict[str, Any]:
    """Parse the controller's constrained-JSON action; tolerate stray prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Find the first JSON object.
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return {"action": "RECALL", "query": ""}


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------

def _log_hop(debug_dir: Optional[str], rec: dict) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "debug_walker.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public API — batched-by-hop walk
# ---------------------------------------------------------------------------

def batch_walk_answer_questions(
    questions: List[dict],
    graph: VKGraph,
    faiss_index,
    llm,
    siglip,
    video_path: Optional[str] = None,
    frame_store=None,
    mcq: bool = True,
    debug_dir: Optional[str] = None,
    max_hops: int = 4,
    theta_cov: float = 1.0,
    k: int = 5,
) -> List[AnswerResult]:
    """Answer all questions via the warrant-gated traversal, batched by hop.

    Per hop, at most three batched ``llm.chat`` calls are issued across all live
    states: one controller pass and two answerer read-outs (full + ablated).
    """
    # Seed all states deterministically (no LLM).
    states: List[WalkState] = [
        seed_state(q, graph, faiss_index, siglip, frame_store, video_path, mcq)
        for q in questions
    ]

    # Initial warrant (coverage only; elasticity unknown → treated as 1).
    for st in states:
        sub = _subgraph_of(graph, st.node_ids)
        st.warrant = warrant_mod.compute_warrant(
            sub, st.qtype, st.question, st.options, bool(st.frames),
            answer_full=None, answer_ablated=None, mcq=mcq, theta_cov=theta_cov,
            discriminated=st.discriminated,
        )

    def live() -> List[WalkState]:
        return [s for s in states if not s.done]

    for hop in range(max_hops):
        live_states = live()
        if not live_states:
            break

        # ---- 1) Controller batch: one action per live state. ----
        ctrl_msgs = [_controller_messages(s, graph) for s in live_states]
        ctrl_out = llm.chat(
            messages=ctrl_msgs,
            sampling_params=_CONTROLLER_SAMPLING,
            use_tqdm=False,
        )
        actions = [_parse_action(o.outputs[0].text) for o in ctrl_out]

        # ---- 2) Resolve each action (verify ANSWER / STOP, else execute). ----
        for st, action in zip(live_states, actions):
            st.hop = hop
            name = (action.get("action") or "").upper()

            if name == "ANSWER":
                letter = (action.get("letter") or "").upper()
                cited = action.get("cited_node_ids") or []
                sub = _subgraph_of(graph, st.node_ids)
                ok, why = verify(letter, cited, sub, st.qtype, st.question,
                                 has_frames=bool(st.frames))
                st.history.append({"hop": hop, "action": "ANSWER", "letter": letter,
                                   "cited": cited, "verify_ok": ok, "verify_why": why})
                if ok and st.warrant and st.warrant.satisfied:
                    st.final_answer = letter
                    st.current_answer = letter
                    st.cited_node_ids = cited
                    st.done = True
                    _log_hop(debug_dir, _hop_record(st, action, ok, why))
                    continue
                # Rejected / not yet warranted → forced backtrack toward a gap.
                action = _forced_action(st, graph)
                st.forced_action = action
            elif name == "STOP_REQUEST":
                st.history.append({"hop": hop, "action": "STOP_REQUEST"})
                if st.warrant and st.warrant.satisfied:
                    st.final_answer = st.current_answer
                    st.done = True
                    _log_hop(debug_dir, _hop_record(st, action, True, "stop satisfied"))
                    continue
                action = _forced_action(st, graph)
                st.forced_action = action
            else:
                st.history.append({"hop": hop, "action": name,
                                   "args": {k2: v for k2, v in action.items()
                                            if k2 != "action"}})

            execute_action(action, st, graph, faiss_index, siglip,
                           frame_store, video_path, k=k)

        # ---- 3) Answerer read-outs (full + ablated), batched. ----
        pending = [s for s in live_states if not s.done]
        if not pending:
            continue

        full_msgs = [_answerer_messages(s, graph, s.node_ids) for s in pending]
        full_out = llm.chat(
            messages=full_msgs,
            sampling_params=WALKER_ANSWER_SAMPLING,
            chat_template_kwargs={"enable_thinking": mcq},
            use_tqdm=False,
        )
        for s, o in zip(pending, full_out):
            s.current_answer = extract_mcq_answer(o.outputs[0].text) if mcq \
                else o.outputs[0].text.strip()

        # Ablated probe — only for states whose last action added a ring.
        ablate = [s for s in pending if s.last_ring]
        ablated_answers: Dict[str, Optional[str]] = {s.qid: s.current_answer for s in pending}
        if ablate:
            ab_msgs = [_answerer_messages(s, graph, s.node_ids - s.last_ring)
                       for s in ablate]
            ab_out = llm.chat(
                messages=ab_msgs,
                sampling_params=WALKER_ANSWER_SAMPLING,
                chat_template_kwargs={"enable_thinking": mcq},
                use_tqdm=False,
            )
            for s, o in zip(ablate, ab_out):
                ablated_answers[s.qid] = extract_mcq_answer(o.outputs[0].text) if mcq \
                    else o.outputs[0].text.strip()

        # ---- 4) Recompute warrant; retire satisfied states. ----
        for s in pending:
            sub = _subgraph_of(graph, s.node_ids)
            s.warrant = warrant_mod.compute_warrant(
                sub, s.qtype, s.question, s.options, bool(s.frames),
                answer_full=s.current_answer,
                answer_ablated=ablated_answers.get(s.qid),
                mcq=mcq, theta_cov=theta_cov,
                discriminated=s.discriminated,
            )
            _log_hop(debug_dir, _hop_record(s, None, None, None))
            if s.warrant.satisfied:
                s.final_answer = s.current_answer
                # Capture citations from a verified ANSWER for this letter, if any.
                cur = (s.current_answer or "").upper()
                for h in reversed(s.history):
                    if h.get("action") == "ANSWER" and h.get("verify_ok") \
                       and (h.get("letter") or "").upper() == cur:
                        s.cited_node_ids = h.get("cited", [])
                        break
                s.done = True

    # Emit: verified/satisfied answer, else current read-out (lowest-risk).
    results: List[AnswerResult] = []
    for s in states:
        answer = s.final_answer or s.current_answer or "ERROR"
        results.append(AnswerResult(
            answer=answer,
            intents=[h.get("action", "") for h in s.history],
            subgraph_size=len(s.node_ids),
            keyframes_used=[ts for _, ts in s.frames],
            evidence_nodes=s.cited_node_ids,
        ))
    return results


def _hop_record(state: WalkState, action, verify_ok, verify_why) -> dict:
    w = state.warrant
    return {
        "qid": state.qid,
        "hop": state.hop,
        "action": action,
        "frontier_size": len(state.frontier),
        "subgraph_size": len(state.node_ids),
        "n_frames": len(state.frames),
        "coverage": w.coverage if w else None,
        "gaps": w.gaps if w else None,
        "elasticity": w.elasticity if w else None,
        "current_answer": state.current_answer,
        "citations": state.cited_node_ids,
        "verify_ok": verify_ok,
        "verify_why": verify_why,
        "satisfied": w.satisfied if w else None,
    }
