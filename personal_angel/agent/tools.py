"""Tool registry for the master agent (MRKL-style expert modules).

Each tool returns a ToolResult with an observation string (what the agent
reads), a payload (what the UI shows), an optional belief support in [-1, 1]
and the compute cost charged to the investigation budget.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..events.graph import EventGraph
from ..memory.store import MemoryStore
from ..perception.pipeline import PerceptionResult
from ..schema import ActionDecision, Event, InvestigationState, UserQuestion, new_id
from .actions import ActionExecutor
from .llm import LLMClient, extract_json_object
from .policy import InvestigationPolicy, update_belief
from .safety import SafetyGuard

AnswerProvider = Callable[[UserQuestion], str | None]

@dataclass
class ToolResult:
    observation: str
    payload: dict[str, Any] = field(default_factory=dict)
    support: float | None = None
    cost: float = 0.0
    images: list[str] = field(default_factory=list)
    stop: bool = False

TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "inspect_frames", "cost_key": "inspect_frames",
     "description": "Summarize detections, tracks and poses in a time window and list the evidence frames there.",
     "input": {"start_s": "number", "end_s": "number"}},
    {"name": "run_vlm", "cost_key": "run_vlm",
     "description": "Send up to N evidence frames to the local vision-language model with a focused question. EXPENSIVE; use for verification of the primary hypothesis only.",
     "input": {"evidence_ids": "list of evidence ids (frames)", "question": "string"}},
    {"name": "run_pose_analysis", "cost_key": "run_pose_analysis",
     "description": "Return the measured body-geometry features (torso angle, drop velocity, dwell, stillness) for a person track.",
     "input": {"subject": "PERSON_xx"}},
    {"name": "run_audio_analysis", "cost_key": "run_audio_analysis",
     "description": "Return the transcript (original language + English translation), acoustic events and text-risk scores.",
     "input": {}},
    {"name": "query_memory", "cost_key": "query_memory",
     "description": "Search episodic memory (past incidents at this location) and procedural memory (strategies that worked).",
     "input": {"query": "string"}},
    {"name": "retrieve_policy", "cost_key": "retrieve_policy",
     "description": "Retrieve the operator's policy/rules for this environment (vehicle, nursery, site).",
     "input": {"query": "string"}},
    {"name": "translate", "cost_key": "translate",
     "description": "Translate text to a target language (for questions to the passenger or for foreign speech).",
     "input": {"text": "string", "target_language": "en|es|..."}},
    {"name": "run_critic", "cost_key": "critic",
     "description": "Ask the adversarial critic agent to attack the current hypothesis. Required before any high-cost action.",
     "input": {}},
    {"name": "ask_user", "cost_key": "translate",
     "description": "Ask the present person a short question through the cabin/room speaker and wait for the answer (timeout → 'no_response'). Only when the policy says asking is safe.",
     "input": {"question": "string", "language": "en|es", "options": "list of short expected answers"}},
    {"name": "propose_action", "cost_key": "translate",
     "description": "Propose a consequential action. The policy gate checks belief, harm asymmetry, consent and critic before it executes (simulated).",
     "input": {"action": "one of the allowlisted actions", "rationale": "string", "params": "object"}},
    {"name": "finalize", "cost_key": "translate",
     "description": "Finish: give the evidence-grounded answer, hypothesis, probability, uncertainty and the recommended/executed action.",
     "input": {"answer": "string", "hypothesis": "string", "probability": "0..1", "uncertainty": "string",
               "recommended_action": "string"}},
]

def _densest_cluster(people: list) -> list:
    """People whose boxes overlap/touch each other (expanded by 40%): the knot of a fight in a crowd frame."""
    def expanded(d):
        x1, y1, x2, y2 = d.box_xyxy
        mx, my = (x2 - x1) * 0.4, (y2 - y1) * 0.4
        return (x1 - mx, y1 - my, x2 + mx, y2 + my)

    def touch(a, b) -> bool:
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    boxes = [expanded(d) for d in people]
    best: list = []
    for i in range(len(people)):
        group = [i]
        changed = True
        while changed:
            changed = False
            for j in range(len(people)):
                if j not in group and any(touch(boxes[j], boxes[g]) for g in group):
                    group.append(j)
                    changed = True
        if len(group) > len(best):
            best = group
    return [people[i] for i in best] if len(best) >= 2 else []

class ToolBox:
    def __init__(self, state: InvestigationState, perception: PerceptionResult, graph: EventGraph,
                 memory: MemoryStore, llm: LLMClient, policy: InvestigationPolicy, executor: ActionExecutor,
                 guard: SafetyGuard, config: dict[str, Any], answer_provider: AnswerProvider | None,
                 telemetry: Any, fixture: dict[str, Any] | None = None) -> None:
        self.state = state
        self.perception = perception
        self.graph = graph
        self.memory = memory
        self.llm = llm
        self.policy = policy
        self.executor = executor
        self.guard = guard
        self.config = config
        self.answer_provider = answer_provider
        self.telemetry = telemetry
        self.fixture = fixture or {}
        self.costs = {k: float(v) for k, v in config.get("agent", {}).get("costs", {}).items()}
        self.max_images = int(config.get("llm", {}).get("max_images_per_call", 6))
        self.audio_reported = False
        self.language = self._detect_language()

    def _detect_language(self) -> str:
        segs = self.perception.audio_segments
        if segs and segs[0].language and segs[0].language != "unknown":
            return segs[0].language
        return "en"

    def subject_is_infant(self, primary: Event | None) -> bool:
        if primary is None:
            return False
        a = primary.attributes or {}
        if a.get("age_group") == "baby" or a.get("holder_age_group") == "baby":
            return True
        scene = getattr(self.perception, "scene", None)
        try:
            tid = int(primary.subject.split("_")[1]) if primary.subject.startswith("PERSON_") and primary.subject[7:].isdigit() else None
        except ValueError:
            tid = None
        return bool(scene and tid is not None and scene.age_of(tid) == "baby")

    def primary_event(self) -> Event | None:
        candidates = [e for e in self.state.events if e.kind != "normal_activity"]
        if not candidates:
            return self.state.events[0] if self.state.events else None

        if self.state.hypotheses and self.state.hypotheses[-1].kind:
            same = [e for e in candidates if e.kind == self.state.hypotheses[-1].kind]
            if same:
                return max(same, key=lambda e: e.severity * 0.6 + e.confidence * 0.4)
        return max(candidates, key=lambda e: e.severity * 0.6 + e.confidence * 0.4)

    def specs(self) -> list[dict[str, Any]]:
        return TOOL_SPECS

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        ok, why = self.guard.authorize_tool(name)
        if not ok:
            return ToolResult(f"DENIED: {why}. Choose another tool or finalize.", {"denied": True})
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return ToolResult(f"Unknown tool '{name}'. Available: {[t['name'] for t in TOOL_SPECS]}")
        spec = next(t for t in TOOL_SPECS if t["name"] == name)
        cost = self.costs.get(spec["cost_key"], 1.0)
        if self.state.compute_spent + cost > self.state.compute_budget and name not in {"finalize", "propose_action", "ask_user"}:
            return ToolResult(f"BUDGET: {cost:.0f} units needed, {self.state.compute_budget - self.state.compute_spent:.0f} left. Finalize with current evidence.",
                              {"budget_exhausted": True})
        try:
            result = handler(**args) if isinstance(args, dict) else handler()
        except TypeError as error:
            return ToolResult(f"Bad arguments for {name}: {error}. Expected {spec['input']}")
        except Exception as error:
            return ToolResult(f"Tool {name} failed: {type(error).__name__}: {error}")
        result.cost = cost if result.cost == 0.0 else result.cost
        self.state.compute_spent += result.cost
        self.telemetry.increment("tool_calls")
        return result

    def tool_inspect_frames(self, start_s: float = 0.0, end_s: float | None = None, **_: Any) -> ToolResult:
        end_s = self.perception.duration_s if end_s is None else float(end_s)
        start_s = float(start_s)
        frames = [o for o in (self.perception.dense or self.perception.scan) if start_s - 0.01 <= o.timestamp_s <= end_s + 0.01]
        if not frames:
            frames = self.perception.scan
        lines = []
        labels: dict[str, int] = {}
        tracks: set[int] = set()
        for o in frames:
            for d in o.detections:
                labels[d.label] = labels.get(d.label, 0) + 1
                if d.track_id is not None and d.label == "person":
                    tracks.add(d.track_id)
        lines.append(f"{len(frames)} inspected frames between {start_s:.1f}s and {end_s:.1f}s; "
                     f"person tracks {sorted(tracks)}; label counts {json.dumps(labels)}.")
        scene = getattr(self.perception, "scene", None)
        if scene is not None:
            lines.append(f"- scene: {scene.description()}")
        for ev in self.perception.human_events:
            if ev.end_s >= start_s and ev.start_s <= end_s:
                lines.append(f"- human state: {ev.explanation}")
        for w in self.perception.weapons:
            if w.last_s >= start_s and w.first_s <= end_s:
                holder = f"PERSON_{w.holder_track:02d}" if w.holder_track is not None else "no person track (unattached box)"
                extra = (f", zero-shot second opinion: {w.second_opinion} (weapon p={w.weapon_probability:.2f})"
                         if w.second_opinion else "")
                lines.append(f"- weapon-like object: {w.label} conf {w.max_conf:.2f}, persistence {w.persistence:.0%}, holder {holder}{extra}")
        seeds = [e.event_id for e in self.state.events if e.end_s >= start_s and e.start_s <= end_s]
        ranked = self.graph.rank_evidence(seeds, top_k=8)
        ev_lines, images = [], []
        for ev_id, score in ranked:
            ev = self.state.evidence.get(ev_id)
            if ev is None or not (start_s - 2 <= ev.start_s <= end_s + 2):
                continue
            ev_lines.append(f"  {ev_id} [{ev.kind} @ {ev.start_s:.1f}s, rank {score:.3f}]: {ev.description[:140]}")
            if ev.kind == "frame" and ev.path:
                images.append(ev.path)
        if ev_lines:
            lines.append("Evidence ranked by personalized PageRank:")
            lines.extend(ev_lines)
        return ToolResult("\n".join(lines), {"window": [start_s, end_s], "ranked_evidence": ranked}, images=images[:6])

    def tool_run_pose_analysis(self, subject: str = "", **_: Any) -> ToolResult:
        rows = []
        for ev in self.perception.human_events:
            tag = f"PERSON_{ev.track_id:02d}"
            if subject and subject.upper() not in {tag, "ANY", "ALL"}:
                continue
            rows.append(f"{tag} {ev.kind} {ev.start_s:.1f}-{ev.end_s:.1f}s conf {ev.confidence:.2f}: {json.dumps(ev.features)}")
        if not rows:
            return ToolResult("No human-state transitions were measured for that subject (person stayed upright/normal).",
                              support=-0.3)
        return ToolResult("\n".join(rows), {"rows": rows}, support=0.2)

    def tool_run_audio_analysis(self, **_: Any) -> ToolResult:
        p = self.perception
        if not p.audio_segments and not p.acoustic_events:
            return ToolResult("No speech or salient acoustic events were found in the audio track.", support=None, cost=1.0)
        lines = []
        for s in p.audio_segments:
            txt = self.guard.sanitize_untrusted(s.text, "asr_transcript")
            en = self.guard.sanitize_untrusted(s.translation_en or "", "asr_translation") if s.translation_en and s.language != "en" else ""
            lines.append(f"[{s.start_s:.1f}-{s.end_s:.1f}s] ({s.language}) {txt}" + (f" | EN: {en}" if en else ""))
        for a in p.acoustic_events:
            lines.append(f"[{a.start_s:.1f}-{a.end_s:.1f}s] acoustic: {a.label} ({a.confidence:.2f})")
        r = p.text_risk
        lines.append(f"Text-risk ({r.model}): threat={r.threat:.2f} hate={r.hate:.2f} toxicity={r.toxicity:.2f}; matched cues: {r.matched_cues}")
        support = None
        primary = self.primary_event()
        if primary and primary.kind in {"threatening_speech", "hateful_speech", "distress_speech"}:
            support = 0.5 if max(r.threat, r.hate) >= 0.5 or primary.kind == "distress_speech" else -0.4
        cost = 1.0 if self.audio_reported else 0.0
        self.audio_reported = True
        return ToolResult("\n".join(lines), {"segments": [s.to_dict() for s in p.audio_segments], "risk": r.to_dict()},
                          support=support, cost=cost)

    def tool_run_vlm(self, evidence_ids: list[str] | None = None, question: str = "", **_: Any) -> ToolResult:
        ids = evidence_ids or []
        images = []
        for ev_id in ids:
            ev = self.state.evidence.get(ev_id)
            if ev and ev.kind == "frame" and ev.path and Path(ev.path).exists():
                images.append(ev.path)
        if not images:
            primary = self.primary_event()
            seeds = [primary.event_id] if primary else []
            for ev_id, _ in self.graph.rank_evidence(seeds, top_k=8):
                ev = self.state.evidence.get(ev_id)
                if ev and ev.kind == "frame" and ev.path and Path(ev.path).exists():
                    images.append(ev.path)
        primary = self.primary_event()

        images = self._with_peak_frames(images, primary)
        images = images[: self.max_images]
        if not images:
            return ToolResult("No evidence frames are available for the VLM (audio-only media?).", cost=1.0)
        crops: list[str] = []
        if primary is not None and primary.kind == "weapon_visible":
            crops = self._weapon_crops(primary)
        elif primary is not None and primary.kind == "aggressive_interaction":
            crops = self._fight_crops(primary)
        if crops:
            images = images[: max(1, self.max_images - len(crops))] + crops
        hypothesis = self.state.hypotheses[-1].statement if self.state.hypotheses else (primary.summary if primary else "")
        if not self.llm.is_real:
            return self._fixture_vlm(images, question, primary)
        scene = getattr(self.perception, "scene", None)
        scene_line = f"Scene understanding so far: {scene.description()}\n" if scene is not None else ""
        weapon_note = ("The RED boxes are weapon *candidates* from a detector that is known to confuse toys, phones, remote "
                       "controls and tools with guns/knives. Say explicitly whether the boxed object is a real firearm/blade, "
                       "a toy, a phone/other object, or not determinable.\n"
                       + (f"The last {len(crops)} image(s) are ZOOMED CROPS of the red box — judge the object from them.\n" if crops else "")
                       if primary is not None and primary.kind == "weapon_visible" else "")
        fight_note = ""
        if primary is not None and primary.kind == "aggressive_interaction":
            group = bool(primary.attributes.get("group_fight"))
            fight_note = (
                "The question is whether people are PHYSICALLY FIGHTING (punching, kicking, shoving, grappling, someone being "
                "knocked or thrown down, a crowd surging around a struggle) as opposed to playing, dancing, greeting or sport. "
                "A crowd brawl counts as fighting even if you cannot tell who started it or who is the victim — do not answer "
                "'no aggression' merely because the two named person ids are not the ones fighting; judge the scene. "
                "Motion blur, several people converging on one spot, arms swinging and bodies on the floor are signs of a real fight. "
                + (f"The last {len(crops)} image(s) are ZOOMED CROPS of the region where the detector saw striking motions.\n" if crops else "\n")
                + ("The hypothesis is a multi-person brawl: say how many people are involved and how violent it looks (0-10).\n" if group else "")
            )
        prompt = (
            "You are the vision specialist of a safety-investigation system. Look ONLY at the attached frames "
            f"({len(images)} frames, chronological; boxes and labels were drawn by an object detector and may be wrong). "
            "Answer the investigator's question and judge whether the frames support the hypothesis. Describe the environment "
            "and the people (approximate age group, posture, what they hold). Do not identify people. Do not assume anything "
            "not visible.\n" + scene_line + weapon_note + fight_note +
            f"Hypothesis: {hypothesis}\nQuestion: {question or 'Describe what is happening and whether the hypothesis holds.'}\n"
            "Reply with JSON only: {\"description\": str, \"answer\": str, \"scene\": str, \"supports_hypothesis\": number in [-1,1], "
            "\"confidence\": number in [0,1], \"alternatives\": [str]}"
        )
        t0 = time.perf_counter()
        resp = self.llm.chat([{"role": "system", "content": "You answer with strict JSON."}, {"role": "user", "content": prompt}],
                             json_mode=True, images=images, max_tokens=600)
        self.telemetry.record_model_call("vlm", resp.model, resp.tokens_in, resp.tokens_out, resp.latency_ms, len(images))
        data = extract_json_object(resp.content) or {"description": resp.content[:400], "answer": resp.content[:400],
                                                     "supports_hypothesis": 0.0, "confidence": 0.3, "alternatives": []}
        support = float(data.get("supports_hypothesis", 0.0))
        support = max(-1.0, min(1.0, support))
        note = ""
        if primary is not None and primary.kind == "aggressive_interaction" and support < 0.2:

            from .critic import reconcile_verdict

            verdict, why = reconcile_verdict("aggressive_interaction", "weakened",
                                             f"{data.get('description', '')} {data.get('answer', '')}", "", "")
            if verdict == "supported":
                support = max(support, 0.4)
                note = " | reconciled: the description narrates a physical fight, so it supports the incident (+0.40)"
                data["supports_hypothesis"] = support
                data["reconciled"] = why
        obs = (f"VLM ({resp.model}, {len(images)} frames, {(time.perf_counter() - t0) * 1000:.0f} ms): {data.get('description', '')} "
               f"| answer: {data.get('answer', '')} | supports_hypothesis={support:+.2f} confidence={float(data.get('confidence', 0)):.2f} "
               f"| alternatives: {data.get('alternatives', [])}{note}")
        return ToolResult(obs, {"vlm": data, "images": images}, support=support, images=images)

    def _with_peak_frames(self, images: list[str], primary: Event | None) -> list[str]:
        if primary is None or primary.kind == "normal_activity":
            return images
        inside = [ev for ev in self.state.evidence.values() if ev.kind == "frame" and ev.path and Path(ev.path).exists()
                  and primary.start_s - 0.3 <= ev.start_s <= primary.end_s + 0.3]
        if not inside:
            return images
        have = set(images)
        if any(Path(p).exists() and p in {ev.path for ev in inside} for p in images):
            return images
        mid = (primary.start_s + primary.end_s) / 2
        inside.sort(key=lambda ev: (abs(ev.start_s - mid), -ev.score))
        extra = [ev.path for ev in inside if ev.path not in have][:2]
        keep = max(1, self.max_images - len(extra))
        return images[:keep] + extra

    def _run_dir(self, primary: Event) -> Path | None:
        for ev_id in primary.evidence_ids:
            ev = self.state.evidence.get(ev_id)
            if ev is not None and ev.path:
                return Path(ev.path).parent
        for ev in self.state.evidence.values():
            if ev.path:
                return Path(ev.path).parent
        return None

    def _fight_crops(self, primary: Event) -> list[str]:
        """Zoomed crops around the people involved in the striking motions: CCTV figures are ~50 px tall in a full
        frame, far too small for a 7B vision model to see punches. Two moments: the middle and the late part of the event."""
        try:
            import cv2
        except Exception:
            return []
        run_dir = self._run_dir(primary)
        if run_dir is None or self.perception.media_kind != "video":
            return []
        wanted_ids: set[int] = set()
        for tag in [primary.subject, primary.obj or ""] + list(primary.attributes.get("strikers") or []):
            if isinstance(tag, str) and tag.startswith("PERSON_") and tag[7:].lstrip("-").isdigit():
                wanted_ids.add(int(tag[7:]))
        dur = max(0.0, primary.end_s - primary.start_s)
        times = [primary.start_s + 0.5 * dur, primary.start_s + 0.8 * dur] if dur > 0.6 else [primary.start_s + 0.5 * dur]
        out: list[str] = []
        try:
            from ..video import VideoReader

            reader = VideoReader(self.perception.media_path)
        except Exception:
            return []
        for k, t in enumerate(times):
            obs = self.perception.observation_at(t)
            if obs is None:
                continue
            people = [d for d in obs.detections if d.label == "person"]
            if not people:
                continue
            chosen = [d for d in people if d.track_id in wanted_ids] or []
            if len(chosen) < 2:

                chosen = _densest_cluster(people) or people
            x1 = min(d.box_xyxy[0] for d in chosen); y1 = min(d.box_xyxy[1] for d in chosen)
            x2 = max(d.box_xyxy[2] for d in chosen); y2 = max(d.box_xyxy[3] for d in chosen)
            try:
                frame = reader.frame_at(obs.timestamp_s)
            except Exception:
                frame = None
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if (x2 - x1) * (y2 - y1) >= 0.6 * w * h:
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = max((x2 - x1) * 1.5, 220), max((y2 - y1) * 1.5, 220)
            xa, ya = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
            xb, yb = int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2))
            crop = frame[ya:yb, xa:xb]
            if crop.size == 0:
                continue
            scale = 640 / max(crop.shape[:2])
            if scale > 1:
                crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
            path = run_dir / f"crop_fight_{obs.frame_index:06d}_{k}.jpg"
            cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            out.append(str(path))
        return out

    def _weapon_crops(self, primary: Event) -> list[str]:
        """Zoomed crops of the best weapon sighting(s): small objects are what small VLMs miss in full frames."""
        try:
            import cv2
        except Exception:
            return []
        out: list[str] = []
        run_dir = Path(self.state.evidence[primary.evidence_ids[0]].path).parent if primary.evidence_ids and self.state.evidence.get(primary.evidence_ids[0]) and self.state.evidence[primary.evidence_ids[0]].path else None
        if run_dir is None:
            return []
        for w in self.perception.weapons[:2]:
            if w.best_frame is None or w.best_box is None:
                continue
            obs = next((o for o in (self.perception.dense or self.perception.scan) if o.frame_index == w.best_frame), None)
            if obs is None:
                continue
            try:
                from ..video import VideoReader

                frame = VideoReader(self.perception.media_path).frame_at(obs.timestamp_s) if self.perception.media_kind == "video" else cv2.imread(self.perception.media_path)
            except Exception:
                frame = None
            if frame is None:
                continue
            h, wd = frame.shape[:2]
            x1, y1, x2, y2 = w.best_box
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 48) * 2.2, max(y2 - y1, 48) * 2.2
            side = max(bw, bh)
            xa, ya = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            xb, yb = int(min(wd, cx + side / 2)), int(min(h, cy + side / 2))
            crop = frame[ya:yb, xa:xb]
            if crop.size == 0:
                continue
            scale = 448 / max(crop.shape[:2])
            if scale > 1:
                crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
            path = run_dir / f"crop_{w.label}_{w.best_frame:06d}.jpg"
            cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            out.append(str(path))
        return out

    def _fixture_vlm(self, images: list[str], question: str, primary: Event | None) -> ToolResult:
        spec = self.fixture.get("vlm", {})
        kind = primary.kind if primary else "normal_activity"
        support = float(spec.get("support", {}).get(kind, 0.55 if kind != "normal_activity" else -0.5))
        desc = spec.get("description", {}).get(kind) or (
            f"[fixture VLM] The frames are consistent with '{kind}': {primary.summary if primary else 'nothing unusual'}")
        obs = (f"VLM (fixture, {len(images)} frames): {desc} | supports_hypothesis={support:+.2f} confidence=0.7 "
               f"| alternatives: {spec.get('alternatives', {}).get(kind, ['benign explanation not ruled out'])}")
        self.telemetry.record_model_call("vlm", "fixture", 900 * len(images), 120, 12.0, len(images))
        return ToolResult(obs, {"vlm": {"description": desc, "supports_hypothesis": support}, "images": images},
                          support=support, images=images)

    def _policy_source(self) -> str | None:
        loc = self.perception.location
        return {"CAR_CABIN": "vehicle", "VEHICLE": "vehicle", "NURSERY": "childcare"}.get(loc, "site")

    def tool_query_memory(self, query: str = "", **_: Any) -> ToolResult:
        hits = self.memory.search(query or (self.primary_event().summary if self.primary_event() else "incident"),
                                  tiers=("episodic", "procedural"), top_k=4, scope=self.perception.location)
        self.state.retrieved.extend(hits)
        if not hits:
            return ToolResult("Memory: no relevant past episodes or strategies at this location yet.")
        lines = [f"- ({h['tier']}, score {h['score']:.2f}) {self.guard.sanitize_untrusted(h['text'][:420], 'memory')}" for h in hits]
        return ToolResult("Memory hits — these are OTHER recordings from the past, useful only as strategy context; they are "
                          "NOT evidence about the current media and must not change the belief:\n" + "\n".join(lines),
                          {"hits": hits})

    def tool_retrieve_policy(self, query: str = "", **_: Any) -> ToolResult:
        hits = self.memory.search(query or "escalation policy", tiers=("semantic",), top_k=3,
                                  prefer_source=self._policy_source())
        self.state.retrieved.extend(hits)
        if not hits:
            return ToolResult("No policy document matched.")
        lines = [f"- ({h['meta'].get('source', 'policy')}, score {h['score']:.2f}) {self.guard.sanitize_untrusted(h['text'][:400], 'policy_doc')}" for h in hits]
        return ToolResult("Policy excerpts:\n" + "\n".join(lines), {"hits": hits})

    def tool_translate(self, text: str = "", target_language: str = "en", **_: Any) -> ToolResult:
        if not text:
            return ToolResult("Nothing to translate.")
        for s in self.perception.audio_segments:
            if text.strip() and text.strip() in s.text and s.translation_en and target_language.startswith("en"):
                return ToolResult(f"Translation (ASR translate task): {s.translation_en}", {"translation": s.translation_en})
        if not self.llm.is_real:
            return ToolResult(f"[fixture translation to {target_language}] {text}", {"translation": text})
        resp = self.llm.chat([{"role": "system", "content": "You are a precise translator. Output only the translation."},
                              {"role": "user", "content": f"Translate to {target_language}:\n{text}"}], max_tokens=300)
        self.telemetry.record_model_call("translate", resp.model, resp.tokens_in, resp.tokens_out, resp.latency_ms)
        return ToolResult(f"Translation: {resp.content.strip()}", {"translation": resp.content.strip()})

    def tool_run_critic(self, **_: Any) -> ToolResult:
        from .critic import run_critic

        primary = self.primary_event()
        if not self.state.hypotheses:
            return ToolResult("No hypothesis to criticize yet; inspect evidence first.")
        hyp = self.state.hypotheses[-1]
        verdict = run_critic(self.llm, self.state, hyp, primary, self.telemetry, self.fixture)
        hyp.critic_verdict = verdict["verdict"]
        hyp.critic_notes = verdict.get("notes")
        support = {"supported": 0.35, "weakened": -0.45, "refuted": -1.2}.get(verdict["verdict"], 0.0)
        self.telemetry.increment("critic_calls")
        obs = (f"CRITIC verdict: {verdict['verdict'].upper()} — strongest alternative: {verdict.get('strongest_alternative')} | "
               f"missing evidence: {verdict.get('missing_evidence')} | notes: {verdict.get('notes')}")
        return ToolResult(obs, {"critic": verdict}, support=support)

    def tool_ask_user(self, question: str = "", language: str | None = None, options: list[str] | None = None,
                      timeout_s: float | None = None, **_: Any) -> ToolResult:
        primary = self.primary_event()
        kind = primary.kind if primary else "normal_activity"
        p = self.state.hypotheses[-1].probability if self.state.hypotheses else 0.5
        asked = any(q for q in self.state.questions)
        allowed, why = self.policy.should_ask_user(kind, p, asked, self.state.questions[-1].answer if asked else None,
                                                   can_answer=not self.subject_is_infant(primary))
        if not allowed and not asked:
            return ToolResult(f"POLICY: asking the person is not appropriate now — {why}.", {"denied": True})
        if asked:
            q = self.state.questions[-1]
            return ToolResult(f"Already asked: '{q.text}' → answer: {q.answer or 'pending'}", {"denied": True})
        language = language or self.language
        timeout_s = float(timeout_s or self.config.get("agent", {}).get("ask_timeout_s", 20))
        prof = self.policy.profile(kind)
        default = prof.get("silent_default") or "no_response"
        if not question:
            question = prof.get("check_question", {}).get(language) or prof.get("check_question", {}).get("en", "Are you okay?")
        q = UserQuestion(new_id("q"), question, language, options or ["yes", "no", "no_response"], timeout_s, default)
        self.state.questions.append(q)
        self.state.status = "waiting_user"
        self.executor.execute("ask_user", {"question": question}, [], question, language)
        answer = None
        if self.answer_provider is not None:
            answer = self.answer_provider(q)
        elif self.fixture.get("user_answer") is not None:
            answer = str(self.fixture["user_answer"])
        q.answer = answer or default
        q.answered_at = time.time()
        self.state.status = "running"
        norm = normalize_answer(q.answer)
        meaning = interpret_answer(kind, norm)
        support = {"no_response": 0.6, "needs_help": 0.7, "moderate": 0.4, "declines_help": -0.5, "unclear": 0.1}.get(meaning or "", 0.0)
        return ToolResult(f"Asked ({language}): \"{question}\" → the person answered: \"{q.answer}\" (interpreted: {meaning}).",
                          {"question": q.to_dict(), "normalized": norm, "meaning": meaning}, support=support)

    def tool_propose_action(self, action: str = "", rationale: str = "", params: dict[str, Any] | None = None, **_: Any) -> ToolResult:
        primary = self.primary_event()
        kind = primary.kind if primary else "normal_activity"
        hyp = self.state.hypotheses[-1] if self.state.hypotheses else None
        p = hyp.probability if hyp else 0.3
        severity = primary.severity if primary else 0.0
        answer = interpret_answer(kind, normalize_answer(self.state.questions[-1].answer)) if self.state.questions and self.state.questions[-1].answer else None
        gate = self.policy.gate_action(action, kind, p, severity, hyp.critic_verdict if hyp else None, answer,
                                       self.state.compute_budget - self.state.compute_spent)
        cost = self.policy.action_cost.get(action, 0.5)
        decision = ActionDecision(action, rationale, risk_if_wrong=round((1 - p) * cost, 3),
                                  expected_benefit=round(p * severity, 3), requires_confirmation=gate.requires_confirmation,
                                  executed=False, simulated=self.policy.simulate, params=params or {})
        if not gate.allowed:
            decision.result = f"REJECTED by policy: {gate.reason}"
            self.state.decisions.append(decision)
            self.guard.audit("action_rejected", {"action": action, "reason": gate.reason})
            return ToolResult(f"POLICY GATE rejected '{action}': {gate.reason}. Suggestion: {gate.suggestion}",
                              {"decision": decision.to_dict(), "gate": gate.__dict__})
        evidence_paths = [e.path for e in self.state.evidence.values() if e.path][:6]
        summary = rationale or (primary.summary if primary else "")
        result = self.executor.execute(action, params or {}, evidence_paths, summary, self.language)
        decision.executed = bool(result.get("ok"))
        decision.result = result.get("result", "SIMULATED_OK")
        decision.params = {**(params or {}), "dispatch": result.get("dispatch")}
        self.state.decisions.append(decision)
        self.guard.audit("action_executed", {"action": action, "simulated": decision.simulated})
        return ToolResult(f"ACTION '{action}' {'SIMULATED' if decision.simulated else 'EXECUTED'} — gate: {gate.reason}. "
                          f"Dispatch: {json.dumps(result.get('dispatch'), default=str)[:300]}",
                          {"decision": decision.to_dict()})

    def tool_finalize(self, answer: str = "", hypothesis: str = "", probability: float | None = None,
                      uncertainty: str = "", recommended_action: str = "", headline: str = "", **_: Any) -> ToolResult:
        self.state.final_answer = answer or hypothesis
        self.state.final_headline = headline or None
        self.state.final_uncertainty = uncertainty or None
        if hypothesis and self.state.hypotheses:
            self.state.hypotheses[-1].statement = hypothesis
        if probability is not None and self.state.hypotheses:

            p = self.state.hypotheses[-1].probability
            self.state.hypotheses[-1].probability = max(p - 0.15, min(p + 0.15, float(probability)))
        self.state.status = "done"
        return ToolResult("Investigation finalized.", {"answer": answer, "uncertainty": uncertainty,
                                                       "recommended_action": recommended_action}, stop=True)

def normalize_answer(answer: str | None) -> str:
    """Raw answer → {no_response, help, okay, yes, no, other}. 'okay' = the person states they are fine."""
    if answer is None:
        return "no_response"
    a = answer.strip().lower()
    if not a or a in {"no_response", "silence", "unresponsive", "..."}:
        return "no_response"
    if any(w in a for w in ("help", "ayuda", "ambulance", "ambulancia", "hospital", "hurt", "can't", "cannot", "no puedo")):
        return "help"
    if re.search(r"\b(i'?m|i am|im|estoy|todo|all|everything)\s+(okay|ok|fine|alright|good|bien)\b", a) or a in {"fine", "i'm fine", "im fine"}:
        return "okay"

    m = re.search(r"(?<![\d.])(10|[0-9])(?![\d.])\s*(/\s*10|out of 10|de 10|sobre 10)?", a)
    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
             "cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10}
    if m:
        return f"pain:{int(m.group(1))}"
    for w, n in words.items():
        if re.search(rf"\b{w}\b", a):
            return f"pain:{n}"
    if any(w in a for w in ("pain", "dolor")):
        return "help"
    if re.match(r"^(no|nope|nah|not)\b", a):
        return "no"
    if re.match(r"^(yes|yeah|yep|sí|si|claro|ok|okay|sure|please|por favor)\b", a) or re.search(r"\b(report|reportar|denunciar|report it)\b", a):
        return "yes"
    return "other"

def interpret_answer(kind: str, normalized: str | None) -> str | None:
    """Question-aware meaning: consent questions ('take you to the ER?') vs wellness checks ('are you okay?').
    Returns needs_help | declines_help | no_response | unclear | None."""
    if normalized is None:
        return None
    from .policy import KIND_PROFILE

    consent = bool(KIND_PROFILE.get(kind, {}).get("consent_needed", False))
    if normalized == "no_response":
        return "no_response"
    if normalized.startswith("pain:"):
        n = int(normalized.split(":")[1])
        return "needs_help" if n >= 7 else ("moderate" if n >= 4 else "declines_help")
    if normalized == "help":
        return "needs_help"
    if normalized == "okay":
        return "declines_help"
    if consent:
        return {"yes": "needs_help", "no": "declines_help"}.get(normalized, "unclear")
    return {"yes": "declines_help", "no": "needs_help"}.get(normalized, "unclear")
