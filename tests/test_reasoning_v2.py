"""Tests for the v2 reasoning fixes: scene inference API, verdicts, forced closure, no repeated tool
calls, infant-on-floor and toy-gun downgrades. All run on the fixture profile (no models)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ANGEL_MEMORY__PATH", str(Path(tempfile.gettempdir()) / "angel_test_memory.sqlite"))
os.environ.setdefault("ANGEL_OUTPUT__DIRECTORY", str(Path(tempfile.gettempdir()) / "angel_test_runs"))
os.environ.setdefault("ANGEL_SECURITY__AUDIT_LOG", str(Path(tempfile.gettempdir()) / "angel_test_runs" / "audit.jsonl"))

from personal_angel.agent.master import _signature, verdict_for
from personal_angel.perception.scene import (LOCATION_LABEL, SceneUnderstanding, note_location,
                                            understand_scene)
from personal_angel.schema import ActionDecision, Event, Hypothesis, InvestigationState

def test_scene_without_model_uses_note_only_for_strong_hints():
    su = understand_scene(None, {}, [], [], "rear camera of a robotaxi cabin")
    assert su.location == "CAR_CABIN" and su.backend == "note_only"
    su2 = understand_scene(None, {}, [], [], "please check this")
    assert su2.location == "ROOM_01"
    assert note_location("the baby is in the nursery") == "NURSERY"
    assert note_location("nothing special") is None
    assert LOCATION_LABEL["HOME_ROOM"] == "home interior"

def test_scene_description_and_age_lookup():
    from personal_angel.perception.scene import ObjectCheck, PersonProfile

    su = SceneUnderstanding(location="HOME_ROOM", label="home interior", confidence=0.8,
                            people=[PersonProfile(1, "baby", 0.9)], object_checks=[ObjectCheck("gun", 42, "toy", 0.12)])
    assert su.age_of(1) == "baby" and su.age_of(2) is None
    d = su.description()
    assert "baby" in d and "toy" in d and "home interior" in d
    assert su.to_dict()["people"][0]["age_group"] == "baby"

def _state_with(kind: str, severity: float, p: float, executed: list[str]) -> tuple[InvestigationState, Event]:
    st = InvestigationState(run_id="t", objective="o", scenario_hint=None, media_path="x.mp4", media_kind="video")
    ev = Event("evt_1", kind, 0.0, 2.0, "PERSON_01", "X", None, "HOME_ROOM", 0.7, severity, "s", [], {})
    st.events.append(ev)
    st.hypotheses.append(Hypothesis("h", kind, p, []))
    for a in executed:
        st.decisions.append(ActionDecision(a, "r", 0.1, 0.5, False, True, True, {}))
    return st, ev

def test_verdict_levels():
    st, ev = _state_with("weapon_visible", 0.9, 0.95, ["call_police_share_location"])
    assert verdict_for(st, ev)["level"] == "alert"
    st, ev = _state_with("fall", 0.7, 0.5, ["notify_owner"])
    assert verdict_for(st, ev)["level"] == "watch"
    st, ev = _state_with("weapon_visible", 0.9, 0.05, [])
    assert verdict_for(st, ev)["level"] == "clear"
    st, ev = _state_with("normal_activity", 0.0, 0.8, [])
    assert verdict_for(st, ev)["level"] == "clear"

def test_signature_is_order_independent():
    assert _signature("inspect_frames", {"start_s": 1, "end_s": 2}) == _signature("inspect_frames", {"end_s": 2, "start_s": 1})
    assert _signature("run_vlm", {"q": "a"}) != _signature("run_vlm", {"q": "b"})

def test_fall_rule_downgrades_slow_low_posture_transition():
    """A crawling person (already low) drifting to horizontal with no drop is 'person_down', not 'fall'."""
    from personal_angel.perception.pose import HumanStateAnalyzer
    from personal_angel.schema import PoseObservation
    from personal_angel.perception.fixtures import synth_keypoints

    an = HumanStateAnalyzer({"fall": {"drop_velocity_h_per_s": 0.6, "on_ground_dwell_s": 1.0}})
    t = 0.0
    for i in range(30):
        posture = "crouched" if i < 10 else "lying"
        box = (100.0, 200.0, 300.0, 330.0) if posture == "crouched" else (80.0, 240.0, 340.0, 340.0)
        kp = synth_keypoints(list(box), posture)
        if posture == "crouched":
            kp = [(x + (70.0 if j <= 8 else 0.0), y, c) for j, (x, y, c) in enumerate(kp)]
        an.observe([PoseObservation(track_id=1, keypoints=kp, box_xyxy=box)], t)
        t += 0.25
    kinds = {e.kind for e in an.analyze()}
    assert "fall" not in kinds
    assert "person_down" in kinds or not kinds

def test_report_never_ends_without_verdict(tmp_path):
    """Even with a 3-step budget the run closes with a real answer (forced critic + finalize)."""
    from personal_angel.config import load_profile
    from personal_angel.runner import run_investigation

    cfg = load_profile("fixture")
    cfg["agent"]["max_steps"] = 3
    cfg["output"]["directory"] = str(tmp_path)
    media = ROOT / "tests" / ".fixtures" / "car_gun_in_pocket.mp4"
    if not media.exists():
        import subprocess

        subprocess.run([sys.executable, "scripts/make_synthetic_scenarios.py", "--out", str(media.parent)], cwd=ROOT, check=True)
    r = run_investigation(media, "Decide what to do.", cfg, None, "budget3", lambda q: None)
    assert r["final_answer"] and "stopped at the step" not in r["final_answer"]
    assert r["verdict"]["level"] in {"clear", "watch", "alert"} and r["verdict"]["headline"]
    actions = [s["action"] for s in r["steps"]]
    assert actions[-1] == "finalize" and len(actions) <= 3
    assert "run_critic" in actions
    assert r["scene"]["location"] == "CAR_CABIN"

class _LoopingLLM:
    """A 'real' LLM that always asks for the same inspect_frames call, then answers finalize prompts."""
    is_real = True
    name = "looping_fake"

    def __init__(self) -> None:
        self.calls = 0

    def health(self):
        return {"ok": True, "backend": self.name}

    def chat(self, messages, stream=None, json_mode=False, max_tokens=None, images=None, temperature=None):
        from personal_angel.agent.llm import LLMResponse

        self.calls += 1
        content = messages[-1]["content"]
        if "Write the closing report JSON now" in content:
            body = '{"answer": "Synthesized closing report.", "uncertainty": "Sampled frames only.", "headline": "Weapon verified"}'
        elif "supports_hypothesis" in content:
            body = '{"description": "gun in hand", "answer": "real firearm", "supports_hypothesis": 0.8, "confidence": 0.8, "alternatives": []}'
        elif "verdict" in content.lower() and "critic" in content.lower():
            body = '{"verdict": "supported", "strongest_alternative": "toy", "missing_evidence": [], "notes": "ok"}'
        else:
            body = '{"thought": "look again", "action": "inspect_frames", "action_input": {"start_s": 3.0, "end_s": 9.0}}'
        return LLMResponse(content=body, reasoning="", model=self.name, tokens_in=10, tokens_out=5, latency_ms=1.0)

def test_llm_planner_cannot_loop_on_repeated_tool_calls(tmp_path):
    from personal_angel.agent.master import investigate
    from personal_angel.config import load_profile
    from personal_angel.edge_cloud import EdgeCloudPolicy
    from personal_angel.memory.store import create_memory
    from personal_angel.telemetry import Telemetry

    cfg = load_profile("fixture")
    cfg["agent"]["planner"] = "llm"
    cfg["agent"]["max_steps"] = 8
    media = ROOT / "tests" / ".fixtures" / "car_gun_in_pocket.mp4"
    llm = _LoopingLLM()
    run_dir = tmp_path / "loop"
    run_dir.mkdir()
    memory = create_memory(cfg["memory"], llm, Path(cfg["_project_root"]))
    events = list(investigate(media, "Decide.", cfg, run_dir, llm, memory, Telemetry(0), None, lambda q: None, EdgeCloudPolicy({})))
    report = events[-1]["report"]
    actions = [s["action"] for s in report["steps"]]
    assert actions.count("inspect_frames") == 1, actions
    assert actions[-1] == "finalize" and report["final_answer"].startswith("Synthesized closing report.")
    assert report["verdict"]["headline"] == "Weapon verified"
    assert any(e.get("type") == "warning" and "REJECTED" in e.get("detail", "") for e in events)
    memory.close()

def _audio_result(text: str, english: str, lang: str, toxicity: float, threat: float, acoustic: list[tuple[str, float]], kind: str = "audio"):
    from personal_angel.perception.pipeline import PerceptionResult
    from personal_angel.schema import AcousticEvent, AudioSegment, TextRiskScores

    r = PerceptionResult(media_path="x.wav", media_kind=kind, duration_s=12.0, location="PHONE_LINE")
    r.audio_segments = [AudioSegment(start_s=0.0, end_s=10.0, text=text, language=lang, confidence=0.9, translation_en=english)]
    r.acoustic_events = [AcousticEvent(start_s=i * 5.0, end_s=i * 5.0 + 5.0, label=lab, confidence=c) for i, (lab, c) in enumerate(acoustic)]
    r.text_risk = TextRiskScores(threat=threat, hate=0.05, toxicity=toxicity, model="test", matched_cues=[])
    r.audio_kind = "music" if sum(1 for lab, _ in acoustic if lab in {"music", "singing"}) >= max(1, len(acoustic) // 2) else "speech"
    return r

def test_song_is_music_not_a_threat(tmp_path):
    from personal_angel.events.builder import build_events

    r = _audio_result("I will fight them all and burn it down", "I will fight them all and burn it down", "en", 0.7, 0.8,
                      [("music", 0.9), ("singing", 0.8), ("music", 0.85)])
    events, _ = build_events(r, tmp_path)
    kinds = [e.kind for e in events]
    assert "threatening_speech" in kinds
    ev = next(e for e in events if e.kind == "threatening_speech")
    assert ev.severity <= 0.25 and ev.attributes.get("music") is True and "SUNG_LYRICS" == ev.action

def test_abusive_speech_asks_then_advises_or_reports(tmp_path):
    from personal_angel.agent.policy import InvestigationPolicy
    from personal_angel.agent.tools import interpret_answer, normalize_answer
    from personal_angel.events.builder import build_events

    r = _audio_result("Eres un inútil, cállate", "You are useless, shut up", "es", 0.75, 0.1, [("angry argument", 0.6)])
    events, _ = build_events(r, tmp_path)
    assert any(e.kind == "abusive_speech" for e in events)
    pol = InvestigationPolicy({"allowlist": ["advise_user", "notify_owner", "log_only"], "action_cost": {"notify_owner": 0.1}}, {})
    ok, why = pol.should_ask_user("abusive_speech", 0.7, False, None)
    assert ok, why
    assert pol.recommend("abusive_speech", 0.7, interpret_answer("abusive_speech", normalize_answer("no"))) == ["advise_user"]
    assert pol.recommend("abusive_speech", 0.7, interpret_answer("abusive_speech", normalize_answer("yes, report it"))) == ["notify_owner"]
    assert pol.gate_action("advise_user", "abusive_speech", 0.7, 0.45, None, "declines_help", 50).allowed

def test_infants_are_never_asked():
    from personal_angel.agent.policy import InvestigationPolicy

    pol = InvestigationPolicy({}, {})
    ok, why = pol.should_ask_user("person_down", 0.6, False, None, can_answer=False)
    assert not ok and "infant" in why

def _fight_result(n_strikes: int, n_falls: int, activity: float = 0.0):
    """A perception result whose human events fragment into several strikers and people going down (CCTV brawl)."""
    from personal_angel.perception.pipeline import PerceptionResult
    from personal_angel.perception.pose import HumanStateEvent as HumanEvent
    from personal_angel.perception.scene import SceneUnderstanding

    from personal_angel.perception.fixtures import synth_keypoints
    from personal_angel.schema import Detection, FrameObservation, PoseObservation

    r = PerceptionResult(media_path="x.mp4", media_kind="video", duration_s=5.0, location="RETAIL")
    r.scene = SceneUnderstanding(location="RETAIL", label="restaurant/bar", confidence=0.7, activity={"fight": activity, "calm": 0.5})

    tracks = list(range(10, 10 + n_strikes)) + list(range(30, 30 + n_falls)) + [50]
    for k in range(20):
        ts = k * 0.25
        dets, poses = [], []
        for j, tid in enumerate(tracks):
            box = (100.0 + 40.0 * j + 3.0 * k, 120.0, 130.0 + 40.0 * j + 3.0 * k, 200.0)
            dets.append(Detection(label="person", confidence=0.7, box_xyxy=box, track_id=tid))
            poses.append(PoseObservation(track_id=tid, keypoints=synth_keypoints(list(box), "standing"), box_xyxy=box))
        r.dense.append(FrameObservation(frame_index=k, timestamp_s=ts, detections=dets, poses=poses))
    t = 0.5
    for i in range(n_strikes):
        r.human_events.append(HumanEvent("striking_motion", 10 + i, t, t + 0.8, 0.6, {"fast_wrist_frames": 9},
                                         f"PERSON_{10 + i:02d} made 9 rapid arm swings"))
        t += 0.7
    for i in range(n_falls):
        r.human_events.append(HumanEvent("fall", 30 + i, 1.0 + i * 0.6, 2.0 + i * 0.6, 0.45, {"fall_observed": True},
                                         f"PERSON_{30 + i:02d}: a fast downward motion started"))
    return r

def test_group_fight_consolidates_fragmented_strikes(tmp_path):
    from personal_angel.agent.master import _belief_floor, _perception_strength, initial_belief
    from personal_angel.events.builder import build_events
    from personal_angel.schema import InvestigationState

    events, _ = build_events(_fight_result(3, 2, activity=0.4), tmp_path)
    fights = [e for e in events if e.kind == "aggressive_interaction"]
    assert len(fights) == 1 and fights[0].attributes.get("group_fight") is True
    assert fights[0].attributes["strike_events"] == 3 and fights[0].attributes["falls_during"] == 2
    assert fights[0].confidence >= 0.85 and "brawl" in fights[0].summary.lower()
    assert all(e.attributes.get("during_fight") for e in events if e.kind == "fall")
    st = InvestigationState(run_id="t", objective="o", scenario_hint=None, media_path="x.mp4", media_kind="video")
    st.events = events
    hyp, primary = initial_belief(st)
    assert primary.kind == "aggressive_interaction" and "fighting" in hyp.statement.lower() and "roles unknown" in hyp.statement
    assert _belief_floor(primary, st) >= 0.5 and _perception_strength(primary) >= 0.9

def test_single_strike_is_not_a_brawl(tmp_path):
    from personal_angel.events.builder import build_events

    events, _ = build_events(_fight_result(1, 0), tmp_path)
    fights = [e for e in events if e.kind == "aggressive_interaction"]
    assert len(fights) == 1 and not fights[0].attributes.get("group_fight")

def test_critic_role_swap_does_not_refute_a_fight():
    from personal_angel.agent.critic import reconcile_verdict

    v, why = reconcile_verdict("aggressive_interaction", "refuted",
                               "PERSON_01 is a victim of a chaotic brawl, reacting defensively to being struck by others",
                               "visual inspection shows multiple people attacking PERSON_01 simultaneously", "")
    assert v == "supported" and "roles differ" in why
    v, _ = reconcile_verdict("aggressive_interaction", "weakened", "The rapid arm movements could be playful gestures", "", "")
    assert v == "weakened"
    v, _ = reconcile_verdict("aggressive_interaction", "refuted", "Friends hugging and dancing at a party", "no contact seen", "")
    assert v == "refuted"
    assert reconcile_verdict("weapon_visible", "refuted", "the object is a toy gun", "vlm: toy", "")[0] == "refuted"

def test_tiny_figures_get_no_age_and_never_count_as_children():
    from personal_angel.perception.scene import FixtureSceneAnalyzer, understand_scene
    from personal_angel.schema import Detection, FrameObservation

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    obs = FrameObservation(frame_index=0, timestamp_s=0.0, detections=[
        Detection(label="person", confidence=0.8, box_xyxy=(10.0, 10.0, 30.0, 45.0), track_id=5),
        Detection(label="person", confidence=0.8, box_xyxy=(100.0, 40.0, 220.0, 330.0), track_id=6),
    ])
    su = understand_scene(FixtureSceneAnalyzer({"location": "RETAIL"}, None), {0: frame}, [obs], [], None)
    small = next(p for p in su.people if p.track_id == 5)
    assert small.age_group == "unknown" and su.age_of(5) is None and "too small" in su.description()
    assert su.people[0].track_id == 6

def test_gunshot_corroborates_weapon_and_no_question_is_asked(tmp_path):
    from personal_angel.agent.master import HeuristicPlanner, _belief_floor, initial_belief
    from personal_angel.agent.policy import InvestigationPolicy
    from personal_angel.events.builder import build_events
    from personal_angel.perception.pipeline import PerceptionResult, WeaponSighting
    from personal_angel.schema import AcousticEvent, InvestigationState

    r = PerceptionResult(media_path="x.mp4", media_kind="video", duration_s=3.0, location="OUTDOOR")
    w = WeaponSighting(track_id=7, label="gun", first_s=0.0, last_s=3.0, frames_seen=28, frames_possible=30, max_conf=0.97,
                       holder_track=1, evidence_frames=[0, 5, 10], second_opinion="weapon", weapon_probability=1.0)
    r.weapons = [w]
    r.acoustic_events = [AcousticEvent(start_s=0.0, end_s=1.0, label="gunshot", confidence=0.97)]
    events, _ = build_events(r, tmp_path)
    weapon = next(e for e in events if e.kind == "weapon_visible")
    assert weapon.attributes.get("gunshot_heard") == 0.97 and "toy does not fire" in weapon.summary
    st = InvestigationState(run_id="t", objective="o", scenario_hint=None, media_path="x.mp4", media_kind="video")
    st.events = events
    hyp, primary = initial_belief(st)
    assert primary.kind == "weapon_visible" and _belief_floor(primary, st) >= 0.75
    pol = InvestigationPolicy({}, {})
    assert not pol.should_ask_user("weapon_visible", 0.8, False, None)[0]
