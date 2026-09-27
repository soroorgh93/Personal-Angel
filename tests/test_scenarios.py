"""End-to-end agent-loop tests on synthetic fixtures (software tests, not accuracy claims)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / ".fixtures"
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANGEL_MEMORY__PATH", str(Path(tempfile.gettempdir()) / "angel_test_memory.sqlite"))
os.environ.setdefault("ANGEL_OUTPUT__DIRECTORY", str(Path(tempfile.gettempdir()) / "angel_test_runs"))
os.environ.setdefault("ANGEL_SECURITY__AUDIT_LOG", str(Path(tempfile.gettempdir()) / "angel_test_runs" / "audit.jsonl"))

from personal_angel.runner import run_investigation

@pytest.fixture(scope="session", autouse=True)
def synthetic_media():
    if not (DATA / "car_pregnant_passenger.mp4").exists():
        subprocess.run([sys.executable, "scripts/make_synthetic_scenarios.py", "--out", str(DATA)], cwd=ROOT, check=True)

def _run(name: str, answer: str | None = None, ask: str = "Did anything dangerous happen? Decide what to do."):
    provider = (lambda q: answer) if answer is not None else (lambda q: None)
    return run_investigation(DATA / name, ask, "fixture", None, f"pytest-{name}", provider)

def test_car_pregnant_no_answer_reroutes_and_calls_911():
    r = _run("car_pregnant_passenger.mp4", answer=None)
    kinds = {e["kind"] for e in r["events"]}
    assert {"slump_unresponsive", "distress_speech"} <= kinds
    assert r["questions"] and r["questions"][0]["language"] == "es"
    assert r["executed_actions"] == ["reroute_to_hospital", "call_911"]
    assert r["belief"] > 0.8
    assert any(s["action"] == "run_critic" for s in r["steps"])
    assert r["disclosure"]["all_consequential_actions_simulated"]

def test_car_pregnant_declines_continues_route():
    r = _run("car_pregnant_passenger.mp4", answer="No, estoy bien")
    assert r["executed_actions"] == ["continue_route"]
    rejected = [d for d in r["decisions"] if not d["executed"]]
    assert all("reroute" not in d["action"] for d in rejected) or any("declined" in (d["result"] or "") for d in rejected)

def test_elder_fall_asks_then_calls_911():
    r = _run("elder_fall_livingroom.mp4", answer=None)
    assert any(e["kind"] == "fall" for e in r["events"])
    fall = next(e for e in r["events"] if e["kind"] == "fall")
    assert 3.5 <= fall["start_s"] <= 6.0
    assert fall["attributes"]["on_ground_dwell_s"] >= 2.0
    assert r["questions"][0]["language"] == "en"
    assert "call_911" in r["executed_actions"]

def test_gun_in_car_never_asks_and_alerts_police():
    r = _run("car_gun_in_pocket.mp4")
    assert any(e["kind"] == "weapon_visible" and e["attributes"]["is_firearm"] for e in r["events"])
    assert r["questions"] == []
    assert r["executed_actions"][0] == "call_police_share_location"
    steps = [s["action"] for s in r["steps"]]
    assert steps.index("run_critic") < steps.index("propose_action")

def test_nursery_aggression_notifies_parents():
    r = _run("nursery_caregiver_baby.mp4")
    ev = next(e for e in r["events"] if e["kind"] == "aggressive_interaction")
    assert ev["obj"] == "PERSON_02" and ev["attributes"]["crying_heard"]
    assert "notify_parents" in r["executed_actions"]
    assert r["questions"] == []

def test_spanish_voicemail_translated_and_reported():
    r = _run("spanish_threat_voicemail.wav")
    ev = next(e for e in r["events"] if e["kind"] == "threatening_speech")
    assert ev["attributes"]["language"] == "es"
    assert "kill" in ev["attributes"]["english"].lower()
    assert any("matar" in c for c in ev["attributes"]["cues"])
    assert "call_police_share_location" in r["executed_actions"]

def test_normal_video_logs_only_and_skips_vlm():
    r = _run("normal_room_walk.mp4")
    assert r["events"][0]["kind"] == "normal_activity"
    assert r["executed_actions"] == []
    assert not any(s["action"] == "run_vlm" for s in r["steps"])
    assert r["telemetry"]["frames_skipped"] > 0

def test_report_artifacts_written(tmp_path):
    r = _run("normal_room_walk.mp4")
    run_dir = Path(os.environ["ANGEL_OUTPUT__DIRECTORY"]) / r["run_id"]
    for name in ("report.json", "events.jsonl", "reasoning_trace.jsonl", "telemetry.json"):
        assert (run_dir / name).exists()
    data = json.loads((run_dir / "report.json").read_text())
    assert data["edge_cloud"]["summary"]["cloud_eligible"] == 0
