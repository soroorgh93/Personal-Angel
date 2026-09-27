"""Component tests: server routes, evaluation scoring, semantic cache, edge/cloud policy, triage fixture."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ANGEL_MEMORY__PATH", str(Path(tempfile.gettempdir()) / "angel_test_memory.sqlite"))
os.environ.setdefault("ANGEL_OUTPUT__DIRECTORY", str(Path(tempfile.gettempdir()) / "angel_test_runs"))
os.environ.setdefault("ANGEL_SECURITY__AUDIT_LOG", str(Path(tempfile.gettempdir()) / "angel_test_runs" / "audit.jsonl"))
FIXTURES = ROOT / "tests" / ".fixtures"

from personal_angel.edge_cloud import EdgeCloudPolicy
from personal_angel.evaluation import score_case, summarize
from personal_angel.memory.semantic_cache import SemanticCache
from personal_angel.perception.triage import FixtureTriage

def test_edge_cloud_local_only_never_marks_cloud():
    pol = EdgeCloudPolicy({"mode": "local_only", "privacy_sensitive_kinds": ["video", "audio"]})
    d = pol.decide("vlm", ["video"], latency_budget_ms=500)
    assert d.placement == "local" and "privacy" in d.reasons and "latency" in d.reasons
    hybrid = EdgeCloudPolicy({"mode": "hybrid", "cloud_allowed_for": ["model_updates"], "privacy_sensitive_kinds": ["video"]})
    assert hybrid.decide("model_updates", []).placement == "cloud_eligible"
    assert hybrid.decide("model_updates", ["video"]).placement == "local"

def test_semantic_cache_exact_and_semantic_hits():
    cache = SemanticCache(threshold=0.5)
    cache.put("run1", "Which person had the knife and when?", "PERSON_01 at 5s")
    assert cache.get("run1", "which person had the knife and when?")["kind"] == "exact"
    hit = cache.get("run1", "Which person had the knife, and when was it?")
    assert hit is not None and hit["kind"] in {"semantic", "exact"}
    assert cache.get("run2", "Which person had the knife and when?") is None
    assert cache.stats()["hit_rate"] is not None

def test_score_case_and_summary():
    report = {"events": [{"kind": "fall", "start_s": 4.0, "end_s": 18.0}], "hypothesis": {"critic_verdict": "supported"},
              "executed_actions": ["call_911", "notify_owner"], "questions": [{"language": "en"}],
              "steps": [{"step": 1, "action": "inspect_frames", "observation": "ok"}, {"step": 2, "action": "run_critic", "observation": "x"}],
              "decisions": [{"executed": True, "action": "call_911"}], "belief": 0.9,
              "telemetry": {"wall_time_s": 1.0, "cost": {"cloud_equivalent_usd": 0.01, "local_energy_cost_usd": 0.0}, "model_calls": 2,
                            "input_tokens": 10, "output_tokens": 5, "model_latency_ms_total": 3, "frames_processed": 5, "frames_total": 10},
              "compute": {"spent": 20}}
    case = {"id": "x", "expected": {"kinds": ["fall"], "window": [4, 18], "must_ask": True, "language": "en",
                                     "actions": ["call_911", "notify_owner"], "forbidden": [], "critical": True}}
    row = score_case(case, report)
    assert row["application"]["task_success"] and row["model"]["temporal_iou"] == 1.0
    summary = summarize([row])
    assert summary["full"]["task_success_rate"] == 1.0

def test_triage_fixture_scores_weapon_frames():
    sidecar = FIXTURES / "car_gun_in_pocket.mp4.fixture.json"
    if not sidecar.exists():
        subprocess.run([sys.executable, "scripts/make_synthetic_scenarios.py", "--out", str(FIXTURES)], cwd=ROOT, check=True)
    fixture = json.loads(sidecar.read_text())
    tri = FixtureTriage(fixture)
    assert tri.score(None, 8.0)["weapon_visible"] > 0.5
    assert tri.score(None, 1.0)["normal"] > 0.5

def test_server_routes_end_to_end():
    from starlette.testclient import TestClient

    from personal_angel.server.app import create_app

    app = create_app("fixture")
    client = TestClient(app)
    health = client.get("/api/health").json()
    assert health["ok"] and health["local_only"]
    assert isinstance(client.get("/api/library").json()["items"], list)
    clip = FIXTURES / "normal_room_walk.mp4"
    if not clip.exists():
        subprocess.run([sys.executable, "scripts/make_synthetic_scenarios.py", "--out", str(FIXTURES)], cwd=ROOT, check=True)
    run = client.post("/api/runs", json={"media_path": str(clip.relative_to(ROOT)), "question": "anything?", "context": ""}).json()
    run_id = run["run_id"]
    assert client.post("/api/runs", json={"media_path": "../../etc/passwd"}).status_code == 404
    for _ in range(100):
        rep = client.get(f"/api/runs/{run_id}/report")
        if rep.status_code == 200:
            break
        time.sleep(0.3)
    report = rep.json()
    assert report["executed_actions"] == [] and report["final_answer"]
    assert report["verdict"]["level"] == "clear" and report["scene"]["location"]
    chat = client.post(f"/api/runs/{run_id}/chat", json={"question": "what happened?"}).json()
    assert "answer" in chat
    chat2 = client.post(f"/api/runs/{run_id}/chat", json={"question": "what happened?"}).json()
    assert chat2["cache"] == "exact"
