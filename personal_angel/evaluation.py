"""Four-layer evaluation (DATA-236 lecture 09 / GEMMAS-style):

  model layer        – event detection P/R per kind, temporal IoU, latency, tokens
  application layer  – task success (right action set), ask/not-ask correctness,
                       language, false-escalation and missed-critical rates
  process layer      – steps, tool-selection accuracy, unnecessary-path ratio (UPR),
                       critic veto rate, policy rejections recovered
  business layer     – cloud-equivalent cost, local energy, wall time, compute units
plus an LLM-as-judge (G-Eval style rubric) for reasoning quality / faithfulness.

A case manifest is JSONL:
{"id": "...", "media": "tests/.fixtures/x.mp4", "question": "...", "hint": "car_cabin",
 "answer": null, "expected": {"kinds": ["slump_unresponsive"], "window": [8, 24],
 "must_ask": true, "language": "es", "actions": ["reroute_to_hospital", "call_911"],
 "forbidden": ["call_police_share_location"], "critical": true}}
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .agent.llm import create_llm, extract_json_object
from .config import load_profile
from .runner import run_investigation

INFO_TOOLS = {"inspect_frames", "run_vlm", "run_audio_analysis", "run_pose_analysis", "query_memory", "retrieve_policy", "translate", "run_critic"}
HIGH_COST = {"call_911", "call_police_share_location", "reroute_to_hospital", "notify_parents"}

VARIANTS: dict[str, dict[str, Any]] = {
    "full": {},
    "no_critic": {"agent": {"critic_enabled": False}, "policy": {"require_critic": False}},
    "no_ask_user": {"agent": {"ask_user_enabled": False}},
    "no_vlm": {"agent": {"skip_tools": ["run_vlm"]}},
    "no_memory_rag": {"agent": {"skip_tools": ["query_memory", "retrieve_policy"]}},
    "no_pagerank": {"graph": {"pagerank_enabled": False}},
    "fixed_pipeline": {"agent": {"critic_enabled": False, "ask_user_enabled": False, "skip_tools": ["query_memory", "retrieve_policy"]},
                       "policy": {"require_critic": False}},
    "no_audio": {"audio": {"backend": "fixture", "_mute": True}},
}

def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out

def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0

def score_case(case: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    exp = case.get("expected", {})
    kinds_found = {e["kind"] for e in report["events"]}
    exp_kinds = set(exp.get("kinds", []))
    tp = len(kinds_found & exp_kinds)
    precision = tp / max(len(kinds_found - {"normal_activity"}), 1) if exp_kinds else (1.0 if kinds_found <= {"normal_activity"} else 0.0)
    recall = tp / max(len(exp_kinds), 1) if exp_kinds else 1.0
    primary = report.get("hypothesis", {}) or {}
    iou = None
    if exp.get("window") and exp_kinds:
        ev = next((e for e in report["events"] if e["kind"] in exp_kinds), None)
        if ev:
            iou = temporal_iou((ev["start_s"], ev["end_s"]), tuple(exp["window"]))
    executed = report.get("executed_actions", [])
    exp_actions = exp.get("actions", [])
    forbidden = set(exp.get("forbidden", []))
    action_exact = set(executed) == set(exp_actions)
    action_recall = len(set(executed) & set(exp_actions)) / max(len(exp_actions), 1) if exp_actions else (1.0 if not executed or executed == ["log_only"] else 0.0)
    forbidden_hit = bool(forbidden & set(executed))
    asked = bool(report.get("questions"))
    ask_ok = asked == bool(exp.get("must_ask", False))
    lang_ok = (not asked) or (not exp.get("language")) or report["questions"][0]["language"] == exp.get("language")
    critical = bool(exp.get("critical", False))
    escalated = any(a in HIGH_COST for a in executed)
    false_escalation = escalated and not critical and not exp_actions
    missed_critical = critical and not escalated
    steps = report.get("steps", [])
    belief_changes = {s["step"] for s in steps if s.get("observation") and "support" in s.get("observation", "")}

    tool_seq = [s["action"] for s in steps]
    unnecessary = 0
    for s in steps:
        obs = s.get("observation") or ""
        if s["action"] in INFO_TOOLS and ("DENIED" in obs or "no relevant" in obs.lower() or "No policy" in obs):
            unnecessary += 1
    upr = unnecessary / max(len(steps), 1)
    rejections = [d for d in report.get("decisions", []) if not d.get("executed")]
    recovered = sum(1 for d in rejections if any(x.get("executed") for x in report.get("decisions", [])))
    critic_steps = [s for s in steps if s["action"] == "run_critic"]
    critic_verdict = (report.get("hypothesis") or {}).get("critic_verdict")
    t = report.get("telemetry", {})
    return {
        "case": case.get("id"), "variant": case.get("_variant", "full"),
        "model": {"event_precision": round(precision, 3), "event_recall": round(recall, 3), "temporal_iou": None if iou is None else round(iou, 3),
                  "belief": report.get("belief"), "model_calls": t.get("model_calls"), "tokens_in": t.get("input_tokens"), "tokens_out": t.get("output_tokens"),
                  "model_latency_ms": t.get("model_latency_ms_total")},
        "application": {"task_success": bool(action_exact and ask_ok and not forbidden_hit), "action_exact": action_exact, "action_recall": round(action_recall, 3),
                        "forbidden_action": forbidden_hit, "ask_correct": ask_ok, "language_correct": lang_ok,
                        "false_escalation": false_escalation, "missed_critical": missed_critical, "executed": executed},
        "process": {"steps": len(steps), "tool_sequence": tool_seq, "unnecessary_path_ratio": round(upr, 3),
                    "policy_rejections": len(rejections), "rejections_recovered": recovered, "critic_ran": bool(critic_steps),
                    "critic_verdict": critic_verdict, "critic_veto": critic_verdict == "refuted"},
        "business": {"wall_time_s": t.get("wall_time_s"), "cloud_equivalent_usd": (t.get("cost") or {}).get("cloud_equivalent_usd"),
                     "local_energy_usd": (t.get("cost") or {}).get("local_energy_cost_usd"), "compute_units": (report.get("compute") or {}).get("spent"),
                     "frames_processed": t.get("frames_processed"), "frames_total": t.get("frames_total")},
    }

JUDGE_RUBRIC = """You are grading the reasoning trace of a safety-investigation agent (G-Eval style).
Score 1-5 for each criterion and explain briefly:
1. faithfulness: every claim in the final answer is supported by cited evidence/events (no invented facts)
2. reasoning_quality: steps are logical, cheap evidence precedes expensive verification, the critic was consulted before escalation
3. action_appropriateness: the action set matches the evidence and the harm asymmetry (never confronts a weapon holder; asks consent for medical reroute)
4. clarity: the operator can understand what happened and why from the trace alone
Reply with JSON: {"faithfulness": n, "reasoning_quality": n, "action_appropriateness": n, "clarity": n, "notes": str}"""

def judge(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    llm = create_llm(config.get("llm", {}))
    if not llm.is_real:

        steps = [s["action"] for s in report.get("steps", [])]
        vlm_after_inspect = ("run_vlm" not in steps) or (steps.index("run_vlm") > steps.index("inspect_frames") if "inspect_frames" in steps else True)
        critic_before_action = ("propose_action" not in steps) or ("run_critic" in steps and steps.index("run_critic") < steps.index("propose_action"))
        return {"faithfulness": 5 if report.get("hypothesis", {}) and report["hypothesis"].get("evidence_ids") else 3,
                "reasoning_quality": 5 if vlm_after_inspect and critic_before_action else 3,
                "action_appropriateness": 5 if not any(d.get("executed") and d["action"] == "ask_user" for d in report.get("decisions", [])) else 4,
                "clarity": 4, "notes": "rule-based judge (no LLM in this profile)", "judge": "rules"}
    trace = json.dumps({"final_answer": report["final_answer"], "hypothesis": report["hypothesis"], "events": report["events"][:6],
                        "steps": [{"thought": s["thought"], "action": s["action"], "observation": (s.get("observation") or "")[:300]} for s in report["steps"]],
                        "decisions": report["decisions"]}, default=str)[:14000]
    resp = llm.chat([{"role": "system", "content": JUDGE_RUBRIC}, {"role": "user", "content": trace}], json_mode=True, max_tokens=400)
    data = extract_json_object(resp.content) or {}
    data["judge"] = resp.model
    return data

def run_suite(manifest: Path, profile: str, variants: list[str], out_dir: Path, with_judge: bool = True) -> dict[str, Any]:
    cases = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = load_profile(profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        config = _merge(base, VARIANTS.get(variant, {}))
        if VARIANTS.get(variant, {}).get("audio", {}).get("_mute"):
            config["audio"]["backend"] = "fixture"
            config["audio"]["_mute"] = True
        for case in cases:
            media = Path(base["_project_root"]) / case["media"]
            answer = case.get("answer")
            provider = (lambda q, a=answer: a) if answer is not None else (lambda q: None)
            t0 = time.perf_counter()
            report = run_investigation(media, case.get("question", "Did anything dangerous happen? Decide what to do."), config,
                                       case.get("hint"), f"eval-{variant}-{case['id']}", provider)
            row = score_case({**case, "_variant": variant}, report)
            row["business"]["eval_wall_s"] = round(time.perf_counter() - t0, 2)
            if with_judge:
                row["judge"] = judge(report, config)
            rows.append(row)
            print(f"[{variant}] {case['id']}: success={row['application']['task_success']} actions={row['application']['executed']} steps={row['process']['steps']}")
    summary = summarize(rows)
    (out_dir / "results.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2, default=str))
    (out_dir / "results.md").write_text(to_markdown(summary))
    return {"rows": rows, "summary": summary}

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_variant.setdefault(r["variant"], []).append(r)
    summary = {}
    for variant, items in by_variant.items():
        def mean(path: list[str]) -> float | None:
            vals = []
            for it in items:
                v: Any = it
                for p in path:
                    v = (v or {}).get(p) if isinstance(v, dict) else None
                if isinstance(v, bool):
                    v = float(v)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            return round(statistics.mean(vals), 3) if vals else None
        summary[variant] = {
            "cases": len(items),
            "task_success_rate": mean(["application", "task_success"]),
            "action_recall": mean(["application", "action_recall"]),
            "false_escalation_rate": mean(["application", "false_escalation"]),
            "missed_critical_rate": mean(["application", "missed_critical"]),
            "ask_correct_rate": mean(["application", "ask_correct"]),
            "event_precision": mean(["model", "event_precision"]), "event_recall": mean(["model", "event_recall"]),
            "temporal_iou": mean(["model", "temporal_iou"]),
            "mean_steps": mean(["process", "steps"]), "unnecessary_path_ratio": mean(["process", "unnecessary_path_ratio"]),
            "critic_run_rate": mean(["process", "critic_ran"]),
            "mean_model_calls": mean(["model", "model_calls"]), "mean_tokens_out": mean(["model", "tokens_out"]),
            "mean_wall_s": mean(["business", "wall_time_s"]), "mean_cloud_equiv_usd": mean(["business", "cloud_equivalent_usd"]),
            "mean_compute_units": mean(["business", "compute_units"]),
            "judge_faithfulness": mean(["judge", "faithfulness"]), "judge_reasoning": mean(["judge", "reasoning_quality"]),
            "judge_actions": mean(["judge", "action_appropriateness"]),
        }
    return summary

def to_markdown(summary: dict[str, Any]) -> str:
    cols = ["task_success_rate", "false_escalation_rate", "missed_critical_rate", "ask_correct_rate", "event_recall", "temporal_iou",
            "mean_steps", "unnecessary_path_ratio", "mean_model_calls", "mean_tokens_out", "mean_wall_s", "mean_cloud_equiv_usd", "judge_reasoning"]
    lines = ["| variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for variant, s in summary.items():
        lines.append(f"| {variant} | " + " | ".join("—" if s.get(c) is None else str(s[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"
