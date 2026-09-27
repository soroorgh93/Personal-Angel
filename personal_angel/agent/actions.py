"""Action executors. Every consequential action is SIMULATED for the project:
it produces a realistic, timestamped dispatch record (what would be sent to
whom, with which evidence) but never contacts the outside world. The
`ActionExecutor` is the only place that could ever be wired to a real
integration, behind the permission table in safety.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "ask_user": {"label": "Ask the person a question", "external": False, "icon": "💬"},
    "continue_route": {"label": "Continue the planned route / keep monitoring", "external": False, "icon": "🟢"},
    "log_only": {"label": "Log the observation, no escalation", "external": False, "icon": "📝"},
    "advise_user": {"label": "Give the person safety advice (no report filed)", "external": False, "icon": "🧭"},
    "reroute_to_hospital": {"label": "Reroute the vehicle to the nearest hospital ER and notify the hospital", "external": True, "icon": "🏥"},
    "call_911": {"label": "Call 911 (medical emergency) and share the live location", "external": True, "icon": "🚑"},
    "call_police_share_location": {"label": "Call 911 (police) and share the live location + evidence", "external": True, "icon": "🚔"},
    "notify_parents": {"label": "Call/notify the family (parents or guardians) with the evidence clip", "external": True, "icon": "👪"},
    "notify_owner": {"label": "Notify the owner/operator", "external": True, "icon": "📣"},
    "notify_security": {"label": "Notify on-site security", "external": True, "icon": "🛡️"},
}

class ActionExecutor:
    def __init__(self, run_dir: Path, simulate: bool = True, location_hint: str = "unknown") -> None:
        self.run_dir = run_dir
        self.simulate = simulate
        self.location_hint = location_hint
        self.log_path = run_dir / "actions.jsonl"

    def execute(self, action: str, params: dict[str, Any], evidence_paths: list[str], summary: str,
                language: str = "en") -> dict[str, Any]:
        spec = ACTION_CATALOG.get(action)
        if spec is None:
            return {"ok": False, "error": f"unknown action {action}"}
        record: dict[str, Any] = {"action": action, "label": spec["label"], "simulated": self.simulate or not spec["external"],
                                  "timestamp": time.time(), "params": params, "evidence": evidence_paths[:6]}
        gps = params.get("location") or {"lat": 37.3352, "lon": -121.8811, "label": "San José State University (demo)"}
        if action == "reroute_to_hospital":
            record["dispatch"] = {"to": "vehicle_navigation + hospital ER desk", "destination": "Regional Medical Center ER (nearest, 2.1 mi, 7 min)",
                                  "mode": "safe_pullover_then_reroute", "hospital_notified": "incoming passenger, ETA 7 min (simulated)",
                                  "location": gps, "message": summary}
        elif action == "call_911":
            record["dispatch"] = {"to": "PSAP 911 (simulated)", "caller": "PersonalAngel autonomous cabin/room monitor",
                                  "location": gps, "nature": params.get("nature", summary[:160]),
                                  "callback": params.get("callback", "operator console")}
        elif action == "call_police_share_location":
            record["dispatch"] = {"to": "Police non-emergency / 911 (simulated)", "location": gps,
                                  "report": params.get("report", summary[:240]), "evidence_shared": evidence_paths[:4],
                                  "do_not_confront": True}
        elif action == "notify_parents":
            record["dispatch"] = {"to": params.get("contacts", ["parent_primary", "parent_secondary"]),
                                  "channel": "push+sms (simulated)", "message": summary[:240], "evidence_shared": evidence_paths[:4]}
        elif action == "notify_owner":
            record["dispatch"] = {"to": "owner/operator", "channel": "push (simulated)", "message": summary[:240]}
        elif action == "notify_security":
            record["dispatch"] = {"to": "on-site security", "channel": "radio/app (simulated)", "message": summary[:240],
                                  "evidence_shared": evidence_paths[:4]}
        elif action == "continue_route":
            record["dispatch"] = {"to": "vehicle_navigation", "message": "continue route; monitoring at elevated rate for 10 min"}
        elif action == "advise_user":
            record["dispatch"] = {"to": "the person (console / speaker)", "advice": params.get("advice") or [
                "Keep this recording and the transcript — they are evidence if it continues.",
                "Do not reply or engage; block the number / account if it is a call or message.",
                "Tell a trusted person, and your employer/school/platform if it is related to them.",
                "If it turns into a threat to your safety, call 911 (or the local emergency number) — this system will escalate automatically.",
                "Support lines: 988 (US, crisis) · local domestic-abuse or harassment hotline."]}
        elif action == "log_only":
            record["dispatch"] = {"to": "audit_log"}
        elif action == "ask_user":
            record["dispatch"] = {"to": "cabin_speaker/screen", "language": language, "text": params.get("question")}
        record["result"] = "SIMULATED_OK" if record["simulated"] else "EXECUTED"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        return {"ok": True, **record}
