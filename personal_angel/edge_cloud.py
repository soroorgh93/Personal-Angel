"""Explicit, logged edge-vs-cloud placement decisions.

The project asks for a *measurable, defensible* decision about when work may
leave the device. Every computation in a run is classified with a reason; the
report shows the table. In `local_only` mode nothing is ever sent out. In
`hybrid` mode only tasks in `cloud_allowed_for` that carry no privacy-sensitive
payload may be marked cloud-eligible (they still are not sent by this code —
the executor for cloud tasks is a stub so the project rule is never broken).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FACTORS = ("privacy", "latency", "bandwidth", "reliability", "cost", "capability")

@dataclass
class PlacementDecision:
    task: str
    placement: str
    reasons: dict[str, str] = field(default_factory=dict)
    payload_kinds: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

class EdgeCloudPolicy:
    def __init__(self, config: dict[str, Any]) -> None:
        self.mode = str(config.get("mode", "local_only"))
        self.allowed = set(config.get("cloud_allowed_for", []))
        self.sensitive = set(config.get("privacy_sensitive_kinds", []))
        self.decisions: list[PlacementDecision] = []

    def decide(self, task: str, payload_kinds: list[str], latency_budget_ms: float | None = None,
               bytes_estimate: float | None = None) -> PlacementDecision:
        reasons: dict[str, str] = {}
        sensitive = [k for k in payload_kinds if k in self.sensitive]
        if sensitive:
            reasons["privacy"] = f"payload contains {', '.join(sensitive)}: must not leave the device"
        if latency_budget_ms is not None and latency_budget_ms < 1500:
            reasons["latency"] = f"budget {latency_budget_ms:.0f} ms < typical cloud round-trip + upload"
        if bytes_estimate and bytes_estimate > 5e6:
            reasons["bandwidth"] = f"{bytes_estimate / 1e6:.1f} MB upload not justified"
        reasons["reliability"] = "vehicle/room may be offline; decision must not depend on connectivity"
        reasons["cost"] = "per-token cloud pricing vs. amortized local compute"
        placement = "local"
        if self.mode == "hybrid" and task in self.allowed and not sensitive:
            placement = "cloud_eligible"
            reasons["capability"] = "non-sensitive enrichment may use larger cloud models"
        decision = PlacementDecision(task, placement, reasons, payload_kinds)
        self.decisions.append(decision)
        return decision

    def export(self) -> dict[str, Any]:
        return {"mode": self.mode, "cloud_allowed_for": sorted(self.allowed),
                "decisions": [d.to_dict() for d in self.decisions],
                "summary": {"local": sum(1 for d in self.decisions if d.placement == "local"),
                            "cloud_eligible": sum(1 for d in self.decisions if d.placement == "cloud_eligible")}}
