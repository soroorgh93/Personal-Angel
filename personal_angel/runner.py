"""High-level entry point used by the CLI, the server and the tests."""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from .agent.llm import create_llm
from .agent.master import investigate
from .config import load_profile, resolve_path
from .edge_cloud import EdgeCloudPolicy
from .memory.store import create_memory
from .perception.fixtures import load_fixture
from .telemetry import Telemetry

log = logging.getLogger(__name__)

def safe_run_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name)[:80] or "run"

def stream_investigation(media_path: str | Path, objective: str, profile: str | dict[str, Any] = "fixture",
                         scenario_hint: str | None = None, run_name: str | None = None,
                         answer_provider: Callable | None = None) -> Iterator[dict[str, Any]]:
    config = load_profile(profile) if isinstance(profile, str) else profile
    media_path = Path(media_path)
    run_root = resolve_path(config, config.get("output", {}).get("directory", "runs"))
    run_id = safe_run_id(run_name or f"{time.strftime('%Y%m%d-%H%M%S')}-{media_path.stem}")
    run_dir = run_root / run_id
    n = 1
    while run_dir.exists():
        run_dir = run_root / f"{run_id}-{n}"
        n += 1
    run_dir.mkdir(parents=True)
    telemetry = Telemetry(float(config.get("telemetry", {}).get("gpu_sampling_s", 0.5)))
    telemetry.start_gpu_sampling()
    fixture = load_fixture(media_path)
    llm = create_llm(config.get("llm", {}), fixture)
    memory = create_memory(config.get("memory", {}), llm, Path(config["_project_root"]))
    edge_cloud = EdgeCloudPolicy(config.get("edge_cloud", {}))
    yield {"type": "run", "run_id": run_dir.name, "run_dir": str(run_dir), "profile": config.get("project", {}).get("mode"),
           "llm": llm.name}
    try:
        yield from investigate(media_path, objective, config, run_dir, llm, memory, telemetry, scenario_hint,
                               answer_provider, edge_cloud)
    finally:
        telemetry.stop_gpu_sampling()
        memory.close()

def run_investigation(media_path: str | Path, objective: str, profile: str | dict[str, Any] = "fixture",
                      scenario_hint: str | None = None, run_name: str | None = None,
                      answer_provider: Callable | None = None, on_event: Callable[[dict[str, Any]], None] | None = None
                      ) -> dict[str, Any]:
    report: dict[str, Any] | None = None
    for event in stream_investigation(media_path, objective, profile, scenario_hint, run_name, answer_provider):
        if on_event:
            on_event(event)
        if event.get("type") == "final":
            report = event["report"]
    assert report is not None
    return report
