"""Process-wide model registry: heavy perception models (YOLO, pose, CLIP, whisper, CLAP, Detoxify) are loaded
once and stay resident in (unified) memory for the lifetime of the server, instead of being re-created — and
re-read from disk — for every investigation. This is what makes the second demo clip start in under a second on
a GPU workstation. Fixture backends are never cached (they depend on the fixture file)."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MODELS: dict[str, Any] = {}
_KEY_LOCKS: dict[str, threading.Lock] = {}
_LOADED_AT: dict[str, float] = {}

def _key(kind: str, config: dict[str, Any], extra: str = "") -> str:
    try:
        canon = json.dumps({k: v for k, v in config.items() if not str(k).startswith("_")}, sort_keys=True, default=str)
    except Exception:
        canon = repr(config)
    return f"{kind}:{extra}:{canon}"

def cached(kind: str, config: dict[str, Any], factory: Callable[[], Any], extra: str = "") -> Any:
    """Return the resident instance for (kind, config), creating it with `factory` on first use."""
    if str(config.get("backend", "fixture")) in {"fixture", "none", ""}:
        return factory()
    key = _key(kind, config, extra)
    with _LOCK:
        inst = _MODELS.get(key)
        lock = _KEY_LOCKS.setdefault(key, threading.Lock())
    if inst is not None:
        return inst
    with lock:
        with _LOCK:
            inst = _MODELS.get(key)
        if inst is not None:
            return inst
        t0 = time.perf_counter()
        inst = factory()
        with _LOCK:
            _MODELS[key] = inst
            _LOADED_AT[key] = time.perf_counter() - t0
        log.info("model resident: %s (%s) loaded in %.1fs", kind, config.get("backend"), time.perf_counter() - t0)
        return inst

def status() -> dict[str, Any]:
    with _LOCK:
        return {"resident": sorted(k.split(":", 1)[0] for k in _MODELS), "count": len(_MODELS),
                "load_seconds": {k.split(":", 1)[0]: round(v, 1) for k, v in _LOADED_AT.items()}}

def warm_up(config: dict[str, Any], project_root) -> dict[str, Any]:
    """Load every perception model of a profile ahead of the first run (called in a background thread at server
    start). Failures are logged and reported, never raised: the app must still come up."""
    from pathlib import Path

    from .audio import create_audio_analyzer
    from .detector import create_detector
    from .pose import create_pose_estimator
    from .scene import create_scene_analyzer

    root = Path(project_root)
    report: dict[str, Any] = {"ok": True, "errors": {}}
    t0 = time.perf_counter()
    steps: list[tuple[str, Callable[[], Any]]] = [
        ("scene", lambda: create_scene_analyzer(config.get("scene", {}), root, None, None)),
        ("detector", lambda: create_detector(config.get("detector", {}), root, None)),
        ("pose", lambda: create_pose_estimator({**config.get("pose", {}), "_project_root": str(root)}, None)),
        ("audio", lambda: create_audio_analyzer(config.get("audio", {}), None)),
    ]
    for name, fn in steps:
        try:
            inst = fn()
            if name == "audio" and hasattr(inst, "warm_up"):
                inst.warm_up()
            elif name in {"detector", "pose"} and hasattr(inst, "warm_up"):
                inst.warm_up()
        except Exception as error:
            log.warning("warm-up of %s failed: %s", name, error)
            report["errors"][name] = f"{type(error).__name__}: {error}"
            report["ok"] = False
    report["seconds"] = round(time.perf_counter() - t0, 1)
    report.update(status())
    return report
