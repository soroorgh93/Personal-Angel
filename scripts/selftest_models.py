"""Self-test of every real backend on this machine — run it after setup or when a card in the UI looks empty.

    python scripts/selftest_models.py --profile pc_cpu

Each line prints OK / FAIL with the exception, so a broken install (e.g. a transformers API change)
is visible immediately instead of silently producing 'unknown' scenes. Nothing here needs the network.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def check(name: str, fn):
    t0 = time.perf_counter()
    try:
        out = fn()
        print(f"  OK   {name:<28} {(time.perf_counter() - t0) * 1000:7.0f} ms  {str(out)[:110]}")
        return True
    except Exception as error:
        print(f"  FAIL {name:<28} {type(error).__name__}: {error}")
        traceback.print_exc(limit=3)
        return False

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="pc_cpu")
    args = parser.parse_args()
    from personal_angel.config import load_profile

    cfg = load_profile(args.profile)
    root = Path(cfg["_project_root"])
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    import cv2

    cv2.rectangle(frame, (250, 120), (390, 460), (200, 180, 160), -1)
    ok = True
    print(f"PersonalAngel self-test (profile {args.profile}, python {sys.version.split()[0]})")
    try:
        import torch, transformers

        print(f"  torch {torch.__version__} cuda={torch.cuda.is_available()} · transformers {transformers.__version__}")
    except Exception as error:
        print(f"  (torch/transformers import problem: {error})")

    def scene():
        from personal_angel.perception.scene import create_scene_analyzer

        an = create_scene_analyzer(cfg.get("scene", {}), root, None, None)
        if an is None:
            raise RuntimeError("scene analyzer could not be created (backend %s)" % cfg.get("scene", {}).get("backend"))
        r = an.selftest()
        loc = an.scene([frame])
        return f"{r} scene={loc[0]} {loc[2]:.2f}"

    ok &= check("scene model (CLIP)", scene)

    def detector():
        from personal_angel.perception.detector import create_detector

        det = create_detector(cfg.get("detector", {}), root, None)
        return f"{det.name}: {len(det.detect(frame, 0.0, 0))} detections on a blank frame"

    ok &= check("detector (YOLO11 + weapon)", detector)

    def pose():
        from personal_angel.perception.pose import create_pose_estimator

        est = create_pose_estimator({**cfg.get("pose", {}), "_project_root": str(root)}, None)
        return f"{est.name}: {len(est.estimate(frame, 0.0, 0, [(1, (250.0, 120.0, 390.0, 460.0))]))} poses"

    ok &= check("pose (YOLO11-pose)", pose)

    def audio():
        import wave

        from personal_angel.perception.audio import create_audio_analyzer

        wav = root / "runs" / "_selftest.wav"
        wav.parent.mkdir(exist_ok=True)
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        pcm = (0.05 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
        an = create_audio_analyzer(cfg.get("audio", {}), None)
        segs, acoustic, risk = an.analyze(wav)
        return f"{an.name}: {len(segs)} segments, {len(acoustic)} acoustic events, risk model {risk.model}"

    ok &= check("audio (whisper + detoxify + CLAP)", audio)

    def llm():
        from personal_angel.agent.llm import create_llm

        c = create_llm(cfg.get("llm", {}))
        h = c.health()
        if not h.get("ok"):
            raise RuntimeError(f"model endpoint not reachable: {h}")
        r = c.chat([{"role": "user", "content": 'Reply with JSON {"ok": true}'}], json_mode=True, max_tokens=20)
        return f"{c.name}: {r.content[:40]!r} ({r.latency_ms:.0f} ms)"

    ok &= check("language/vision model", llm)
    print("\nALL OK" if ok else "\nSome checks FAILED — send this output (runs/desktop.log has the same tracebacks).")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
