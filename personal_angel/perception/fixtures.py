"""Fixture support: replay annotations from `<media>.fixture.json`.

A fixture describes tracks with keyframes; boxes are linearly interpolated and
17 COCO keypoints are synthesized from a named posture so the fall/slump state
machines run on real geometry. Audio and LLM scripts can also be provided.

Example:
{
  "location": "CAR_CABIN",
  "tracks": [
    {"track_id": 1, "label": "person",
     "keyframes": [
       {"t": 0.0, "box": [200, 80, 420, 470], "posture": "seated"},
       {"t": 6.0, "box": [200, 80, 420, 470], "posture": "seated"},
       {"t": 8.0, "box": [180, 160, 430, 480], "posture": "slumped"},
       {"t": 20.0, "box": [180, 160, 430, 480], "posture": "slumped"}]},
    {"track_id": 7, "label": "knife", "source": "weapon_detector", "holder": 1,
     "keyframes": [{"t": 3.0, "box": [300, 300, 340, 360]}, {"t": 9.0, "box": [300, 300, 340, 360]}]}
  ],
  "audio": {"segments": [...], "acoustic": [...], "risk": {...}},
  "llm_script": [...]
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Detection, PoseObservation

POSTURES = {

    "upright": [(0.5, 0.06), (0.47, 0.05), (0.53, 0.05), (0.44, 0.06), (0.56, 0.06),
                (0.35, 0.2), (0.65, 0.2), (0.3, 0.38), (0.7, 0.38), (0.28, 0.52), (0.72, 0.52),
                (0.4, 0.55), (0.6, 0.55), (0.4, 0.75), (0.6, 0.75), (0.4, 0.97), (0.6, 0.97)],
    "seated": [(0.5, 0.08), (0.47, 0.07), (0.53, 0.07), (0.44, 0.08), (0.56, 0.08),
               (0.36, 0.25), (0.64, 0.25), (0.3, 0.45), (0.7, 0.45), (0.32, 0.6), (0.68, 0.6),
               (0.4, 0.62), (0.6, 0.62), (0.3, 0.8), (0.7, 0.8), (0.3, 0.97), (0.7, 0.97)],
    "slumped": [(0.72, 0.42), (0.7, 0.4), (0.74, 0.4), (0.67, 0.42), (0.77, 0.42),
                (0.4, 0.38), (0.66, 0.48), (0.3, 0.55), (0.62, 0.66), (0.28, 0.7), (0.6, 0.78),
                (0.4, 0.66), (0.6, 0.66), (0.3, 0.82), (0.7, 0.82), (0.3, 0.97), (0.7, 0.97)],
    "lying": [(0.06, 0.5), (0.05, 0.47), (0.05, 0.53), (0.06, 0.44), (0.06, 0.56),
              (0.2, 0.35), (0.2, 0.65), (0.38, 0.3), (0.38, 0.7), (0.52, 0.28), (0.52, 0.72),
              (0.55, 0.4), (0.55, 0.6), (0.75, 0.4), (0.75, 0.6), (0.97, 0.4), (0.97, 0.6)],
    "crouched": [(0.5, 0.2), (0.47, 0.19), (0.53, 0.19), (0.44, 0.2), (0.56, 0.2),
                 (0.35, 0.4), (0.65, 0.4), (0.3, 0.6), (0.7, 0.6), (0.3, 0.75), (0.7, 0.75),
                 (0.4, 0.7), (0.6, 0.7), (0.35, 0.85), (0.65, 0.85), (0.35, 0.97), (0.65, 0.97)],
    "striking": [(0.5, 0.06), (0.47, 0.05), (0.53, 0.05), (0.44, 0.06), (0.56, 0.06),
                 (0.35, 0.2), (0.65, 0.2), (0.2, 0.25), (0.75, 0.3), (0.05, 0.22), (0.9, 0.2),
                 (0.4, 0.55), (0.6, 0.55), (0.4, 0.75), (0.6, 0.75), (0.4, 0.97), (0.6, 0.97)],
    "infant": [(0.5, 0.15), (0.46, 0.13), (0.54, 0.13), (0.42, 0.16), (0.58, 0.16),
               (0.35, 0.35), (0.65, 0.35), (0.25, 0.5), (0.75, 0.5), (0.2, 0.62), (0.8, 0.62),
               (0.4, 0.65), (0.6, 0.65), (0.35, 0.82), (0.65, 0.82), (0.3, 0.95), (0.7, 0.95)],
}

def fixture_path_for(media_path: str | Path) -> Path:
    media_path = Path(media_path)
    return media_path.with_suffix(media_path.suffix + ".fixture.json")

def load_fixture(media_path: str | Path) -> dict[str, Any]:
    path = fixture_path_for(media_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)

def _interp_box(keyframes: list[dict[str, Any]], t: float) -> tuple[list[float], str] | None:
    if not keyframes:
        return None
    if t < keyframes[0]["t"] - 1e-6 or t > keyframes[-1]["t"] + 1e-6:
        return None
    for a, b in zip(keyframes, keyframes[1:]):
        if a["t"] <= t <= b["t"]:
            span = max(b["t"] - a["t"], 1e-6)
            w = (t - a["t"]) / span
            box = [float(a["box"][i] * (1 - w) + b["box"][i] * w) for i in range(4)]
            posture = a.get("posture", "upright") if w < 0.5 else b.get("posture", a.get("posture", "upright"))
            return box, posture
    kf = keyframes[-1]
    return [float(v) for v in kf["box"]], kf.get("posture", "upright")

def synth_keypoints(box: list[float], posture: str, jitter: float = 0.0, seed: int = 0
                    ) -> list[tuple[float, float, float]]:
    layout = POSTURES.get(posture, POSTURES["upright"])
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rng = np.random.default_rng(seed)
    points = []
    for fx, fy in layout:
        jx = rng.normal(0, jitter) * w if jitter else 0.0
        jy = rng.normal(0, jitter) * h if jitter else 0.0
        points.append((x1 + fx * w + jx, y1 + fy * h + jy, 0.9))
    return points

class FixtureTimeline:
    def __init__(self, fixture: dict[str, Any]) -> None:
        self.fixture = fixture
        self.tracks = fixture.get("tracks", [])

    def detections_at(self, t: float) -> list[Detection]:
        result: list[Detection] = []
        for track in self.tracks:
            interp = _interp_box(track.get("keyframes", []), t)
            if interp is None:
                continue
            box, _ = interp
            is_person = str(track.get("label", "person")) == "person"
            track_id = int(track["track_id"]) if "track_id" in track else None
            if not is_person:
                track_id = int(track["holder"]) if track.get("holder") is not None else None
            result.append(Detection(
                label=str(track.get("label", "person")),
                confidence=float(track.get("confidence", 0.88)),
                box_xyxy=tuple(box),
                track_id=track_id,
                source=str(track.get("source", "detector")),
            ))
        return result

    def poses_at(self, t: float, frame_index: int = 0) -> list[PoseObservation]:
        result: list[PoseObservation] = []
        for track in self.tracks:
            if track.get("label", "person") != "person":
                continue
            interp = _interp_box(track.get("keyframes", []), t)
            if interp is None:
                continue
            box, posture = interp
            jitter = float(track.get("jitter", 0.004))
            result.append(PoseObservation(
                track_id=int(track["track_id"]),
                keypoints=synth_keypoints(box, posture, jitter, seed=frame_index * 31 + int(track["track_id"])),
                box_xyxy=tuple(box),
            ))
        return result
