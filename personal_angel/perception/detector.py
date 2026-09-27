"""Object + weapon + fallen-person detection with identity tracking.

Real backend: Ultralytics YOLO11n (COCO, ByteTrack ids) fused with a
gun/knife YOLO11n (cosgun99/gun-knife-yolo11n, MIT) and an optional
Fallen/Sitting/Standing YOLO11 (melihuzunoglu/human-fall-detection).
Weapon boxes are attached to the person whose box contains/overlaps them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Detection
from .fixtures import FixtureTimeline

log = logging.getLogger(__name__)

WEAPON_LABELS = {"gun", "pistol", "handgun", "rifle", "knife", "firearm", "weapon"}
COCO_TOOL_LABELS = {"knife", "scissors", "baseball bat"}

def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)

def containment(inner: tuple[float, float, float, float],
                outer: tuple[float, float, float, float]) -> float:
    """Fraction of `inner` area inside `outer`."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, inner[2] - inner[0]) * max(0.0, inner[3] - inner[1])
    return inter / max(area, 1e-6)

class Detector:
    name = "abstract"

    def detect(self, frame: np.ndarray, timestamp_s: float, frame_index: int) -> list[Detection]:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset tracker state between videos or windows."""

class FixtureDetector(Detector):
    name = "fixture_detector"

    def __init__(self, fixture: dict[str, Any]) -> None:
        self.timeline = FixtureTimeline(fixture)

    def detect(self, frame: np.ndarray, timestamp_s: float, frame_index: int) -> list[Detection]:
        return attach_objects_to_people(self.timeline.detections_at(timestamp_s))

class UltralyticsDetector(Detector):
    name = "yolo11_fused"

    def __init__(self, config: dict[str, Any], project_root: Path) -> None:
        from ultralytics import YOLO

        self.config = config
        self.device = config.get("device", "auto")
        if self.device == "auto":
            self.device = _auto_device()
        self.conf = float(config.get("conf", 0.35))
        self.weapon_conf = float(config.get("weapon_conf", 0.45))
        self.imgsz = int(config.get("imgsz", 640))
        self.tracker = config.get("tracker", "bytetrack.yaml")
        base = str(config.get("model", "yolo11n.pt"))
        local = project_root / "models" / Path(base).name
        self.model = YOLO(str(local if local.exists() else base))
        self.weapon_model = None
        self.fallen_model = None
        weapon_path = project_root / str(config.get("weapon_model", ""))
        fallen_path = project_root / str(config.get("fallen_model", ""))
        if config.get("weapon_model") and weapon_path.exists():
            self.weapon_model = YOLO(str(weapon_path))
            self.name = "yolo11_fused+weapon"
        else:
            log.warning("Weapon detector weights not found at %s; COCO knife/scissors/bat only", weapon_path)
        if config.get("fallen_model") and fallen_path.exists():
            self.fallen_model = YOLO(str(fallen_path))
        self.classes_of_interest = {str(c) for c in config.get("classes_of_interest", [])}

    def reset(self) -> None:

        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor = None

    def warm_up(self) -> None:
        """One dummy inference so CUDA kernels / TensorRT engines are initialised before the first real frame."""
        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.detect(blank, 0.0, 0)
        finally:
            self.reset()

    def detect(self, frame: np.ndarray, timestamp_s: float, frame_index: int) -> list[Detection]:
        results = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz,
                                   device=self.device, tracker=self.tracker, verbose=False)
        detections: list[Detection] = []
        for r in results:
            names = r.names
            boxes = r.boxes
            if boxes is None:
                continue
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)
            for i in range(len(boxes)):
                label = names[int(boxes.cls[i])]
                if self.classes_of_interest and label not in self.classes_of_interest:
                    continue
                source = "detector"
                if label in COCO_TOOL_LABELS:
                    source = "weapon_detector"
                detections.append(Detection(
                    label=label, confidence=float(boxes.conf[i]),
                    box_xyxy=tuple(float(v) for v in boxes.xyxy[i].tolist()),
                    track_id=int(ids[i]) if ids[i] is not None else None, source=source,
                ))
        if self.weapon_model is not None:
            for r in self.weapon_model.predict(frame, conf=self.weapon_conf, imgsz=self.imgsz,
                                               device=self.device, verbose=False):
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    label = str(r.names[int(r.boxes.cls[i])]).lower()
                    detections.append(Detection(
                        label=label, confidence=float(r.boxes.conf[i]),
                        box_xyxy=tuple(float(v) for v in r.boxes.xyxy[i].tolist()),
                        source="weapon_detector",
                    ))
        if self.fallen_model is not None:
            for r in self.fallen_model.predict(frame, conf=0.5, imgsz=self.imgsz,
                                               device=self.device, verbose=False):
                if r.boxes is None:
                    continue
                for i in range(len(r.boxes)):
                    label = str(r.names[int(r.boxes.cls[i])])
                    if label.lower() not in {"fallen", "fall", "person_down"}:
                        continue
                    detections.append(Detection(
                        label="fallen", confidence=float(r.boxes.conf[i]),
                        box_xyxy=tuple(float(v) for v in r.boxes.xyxy[i].tolist()),
                        source="fallen_detector",
                    ))
        return attach_objects_to_people(detections)

def attach_objects_to_people(detections: list[Detection]) -> list[Detection]:
    """Give weapon/fallen boxes the track id of the person they belong to."""
    people = [d for d in detections if d.label == "person" and d.track_id is not None]
    for det in detections:
        if det.track_id is not None or not people:
            continue
        if det.source in {"weapon_detector", "fallen_detector"} or det.label in COCO_TOOL_LABELS:
            best, best_score = None, 0.0
            for person in people:
                score = containment(det.box_xyxy, person.box_xyxy)
                if det.source == "fallen_detector":
                    score = iou(det.box_xyxy, person.box_xyxy)
                if score > best_score:
                    best, best_score = person, score
            if best is not None and best_score > 0.2:
                det.track_id = best.track_id
    return detections

def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def create_detector(config: dict[str, Any], project_root: Path, fixture: dict[str, Any] | None = None
                    ) -> Detector:
    backend = str(config.get("backend", "fixture"))
    if backend == "fixture":
        return FixtureDetector(fixture or {})
    if backend == "ultralytics":
        from .registry import cached

        det = cached("detector", config, lambda: UltralyticsDetector(config, project_root), str(project_root))
        det.reset()
        return det
    raise ValueError(f"Unknown detector backend: {backend}")
