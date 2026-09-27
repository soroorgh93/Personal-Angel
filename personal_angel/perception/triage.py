"""Stage-1 frame triage: a tiny classifier that scores every sampled frame as
normal / person_down / weapon_visible / distress so the cascade can spend the
detector, pose and VLM budget only where the cheap model is unsure or alarmed.

Backends: `qdirp` (QDiRP-CompassNet20 trained on a GPU, ~2 GMAC/frame),
`mobilenet` (torchvision MobileNetV3-small head, pretrained backbone) and
`fixture` (returns scores derived from the fixture timeline).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)
DEFAULT_LABELS = ["normal", "person_down", "weapon_visible", "distress"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(frame_bgr: np.ndarray, size: int = 240) -> np.ndarray:
    import cv2

    h, w = frame_bgr.shape[:2]
    scale = 256 / min(h, w)
    img = cv2.resize(frame_bgr, (max(size, int(round(w * scale))), max(size, int(round(h * scale)))), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    img = img[y0:y0 + size, x0:x0 + size, ::-1].astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)[None]

class TriageModel:
    name = "abstract"
    labels = DEFAULT_LABELS

    def score(self, frame_bgr: np.ndarray, timestamp_s: float) -> dict[str, float]:
        raise NotImplementedError

class FixtureTriage(TriageModel):
    name = "fixture_triage"

    def __init__(self, fixture: dict[str, Any]) -> None:
        from .fixtures import FixtureTimeline

        self.timeline = FixtureTimeline(fixture)

    def score(self, frame_bgr, timestamp_s):
        scores = {k: 0.02 for k in self.labels}
        dets = self.timeline.detections_at(timestamp_s)
        if any(d.source == "weapon_detector" for d in dets):
            scores["weapon_visible"] = 0.8
        for p in self.timeline.poses_at(timestamp_s):
            x1, y1, x2, y2 = p.box_xyxy
            if (x2 - x1) > 1.1 * (y2 - y1):
                scores["person_down"] = 0.75
        scores["normal"] = max(0.05, 1 - max(v for k, v in scores.items() if k != "normal"))
        return scores

class TorchTriage(TriageModel):
    def __init__(self, checkpoint: Path, backend: str, labels: list[str]) -> None:
        import torch

        self.torch = torch
        self.labels = labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.name = f"{backend}_triage"
        if checkpoint.suffix == ".onnx":
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(checkpoint), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.model = None
        else:
            payload = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
            if backend == "qdirp":
                from .qdirp import QDiRPCompassNet

                model = QDiRPCompassNet(num_classes=len(labels), aux=False)
                model.load_state_dict(payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload, strict=False)
                model.fuse()
            else:
                model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            self.model = model.to(self.device).eval()
            self.session = None

    def score(self, frame_bgr, timestamp_s):
        x = preprocess(frame_bgr)
        if self.session is not None:
            out = self.session.run(None, {self.session.get_inputs()[0].name: x.astype(np.float32)})[0].reshape(-1)
        else:
            with self.torch.no_grad():
                out = self.model(self.torch.from_numpy(x).to(self.device)).reshape(-1).float().cpu().numpy()
        e = np.exp(out - out.max())
        probs = e / e.sum()
        return {label: float(p) for label, p in zip(self.labels, probs)}

def create_triage(config: dict[str, Any], project_root: Path, fixture: dict[str, Any] | None = None) -> TriageModel | None:
    if not bool(config.get("enabled", False)):
        return None
    backend = str(config.get("backend", "fixture"))
    labels = list(config.get("labels", DEFAULT_LABELS))
    if backend == "fixture":
        return FixtureTriage(fixture or {})
    checkpoint = project_root / str(config.get("checkpoint", "models/triage_qdirp.pt"))
    if not checkpoint.exists():
        log.warning("Triage checkpoint %s not found; stage-1 triage disabled", checkpoint)
        return None
    return TorchTriage(checkpoint, backend, labels)
