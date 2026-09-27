"""Pose estimation + human-state analysis (fall, on-ground, seated slump,
striking motion).

The analysis is a transparent state machine over 17 COCO keypoints so every
"possible fall" carries the numbers that triggered it (torso angle, drop
velocity, dwell time). A learned temporal head (TCN/GRU) can be plugged in via
`TemporalFallHead`; the rules remain the explainable floor.

Rule thresholds follow PIFR (PLOS One 2025), Ye 2024, and arXiv 2607.12909.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import PoseObservation
from .fixtures import FixtureTimeline

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

class PoseEstimator:
    name = "abstract"

    def estimate(self, frame: np.ndarray, timestamp_s: float, frame_index: int,
                 people: list[tuple[int | None, tuple[float, float, float, float]]]
                 ) -> list[PoseObservation]:
        raise NotImplementedError

class FixturePoseEstimator(PoseEstimator):
    name = "fixture_pose"

    def __init__(self, fixture: dict[str, Any]) -> None:
        self.timeline = FixtureTimeline(fixture)

    def estimate(self, frame, timestamp_s, frame_index, people):
        return self.timeline.poses_at(timestamp_s, frame_index)

class UltralyticsPoseEstimator(PoseEstimator):
    name = "yolo11n_pose"

    def __init__(self, config: dict[str, Any]) -> None:
        from ultralytics import YOLO

        from .detector import _auto_device

        from pathlib import Path as _Path

        base = str(config.get("model", "yolo11n-pose.pt"))
        local = _Path(config.get("_project_root", ".")) / "models" / _Path(base).name
        self.model = YOLO(str(local if local.exists() else base))
        self.device = config.get("device", "auto")
        if self.device == "auto":
            self.device = _auto_device()
        self.imgsz = int(config.get("imgsz", 640))
        self.name = f"{_Path(base).stem.replace('-', '_')}"

    def warm_up(self) -> None:
        import numpy as _np

        self.estimate(_np.zeros((self.imgsz, self.imgsz, 3), dtype=_np.uint8), 0.0, 0, [])

    def estimate(self, frame, timestamp_s, frame_index, people):
        from .detector import iou

        results = self.model.predict(frame, conf=0.3, imgsz=self.imgsz, device=self.device,
                                     verbose=False)
        observations: list[PoseObservation] = []
        for r in results:
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue
            kp = r.keypoints.data.cpu().numpy()
            if kp.ndim != 3 or kp.shape[0] != len(r.boxes):
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            for i in range(kp.shape[0]):
                box = tuple(float(v) for v in boxes[i])

                best_id, best = None, 0.0
                for track_id, pbox in people:
                    score = iou(box, pbox)
                    if score > best:
                        best, best_id = score, track_id
                if best_id is None:
                    best_id = -(i + 1)
                observations.append(PoseObservation(
                    track_id=int(best_id),
                    keypoints=[(float(x), float(y), float(c)) for x, y, c in kp[i]],
                    box_xyxy=box,
                ))
        return observations

def create_pose_estimator(config: dict[str, Any], fixture: dict[str, Any] | None = None
                          ) -> PoseEstimator:
    backend = str(config.get("backend", "fixture"))
    if backend == "fixture":
        return FixturePoseEstimator(fixture or {})
    if backend == "ultralytics":
        from .registry import cached

        return cached("pose", config, lambda: UltralyticsPoseEstimator(config))
    raise ValueError(f"Unknown pose backend: {backend}")

@dataclass
class PoseFeatures:
    t: float
    torso_angle_deg: float
    aspect_ratio: float
    hip_y: float
    head_y: float
    body_scale: float
    center_x: float
    center_y: float
    head_drop_ratio: float
    lateral_lean_ratio: float
    shoulder_asym_ratio: float
    wrist_speed: float = 0.0
    valid: bool = True

def _pt(kp: list[tuple[float, float, float]], idx: int, min_conf: float = 0.5
        ) -> tuple[float, float] | None:
    x, y, c = kp[idx]
    if c < min_conf or (x == 0.0 and y == 0.0):
        return None
    return (x, y)

def _mid(a: tuple[float, float] | None, b: tuple[float, float] | None) -> tuple[float, float] | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

def compute_features(obs: PoseObservation, t: float) -> PoseFeatures:
    kp = obs.keypoints
    x1, y1, x2, y2 = obs.box_xyxy
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    sh = _mid(_pt(kp, L_SH), _pt(kp, R_SH))
    hip = _mid(_pt(kp, L_HIP), _pt(kp, R_HIP))
    nose = _pt(kp, NOSE) or _mid(_pt(kp, L_EAR), _pt(kp, R_EAR))
    lsh, rsh = _pt(kp, L_SH), _pt(kp, R_SH)
    shoulder_width = abs(lsh[0] - rsh[0]) if lsh and rsh else w * 0.35
    shoulder_width = max(shoulder_width, 1.0)
    body_scale = max(h, 1.0)
    if sh is not None and hip is not None:
        dx, dy = sh[0] - hip[0], sh[1] - hip[1]
        torso_len = math.hypot(dx, dy)
        body_scale = max(torso_len * 2.6, h * 0.5, 1.0)

        angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
        valid = True
    else:
        angle = 90.0 if w > h else 0.0
        valid = False
    head_y = nose[1] if nose else y1
    hip_y = hip[1] if hip else (y1 + y2) / 2
    head_drop = ((nose[1] - sh[1]) / shoulder_width) if (nose and sh) else 0.0
    lateral = (abs(sh[0] - hip[0]) / shoulder_width) if (sh and hip) else 0.0
    asym = (abs(lsh[1] - rsh[1]) / shoulder_width) if (lsh and rsh) else 0.0
    return PoseFeatures(
        t=t, torso_angle_deg=angle, aspect_ratio=w / h, hip_y=hip_y, head_y=head_y,
        body_scale=body_scale, center_x=(x1 + x2) / 2, center_y=(y1 + y2) / 2,
        head_drop_ratio=head_drop, lateral_lean_ratio=lateral, shoulder_asym_ratio=asym,
        valid=valid,
    )

@dataclass
class TrackSequence:
    track_id: int
    features: list[PoseFeatures] = field(default_factory=list)
    wrists: list[tuple[float, tuple[float, float] | None, tuple[float, float] | None]] = field(default_factory=list)
    keypoints_norm: list[np.ndarray] = field(default_factory=list)

    def windows(self, T: int = 32, stride: int = 8) -> list[np.ndarray]:
        """[T,17,5] = (x, y, conf, vx, vy) windows for the learned fall head (see train_fall_tcn.py)."""
        if len(self.keypoints_norm) < T:
            return []
        seq = np.stack(self.keypoints_norm)
        out = []
        for start in range(0, len(seq) - T + 1, stride):
            win = seq[start:start + T]
            vel = np.diff(win[:, :, :2], axis=0, prepend=win[:1, :, :2])
            out.append(np.concatenate([win, vel], axis=2).astype(np.float32))
        return out

@dataclass
class HumanStateEvent:
    kind: str
    track_id: int
    start_s: float
    end_s: float
    confidence: float
    features: dict[str, Any]
    explanation: str

class HumanStateAnalyzer:
    """Consumes pose observations frame by frame, emits HumanStateEvents."""

    def __init__(self, config: dict[str, Any], seated_context: bool = False) -> None:
        fall = config.get("fall", {})
        slump = config.get("slump", {})
        self.lying_angle = float(fall.get("torso_angle_lying_deg", 60))
        self.lying_ar = float(fall.get("aspect_ratio_lying", 1.1))
        self.drop_velocity = float(fall.get("drop_velocity_h_per_s", 0.6))
        self.dwell_s = float(fall.get("on_ground_dwell_s", 2.0))
        self.recover_s = float(fall.get("stumble_recover_s", 2.0))
        self.baseline_s = float(slump.get("baseline_s", 3.0))
        self.head_drop_ratio = float(slump.get("head_drop_ratio", 0.15))
        self.lateral_ratio = float(slump.get("lateral_lean_ratio", 0.45))
        self.stillness_s = float(slump.get("stillness_s", 6.0))
        self.stillness_disp = float(slump.get("stillness_disp_ratio", 0.012))
        self.seated_context = seated_context
        self.tracks: dict[int, TrackSequence] = {}

    def observe(self, poses: list[PoseObservation], t: float) -> None:
        for obs in poses:
            seq = self.tracks.setdefault(obs.track_id, TrackSequence(obs.track_id))
            seq.features.append(compute_features(obs, t))
            seq.wrists.append((t, _pt(obs.keypoints, L_WR), _pt(obs.keypoints, R_WR)))
            x1, y1, x2, y2 = obs.box_xyxy
            w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            kp = np.asarray(obs.keypoints, dtype=np.float32).reshape(17, 3).copy()
            kp[:, 0] = (kp[:, 0] - x1) / w
            kp[:, 1] = (kp[:, 1] - y1) / h
            seq.keypoints_norm.append(kp)

    def analyze(self, fall_head: "TemporalFallHead | None" = None) -> list[HumanStateEvent]:
        events: list[HumanStateEvent] = []
        for seq in self.tracks.values():
            if len(seq.features) < 3:
                continue
            track_events = self._analyze_fall(seq)
            if fall_head is not None:
                track_events = self._fuse_learned(seq, track_events, fall_head)
            events.extend(track_events)
            if self.seated_context:
                events.extend(self._analyze_slump(seq))
            events.extend(self._analyze_striking(seq))
        return events

    @staticmethod
    def _fuse_learned(seq: TrackSequence, events: list[HumanStateEvent], head: "TemporalFallHead") -> list[HumanStateEvent]:
        """Rule ⊕ learned head: the TCN score raises/lowers rule confidence and can add a lower-confidence
        fall candidate the rules missed (alert if rule=CONFIRMED or TCN > 0.7)."""
        windows = seq.windows()
        if not windows:
            return events
        scores = [head.score(w) for w in windows]
        peak = max(scores)
        peak_t = seq.features[min(len(seq.features) - 1, int(scores.index(peak) * 8 + 16))].t
        for ev in events:
            if ev.kind in {"fall", "person_down"}:
                ev.features["tcn_fall_probability"] = round(peak, 3)
                ev.confidence = round(min(0.98, 0.6 * ev.confidence + 0.4 * peak), 3)
                ev.explanation += f" Learned fall head: p={peak:.2f}."
        if peak > 0.7 and not any(e.kind == "fall" for e in events):
            events.append(HumanStateEvent("fall", seq.track_id, max(0.0, peak_t - 1.5), seq.features[-1].t, round(0.4 + 0.4 * peak, 3),
                                          {"tcn_fall_probability": round(peak, 3), "rule_state": "no rule trigger"},
                                          f"PERSON_{seq.track_id:02d}: learned fall head fired (p={peak:.2f}) around {peak_t:.1f}s "
                                          "without a rule-level upright→horizontal transition (lower confidence)."))
        return events

    def _lying(self, f: PoseFeatures) -> bool:
        """Horizontal torso is the primary cue; a wide box alone (seated passenger, cropped upper body)
        must not count as lying — it only corroborates a torso that is already well off vertical."""
        if f.torso_angle_deg >= self.lying_angle:
            return True
        return f.aspect_ratio >= self.lying_ar and (not f.valid or f.torso_angle_deg >= 0.65 * self.lying_angle)

    def _analyze_fall(self, seq: TrackSequence) -> list[HumanStateEvent]:
        feats = seq.features
        events: list[HumanStateEvent] = []

        vel = [0.0]
        for a, b in zip(feats, feats[1:]):
            dt = max(b.t - a.t, 1e-3)
            scale = max((a.body_scale + b.body_scale) / 2, 1.0)
            vel.append((b.hip_y - a.hip_y) / scale / dt)
        ema = []
        acc = 0.0
        for v in vel:
            acc = 0.6 * acc + 0.4 * v
            ema.append(acc)

        state = "UPRIGHT" if not self._lying(feats[0]) else "ON_GROUND_INITIAL"
        onset_t: float | None = None
        ground_t: float | None = None
        peak_vel = 0.0
        angle_before = feats[0].torso_angle_deg
        for i, f in enumerate(feats):
            lying = self._lying(f)
            if state in {"UPRIGHT", "RECOVERED"}:
                if ema[i] > self.drop_velocity or (lying and i > 0 and not self._lying(feats[i - 1])):
                    state = "FALLING"
                    onset_t = feats[max(0, i - 1)].t
                    peak_vel = ema[i]
                    angle_before = feats[max(0, i - 2)].torso_angle_deg
            elif state == "FALLING":
                peak_vel = max(peak_vel, ema[i])
                if lying:
                    state = "ON_GROUND"
                    ground_t = f.t
                elif onset_t is not None and f.t - onset_t > 1.5 and not lying:
                    state = "UPRIGHT"
                    onset_t = None
            elif state == "ON_GROUND":
                assert ground_t is not None and onset_t is not None
                if not lying:
                    if f.t - ground_t < self.recover_s:
                        events.append(HumanStateEvent(
                            "recovered", seq.track_id, onset_t, f.t, 0.5,
                            {"peak_drop_velocity": round(peak_vel, 3)},
                            f"PERSON_{seq.track_id:02d} dropped quickly but stood up within "
                            f"{f.t - ground_t:.1f}s (stumble, not a fall)."))
                    state = "RECOVERED"
                    onset_t, ground_t = None, None
                elif f.t - ground_t >= self.dwell_s:
                    state = "CONFIRMED"
                    end_t = feats[-1].t
                    dwell = end_t - ground_t
                    feats_out = {"peak_drop_velocity_h_per_s": round(peak_vel, 3),
                                 "torso_angle_before_deg": round(angle_before, 1),
                                 "torso_angle_after_deg": round(f.torso_angle_deg, 1),
                                 "on_ground_dwell_s": round(dwell, 2),
                                 "aspect_ratio_after": round(f.aspect_ratio, 2)}
                    slow = peak_vel < 0.5 * self.drop_velocity
                    never_upright = angle_before >= 40.0
                    if slow and never_upright:

                        events.append(HumanStateEvent(
                            "person_down", seq.track_id, onset_t, end_t, 0.45, {**feats_out, "fall_observed": False},
                            f"PERSON_{seq.track_id:02d} was already low (torso {angle_before:.0f}°) and drifted to "
                            f"horizontal at {onset_t:.1f}s with no drop ({peak_vel:.2f} body-heights/s); on the ground "
                            f"for {dwell:.1f}s. No fall was observed — crawling, playing or lying down deliberately."))
                    else:
                        conf = min(0.97, 0.55 + 0.25 * min(peak_vel / max(self.drop_velocity, 1e-3), 1.5)
                                   + 0.1 * min(dwell / 5.0, 1.0))
                        if slow:
                            conf = min(conf, 0.5)
                        events.append(HumanStateEvent(
                            "fall", seq.track_id, onset_t, end_t, conf, {**feats_out, "fall_observed": not slow},
                            f"PERSON_{seq.track_id:02d}: upright→horizontal transition at {onset_t:.1f}s "
                            f"(torso {angle_before:.0f}°→{f.torso_angle_deg:.0f}°, drop {peak_vel:.2f} body-heights/s), "
                            f"then on the ground for {dwell:.1f}s." + (" Slow descent: could be intentional lying down." if slow else "")))
            elif state == "CONFIRMED":
                if not lying:
                    events[-1].end_s = f.t
                    events[-1].features["recovered_at_s"] = round(f.t, 2)
                    state = "RECOVERED"
        if state == "ON_GROUND" and onset_t is not None and ground_t is not None:

            dwell = feats[-1].t - ground_t
            conf = min(0.8, 0.45 + 0.2 * min(peak_vel / max(self.drop_velocity, 1e-3), 1.5) + 0.1 * min(dwell / self.dwell_s, 1.0))
            events.append(HumanStateEvent(
                "fall", seq.track_id, onset_t, feats[-1].t, round(conf, 3),
                {"peak_drop_velocity_h_per_s": round(peak_vel, 3), "torso_angle_before_deg": round(angle_before, 1),
                 "torso_angle_after_deg": round(feats[-1].torso_angle_deg, 1), "on_ground_dwell_s": round(dwell, 2),
                 "clip_ended_on_ground": True, "fall_observed": peak_vel >= 0.5 * self.drop_velocity},
                f"PERSON_{seq.track_id:02d}: upright→horizontal transition at {onset_t:.1f}s (torso {angle_before:.0f}°→"
                f"{feats[-1].torso_angle_deg:.0f}°, drop {peak_vel:.2f} body-heights/s); the recording ends {dwell:.1f}s later "
                f"with the person still on the ground — no recovery was seen."))
        elif state == "FALLING" and onset_t is not None:
            events.append(HumanStateEvent(
                "fall", seq.track_id, onset_t, feats[-1].t, 0.45,
                {"peak_drop_velocity_h_per_s": round(peak_vel, 3), "torso_angle_before_deg": round(angle_before, 1),
                 "clip_ended_while_falling": True, "fall_observed": True},
                f"PERSON_{seq.track_id:02d}: a fast downward motion started at {onset_t:.1f}s (drop {peak_vel:.2f} body-heights/s) "
                f"and the recording ends before the outcome is visible."))
        if state == "ON_GROUND_INITIAL":
            lying_span = [f for f in feats if self._lying(f)]
            if len(lying_span) >= max(3, int(0.7 * len(feats))):
                events.append(HumanStateEvent(
                    "person_down", seq.track_id, feats[0].t, feats[-1].t, 0.55,
                    {"torso_angle_deg": round(float(np.median([f.torso_angle_deg for f in feats])), 1),
                     "duration_s": round(feats[-1].t - feats[0].t, 2)},
                    f"PERSON_{seq.track_id:02d} is horizontal/low for the whole observed span "
                    f"({feats[-1].t - feats[0].t:.1f}s); fall onset not observed (could be intentional lying)."))
        return events

    def _analyze_slump(self, seq: TrackSequence) -> list[HumanStateEvent]:
        feats = seq.features
        base = [f for f in feats if f.t - feats[0].t <= self.baseline_s and f.valid]
        if len(base) < 2:
            base = feats[:2]
        base_head = float(np.median([f.head_drop_ratio for f in base]))
        base_lean = float(np.median([f.lateral_lean_ratio for f in base]))
        base_angle = float(np.median([f.torso_angle_deg for f in base]))
        events: list[HumanStateEvent] = []
        slump_start: float | None = None
        still_since: float | None = None
        last: PoseFeatures | None = None
        for f in feats:
            head_drop = f.head_drop_ratio - base_head
            lean = f.lateral_lean_ratio - base_lean
            angle_drift = f.torso_angle_deg - base_angle
            slumped = head_drop > self.head_drop_ratio or lean > self.lateral_ratio or angle_drift > 25
            if last is not None:
                disp = math.hypot(f.center_x - last.center_x, f.center_y - last.center_y) / max(f.body_scale, 1.0)
                disp += abs(f.head_drop_ratio - last.head_drop_ratio) * 0.15
                if disp < self.stillness_disp:
                    still_since = still_since if still_since is not None else last.t
                else:
                    still_since = None
            if slumped and slump_start is None:
                slump_start = f.t
            if not slumped:
                slump_start = None
            still_for = (f.t - still_since) if still_since is not None else 0.0
            if slump_start is not None and (f.t - slump_start) >= 2.0 and still_for >= self.stillness_s:
                conf = min(0.92, 0.5 + 0.2 * min(head_drop / max(self.head_drop_ratio, 1e-3), 1.5)
                           + 0.15 * min(still_for / 10.0, 1.0))
                events.append(HumanStateEvent(
                    "slump_unresponsive", seq.track_id, slump_start, feats[-1].t, conf,
                    {"head_drop_ratio_delta": round(head_drop, 3), "lateral_lean_delta": round(lean, 3),
                     "torso_angle_drift_deg": round(angle_drift, 1), "still_for_s": round(still_for, 1),
                     "baseline_s": self.baseline_s},
                    f"PERSON_{seq.track_id:02d} (seated): posture drifted from baseline at {slump_start:.1f}s "
                    f"(head dropped {head_drop:.2f} shoulder-widths, lean +{lean:.2f}, torso +{angle_drift:.0f}°) "
                    f"and stayed still for {still_for:.0f}s."))
                break
            last = f
        return events

    def _analyze_striking(self, seq: TrackSequence) -> list[HumanStateEvent]:
        """Fast repeated wrist motion = candidate striking gesture (needs a target)."""
        events: list[HumanStateEvent] = []
        speeds: list[tuple[float, float]] = []
        prev = None
        for (t, lw, rw), f in zip(seq.wrists, seq.features):
            if prev is not None:
                pt, plw, prw = prev
                dt = max(t - pt, 1e-3)
                best = 0.0
                for a, b in ((lw, plw), (rw, prw)):
                    if a and b:
                        best = max(best, math.hypot(a[0] - b[0], a[1] - b[1]) / max(f.body_scale, 1.0) / dt)
                speeds.append((t, best))
            prev = (t, lw, rw)
        fast = [(t, s) for t, s in speeds if s > 1.8]
        if len(fast) >= 3:
            start, end = fast[0][0], fast[-1][0]
            if end - start <= 8.0:
                peak = max(s for _, s in fast)
                events.append(HumanStateEvent(
                    "striking_motion", seq.track_id, start, end, min(0.85, 0.45 + 0.1 * len(fast)),
                    {"fast_wrist_frames": len(fast), "peak_wrist_speed_h_per_s": round(peak, 2)},
                    f"PERSON_{seq.track_id:02d}: {len(fast)} rapid arm swings between {start:.1f}s and {end:.1f}s "
                    f"(peak wrist speed {peak:.1f} body-heights/s)."))
        return events

class TemporalFallHead:
    """Optional learned head on keypoint windows [T,17,5] = (x, y, conf, vx, vy), bbox-normalized
    (see scripts/train_fall_tcn.py). Returns P(fall) for a 32-frame window."""

    def __init__(self, checkpoint: Path) -> None:
        import torch

        self.torch = torch
        self.model = torch.jit.load(str(checkpoint)) if checkpoint.suffix == ".ts" else None
        if self.model is None:
            payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            self.model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.model.eval()

    def score(self, window: np.ndarray) -> float:
        with self.torch.no_grad():
            x = self.torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            out = self.model(x)
            prob = self.torch.softmax(out, dim=-1)[0]
            return float(prob[-1]) if prob.numel() > 1 else float(self.torch.sigmoid(out).item())
