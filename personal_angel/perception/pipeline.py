"""Hierarchical perception: cheap scan → candidate windows → dense inspection.

Outputs a PerceptionResult that the event builder turns into structured Events.
Nothing here calls an LLM.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..schema import AcousticEvent, AudioSegment, Detection, FrameObservation, TextRiskScores
from ..video import (AnnotatedClipWriter, VideoReader, audio_duration_s, extract_audio, media_kind,
                     motion_and_scene_scores, save_evidence_frame, thumbnail)
from .audio import create_audio_analyzer
from .detector import WEAPON_LABELS, COCO_TOOL_LABELS, create_detector
from .fixtures import load_fixture
from .pose import HumanStateAnalyzer, HumanStateEvent, create_pose_estimator
from .scene import LOCATION_LABEL, SceneUnderstanding, create_scene_analyzer, note_location, understand_scene
from .triage import create_triage

log = logging.getLogger(__name__)
ProgressFn = Callable[[str, dict[str, Any]], None]

@dataclass
class CandidateWindow:
    window_id: str
    start_s: float
    end_s: float
    score: float
    reasons: list[str]

@dataclass
class WeaponSighting:
    track_id: int | None
    label: str
    first_s: float
    last_s: float
    frames_seen: int
    frames_possible: int
    max_conf: float
    holder_track: int | None
    evidence_frames: list[int]
    best_frame: int | None = None
    best_box: tuple[float, float, float, float] | None = None
    second_opinion: str | None = None
    weapon_probability: float | None = None

    @property
    def persistence(self) -> float:
        return self.frames_seen / max(self.frames_possible, 1)

@dataclass
class PerceptionResult:
    media_path: str
    media_kind: str
    duration_s: float
    location: str
    scan: list[FrameObservation] = field(default_factory=list)
    dense: list[FrameObservation] = field(default_factory=list)
    windows: list[CandidateWindow] = field(default_factory=list)
    human_events: list[HumanStateEvent] = field(default_factory=list)
    weapons: list[WeaponSighting] = field(default_factory=list)
    audio_segments: list[AudioSegment] = field(default_factory=list)
    acoustic_events: list[AcousticEvent] = field(default_factory=list)
    text_risk: TextRiskScores = field(default_factory=TextRiskScores)
    evidence_frames: dict[int, str] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    backends: dict[str, str] = field(default_factory=dict)
    seated_context: bool = False
    scene: SceneUnderstanding = field(default_factory=SceneUnderstanding)
    clips: list[dict[str, Any]] = field(default_factory=list)
    operator_note: str | None = None
    activity_events: list[dict[str, Any]] = field(default_factory=list)
    fallen_sightings: list[dict[str, Any]] = field(default_factory=list)
    audio_kind: str = "none"

    def observation_at(self, timestamp_s: float) -> FrameObservation | None:
        pool = self.dense or self.scan
        if not pool:
            return None
        return min(pool, key=lambda o: abs(o.timestamp_s - timestamp_s))

def _weapon_like(det: Detection) -> bool:
    return det.source == "weapon_detector" or det.label.lower() in WEAPON_LABELS or det.label in COCO_TOOL_LABELS

def run_perception(media_path: str | Path, config: dict[str, Any], run_dir: Path,
                   progress: ProgressFn | None = None, scenario_hint: str | None = None,
                   telemetry: Any | None = None) -> PerceptionResult:
    """`scenario_hint` is now an optional free-text *operator note*; the environment itself is
    inferred from the pixels (CLIP zero-shot) — a note can only settle an ambiguous scene."""
    media_path = Path(media_path)
    kind = media_kind(media_path)
    fixture = load_fixture(media_path)
    project_root = Path(config.get("_project_root", "."))
    progress = progress or (lambda stage, payload: None)
    note = (scenario_hint or "").strip() or None
    fixture_location = fixture.get("location")
    location = str(fixture_location or note_location(note) or ("PHONE_LINE" if kind == "audio" else "UNKNOWN"))
    result = PerceptionResult(media_path=str(media_path), media_kind=kind, duration_s=0.0, location=location,
                              seated_context=location in {"CAR_CABIN", "VEHICLE"} or bool(fixture.get("seated_context", False)),
                              operator_note=note)
    result.scene.location, result.scene.label = location, LOCATION_LABEL.get(location, location.lower())
    result.scene.operator_note = note
    evidence_dir = run_dir / "evidence"
    t0 = time.perf_counter()

    scene_analyzer = None
    if kind in {"video", "image"}:
        scene_analyzer = create_scene_analyzer(config.get("scene", {}), project_root, fixture, note)
        if scene_analyzer is not None:
            result.backends["scene"] = scene_analyzer.name
        _run_visual(media_path, kind, config, fixture, project_root, evidence_dir, result, progress, telemetry,
                    scene_analyzer, str(fixture_location) if fixture_location else None)
    if kind in {"video", "audio"}:
        _run_audio(media_path, kind, config, fixture, run_dir, result, progress, telemetry)
    result.seated_context = result.location in {"CAR_CABIN", "VEHICLE"} or bool(fixture.get("seated_context", False))
    result.counters["perception_wall_s"] = round(time.perf_counter() - t0, 3)
    return result

def _early_scene(reader: VideoReader, analyzer, fixture_location: str | None, note: str | None,
                 result: PerceptionResult) -> None:
    """Classify the environment from two frames *before* scanning, so state machines that depend on
    it (seated slump detection in a vehicle) run with the right context."""
    if fixture_location:
        result.location = fixture_location
        result.scene.location, result.scene.label, result.scene.confidence = fixture_location, LOCATION_LABEL.get(fixture_location, fixture_location.lower()), 1.0
        result.seated_context = fixture_location in {"CAR_CABIN", "VEHICLE"}
        return
    if analyzer is None:
        return
    frames = {}
    for i, frac in enumerate((0.1, 0.5)):
        f = reader.frame_at(max(0.0, reader.metadata.duration_s * frac))
        if f is not None:
            frames[i] = _small(f)[0]
    if not frames:
        return
    su = understand_scene(analyzer, frames, [], [], note)
    result.scene.location, result.scene.label, result.scene.confidence = su.location, su.label, su.confidence
    result.scene.scores, result.scene.backend = su.scores, su.backend
    result.location = su.location
    result.seated_context = su.location in {"CAR_CABIN", "VEHICLE"}

def _small(frame: np.ndarray, side: int = 640) -> tuple[np.ndarray, float]:
    """Downscale for the scene model; returns (frame, scale) so detector boxes can be mapped onto it."""
    import cv2

    h, w = frame.shape[:2]
    s = min(1.0, side / max(h, w))
    if s >= 1.0:
        return frame, 1.0
    return cv2.resize(frame, (int(w * s), int(h * s))), s

def _run_visual(media_path: Path, kind: str, config: dict[str, Any], fixture: dict[str, Any],
                project_root: Path, evidence_dir: Path, result: PerceptionResult,
                progress: ProgressFn, telemetry: Any | None, scene_analyzer=None,
                fixture_location: str | None = None) -> None:
    vcfg = config.get("video", {})
    detector = create_detector(config.get("detector", {}), project_root, fixture)
    pose = create_pose_estimator({**config.get("pose", {}), "_project_root": str(project_root)}, fixture)
    fall_head = _load_fall_head(config.get("pose", {}), project_root)
    if fall_head is not None:
        result.backends["fall_head"] = "temporal_tcn"
    result.backends["detector"] = detector.name
    result.backends["pose"] = pose.name
    triage = create_triage(config.get("triage", {}), project_root, fixture)
    if triage is not None:
        result.backends["triage"] = triage.name

    if kind == "image":
        import cv2

        frame = cv2.imread(str(media_path))
        if frame is None:
            raise ValueError(f"cannot read image {media_path.name}")
        small, scale = _small(frame)
        if scene_analyzer is not None or fixture_location:
            su = understand_scene(scene_analyzer, {0: small}, [], [], result.operator_note, fixture_location)
            result.scene, result.location = su, su.location
            result.seated_context = su.location in {"CAR_CABIN", "VEHICLE"}
        obs = _observe(frame, 0, 0.0, detector, pose, None)
        obs.image_path = str(save_evidence_frame(frame, evidence_dir / "frame_000000.jpg",
                                                 int(vcfg.get("evidence_max_side", 960)), obs.detections, obs.poses, 0.0))
        result.scan.append(obs)
        result.evidence_frames[0] = obs.image_path
        result.duration_s = 0.0
        result.weapons = _weapon_sightings([obs])
        _second_opinion(result, {0: small}, scene_analyzer, [obs], fixture_location, {0: scale})
        result.counters.update(frames_scanned=1, frames_dense=0, frames_total=1, frames_skipped=0)
        return

    reader = VideoReader(media_path)
    result.duration_s = reader.metadata.duration_s
    if reader.metadata.duration_s and reader.metadata.duration_s < float(vcfg.get("short_clip_s", 15.0)):

        vcfg = {**vcfg, "scan_fps": max(float(vcfg.get("scan_fps", 2.0)), 4.0), "dense_fps": max(float(vcfg.get("dense_fps", 8.0)), 8.0)}
    with _span(telemetry, "scene_early"):
        _early_scene(reader, scene_analyzer, fixture_location, result.operator_note, result)
    progress("SCENE", {"detail": f"Environment inferred: {result.scene.label} ({result.scene.confidence:.0%})",
                       "scene": result.scene.to_dict()})
    analyzer = HumanStateAnalyzer(config.get("pose", {}), seated_context=result.seated_context)
    scan_fps = float(vcfg.get("scan_fps", 2.0))
    max_scan = int(vcfg.get("max_scan_frames", 900))
    progress("SCANNING", {"detail": f"Stage 1: sampling at {scan_fps} fps with {detector.name} + {pose.name}",
                          "duration_s": reader.metadata.duration_s})
    prev_small = None
    scan_frames: list[FrameObservation] = []
    frame_cache: dict[int, np.ndarray] = {}
    with _span(telemetry, "stage1_scan"):
        for frame_index, ts, frame in reader.iter_frames(scan_fps, max_frames=max_scan):
            small = thumbnail(frame)
            motion, scene = motion_and_scene_scores(prev_small, small)
            prev_small = small
            triage_scores = triage.score(frame, ts) if triage is not None else {}
            confidently_normal = bool(triage_scores) and triage_scores.get("normal", 0.0) >= float(vcfg.get("triage_skip_threshold", 0.9)) and motion < 0.08
            if confidently_normal and len(scan_frames) % 3 != 0:

                obs = FrameObservation(frame_index=frame_index, timestamp_s=ts)
                if telemetry is not None:
                    telemetry.increment("frames_triaged_skipped")
            else:
                obs = _observe(frame, frame_index, ts, detector, pose, analyzer)
            obs.triage_scores = triage_scores
            obs.motion_score, obs.scene_change_score = motion, scene
            scan_frames.append(obs)
            if _interesting(obs):
                frame_cache[frame_index] = frame
    result.scan = scan_frames
    result.counters["frames_scanned"] = len(scan_frames)
    result.counters["frames_total"] = reader.metadata.frame_count

    windows = _candidate_windows(scan_frames, reader.metadata.duration_s, vcfg, analyzer)
    result.windows = windows
    progress("CANDIDATES", {"detail": f"Stage 2: {len(windows)} candidate window(s)",
                            "windows": [w.__dict__ for w in windows]})

    dense_fps = float(vcfg.get("dense_fps", 8.0))
    dense_analyzer = HumanStateAnalyzer(config.get("pose", {}), seated_context=result.seated_context)
    dense_frames: list[FrameObservation] = []
    detector.reset()
    make_clips = bool(vcfg.get("annotated_clips", True))
    activity_samples: dict[str, list[tuple[float, int, np.ndarray]]] = {}
    with _span(telemetry, "stage3_dense"):
        for window in windows:
            writer = AnnotatedClipWriter(evidence_dir / f"{window.window_id}_annotated.mp4", dense_fps,
                                         int(vcfg.get("clip_max_side", 720))) if make_clips else None
            n_before = len(dense_frames)
            every = max(1, int(round(dense_fps / 2)))
            for frame_index, ts, frame in reader.iter_frames(dense_fps, window.start_s, window.end_s,
                                                              max_frames=int(dense_fps * 40)):
                obs = _observe(frame, frame_index, ts, detector, pose, dense_analyzer)
                dense_frames.append(obs)
                if _interesting(obs) or len(dense_frames) % max(1, int(dense_fps)) == 0:
                    frame_cache[frame_index] = frame
                bucket = activity_samples.setdefault(window.window_id, [])
                if (len(dense_frames) - n_before) % every == 0 and len(bucket) < 8 and scene_analyzer is not None:
                    bucket.append((ts, frame_index, _small(frame, 448)[0]))
                if writer is not None:
                    try:
                        writer.add(frame, obs.detections, obs.poses, ts, window.window_id.replace("_", " "))
                    except Exception as error:
                        log.warning("annotated clip write failed: %s", error)
                        writer.abort()
                        writer = None
            if writer is not None:
                try:
                    path = writer.close()
                except Exception as error:
                    log.warning("annotated clip encode failed: %s", error)
                    path = None
                if path is not None:
                    result.clips.append({"window_id": window.window_id, "start_s": window.start_s, "end_s": window.end_s,
                                         "path": str(path), "fps": dense_fps, "frames": len(dense_frames) - n_before})
    result.dense = dense_frames
    result.counters["frames_dense"] = len(dense_frames)

    unique = {o.frame_index for o in scan_frames} | {o.frame_index for o in dense_frames}
    total = max(int(reader.metadata.frame_count or 0), len(unique))
    result.counters["frames_total"] = total
    result.counters["frames_unique"] = len(unique)
    result.counters["frames_skipped"] = max(0, total - len(unique))

    if scene_analyzer is not None and activity_samples:
        with _span(telemetry, "activity_zero_shot"):
            _activity_events(result, scene_analyzer, activity_samples, windows, float(vcfg.get("activity_fight_threshold", 0.5)))

    events = dense_analyzer.analyze(fall_head) if dense_frames else []
    scan_events = analyzer.analyze(fall_head)
    result.human_events = _merge_human_events(events, scan_events)
    result.weapons = _weapon_sightings(dense_frames or scan_frames)
    result.fallen_sightings = _fallen_sightings(dense_frames or scan_frames)

    wanted: list[tuple[float, str]] = []
    for ev in result.human_events:
        wanted += [(ev.start_s, f"{ev.kind}_onset"), ((ev.start_s + ev.end_s) / 2, f"{ev.kind}_mid"),
                   (ev.end_s, f"{ev.kind}_end")]
    for w in result.weapons:
        wanted += [(best_sighting_time(dense_frames or scan_frames, w), f"{w.label}_best"),
                   (w.first_s, f"{w.label}_first"), ((w.first_s + w.last_s) / 2, f"{w.label}_mid")]
    for ae in result.activity_events:
        wanted += [(ae["start_s"], "activity_onset"), ((ae["start_s"] + ae["end_s"]) / 2, "activity_peak"), (ae["end_s"], "activity_end")]
    for fs in result.fallen_sightings:
        wanted += [(fs["first_s"], "fallen_first"), (fs["last_s"], "fallen_last")]
    for win in windows:
        wanted.append(((win.start_s + win.end_s) / 2, "window_peak"))
    if not wanted:
        for frac in (0.1, 0.5, 0.9):
            wanted.append((reader.metadata.duration_s * frac, "coverage"))
    pool = dense_frames or scan_frames
    max_side = int(vcfg.get("evidence_max_side", 960))
    saved = 0
    raw_evidence: dict[int, np.ndarray] = {}
    raw_scales: dict[int, float] = {}
    saved_obs: list[FrameObservation] = []
    for ts, why in wanted:
        if not pool:
            break
        obs = min(pool, key=lambda o: abs(o.timestamp_s - ts))
        if obs.frame_index in result.evidence_frames:
            continue
        frame = frame_cache.get(obs.frame_index)
        if frame is None:
            frame = reader.frame_at(obs.timestamp_s)
        if frame is None:
            continue
        path = evidence_dir / f"frame_{obs.frame_index:06d}_{why}.jpg"
        save_evidence_frame(frame, path, max_side, obs.detections, obs.poses, obs.timestamp_s)
        obs.image_path = str(path)
        result.evidence_frames[obs.frame_index] = str(path)
        raw_evidence[obs.frame_index], raw_scales[obs.frame_index] = _small(frame)
        saved_obs.append(obs)
        saved += 1
        if saved >= 24:
            break
    result.counters["evidence_frames"] = saved
    frame_cache.clear()

    with _span(telemetry, "scene_understanding"):
        _second_opinion(result, raw_evidence, scene_analyzer, saved_obs, fixture_location, raw_scales)

def _activity_events(result: PerceptionResult, analyzer, samples: dict[str, list[tuple[float, int, np.ndarray]]],
                     windows: list[CandidateWindow], threshold: float) -> None:
    best_overall: dict[str, float] = {}
    for window in windows:
        rows = samples.get(window.window_id) or []
        if not rows:
            continue
        try:
            per_frame = analyzer.activity_per_frame([f for _, _, f in rows])
        except Exception as error:
            log.warning("activity zero-shot failed: %s", error, exc_info=True)
            result.scene.error = result.scene.error or f"{type(error).__name__}: {error}"
            return
        violent = [(ts, fi, sc.get("fight", 0.0) + sc.get("shove", 0.0), sc) for (ts, fi, _), sc in zip(rows, per_frame)]
        for _, _, _, sc in violent:
            for k, v in sc.items():
                best_overall[k] = max(best_overall.get(k, 0.0), v)
        fallen_hits = [(ts, fi, sc.get("fallen", 0.0)) for (ts, fi, _), sc in zip(rows, per_frame) if sc.get("fallen", 0.0) >= threshold]
        if len(fallen_hits) >= 2:
            peak_f = max(fallen_hits, key=lambda h: h[2])
            result.activity_events.append({
                "window_id": window.window_id, "start_s": round(fallen_hits[0][0], 2), "end_s": round(fallen_hits[-1][0], 2),
                "kind": "fallen_activity", "confidence": round(min(0.9, 0.3 + 0.6 * peak_f[2]), 3), "peak_score": round(peak_f[2], 3),
                "mean_score": round(sum(h[2] for h in fallen_hits) / len(fallen_hits), 3), "frames_hit": len(fallen_hits),
                "frames_checked": len(rows), "peak_frame": peak_f[1], "scores": {"fallen": round(peak_f[2], 3)}})
        hits = [(ts, fi, v) for ts, fi, v, _ in violent if v >= threshold]
        mean_violent = sum(v for _, _, v, _ in violent) / max(len(violent), 1)
        if len(hits) >= 2 or (hits and len(rows) <= 2):
            peak = max(hits, key=lambda h: h[2])
            result.activity_events.append({
                "window_id": window.window_id, "start_s": round(hits[0][0], 2), "end_s": round(hits[-1][0], 2),
                "kind": "violent_activity", "confidence": round(min(0.95, 0.35 + 0.6 * peak[2]), 3),
                "peak_score": round(peak[2], 3), "mean_score": round(mean_violent, 3), "frames_hit": len(hits),
                "frames_checked": len(rows), "peak_frame": peak[1],
                "scores": {k: round(v, 3) for k, v in max(violent, key=lambda h: h[2])[3].items()},
            })
            log.info("activity zero-shot: violent activity in %s (%d/%d frames, peak %.2f)", window.window_id, len(hits), len(rows), peak[2])
    result.scene.activity = best_overall

def _second_opinion(result: PerceptionResult, frames: dict[int, np.ndarray], analyzer, observations,
                    fixture_location: str | None, scales: dict[int, float] | None = None) -> None:
    weapon_boxes: list[tuple[int, str, tuple[float, float, float, float]]] = []
    for w in result.weapons:
        if w.best_frame is not None and w.best_box is not None:
            weapon_boxes.append((w.best_frame, w.label, w.best_box))
    early = result.scene
    su = understand_scene(analyzer, frames, observations, weapon_boxes, result.operator_note, fixture_location, scales)

    if not fixture_location and early.location not in {"UNKNOWN", ""} and su.confidence <= early.confidence + 0.15:
        su.location, su.label, su.confidence = early.location, early.label, max(early.confidence, su.confidence * 0.99)
        if early.scores:
            su.scores = early.scores
    if analyzer is None and not fixture_location:
        su.location, su.label, su.confidence = early.location, early.label, early.confidence
    if not su.scores and early.scores:
        su.scores = early.scores
    if su.location == "UNKNOWN" and early.location != "UNKNOWN":
        su.location, su.label, su.confidence = early.location, early.label, early.confidence
    su.backend = su.backend if analyzer is not None else early.backend
    su.activity = early.activity or su.activity
    su.error = su.error or early.error
    result.scene = su
    result.location = su.location
    for w in result.weapons:
        for check in su.object_checks:
            if check.frame_index == w.best_frame and check.label == w.label:
                w.second_opinion, w.weapon_probability = check.verdict, check.weapon_probability

def _load_fall_head(pose_cfg: dict[str, Any], project_root: Path):
    ckpt = pose_cfg.get("fall", {}).get("temporal_checkpoint")
    if not ckpt:
        return None
    path = project_root / str(ckpt)
    if not path.exists():
        log.warning("Fall TCN checkpoint %s not found; rules only", path)
        return None
    try:
        from .pose import TemporalFallHead

        return TemporalFallHead(path)
    except Exception as error:
        log.warning("Fall TCN unavailable (%s); rules only", error)
        return None

def _observe(frame: np.ndarray, frame_index: int, ts: float, detector, pose, analyzer) -> FrameObservation:
    detections = detector.detect(frame, ts, frame_index)
    people = [(d.track_id, d.box_xyxy) for d in detections if d.label == "person"]
    poses = pose.estimate(frame, ts, frame_index, people) if people or pose.name == "fixture_pose" else []
    if analyzer is not None and poses:
        analyzer.observe(poses, ts)
    return FrameObservation(frame_index=frame_index, timestamp_s=ts, detections=detections, poses=poses)

def _interesting(obs: FrameObservation) -> bool:
    if any(_weapon_like(d) or d.source == "fallen_detector" for d in obs.detections):
        return True
    return obs.motion_score > 0.12 or obs.scene_change_score > 0.35

def _candidate_windows(scan: list[FrameObservation], duration_s: float, vcfg: dict[str, Any],
                       analyzer: HumanStateAnalyzer) -> list[CandidateWindow]:
    if not scan:
        return []
    pad = float(vcfg.get("window_pad_s", 2.0))
    max_windows = int(vcfg.get("max_windows", 4))
    scores: list[tuple[float, float, list[str]]] = []
    motion = np.array([o.motion_score for o in scan])
    m_hi = float(np.percentile(motion, 90)) if len(motion) > 4 else 1.0
    for obs in scan:
        reasons: list[str] = []
        s = 0.0
        if any(_weapon_like(d) for d in obs.detections):
            s += 0.9
            reasons.append("weapon-like object detected")
        if any(d.source == "fallen_detector" for d in obs.detections):
            s += 0.7
            reasons.append("fallen-person detector fired")
        if m_hi > 0 and obs.motion_score >= max(m_hi, 0.08):
            s += 0.4
            reasons.append("motion peak")
        if obs.scene_change_score > 0.35:
            s += 0.2
            reasons.append("scene change")
        if obs.triage_scores:
            alarm = max((v for k, v in obs.triage_scores.items() if k != "normal"), default=0.0)
            if alarm >= 0.5:
                s += 0.5 * alarm
                reasons.append(f"stage-1 triage {max(obs.triage_scores, key=obs.triage_scores.get)} {alarm:.2f}")
        for p in obs.poses:
            from .pose import compute_features

            f = compute_features(p, obs.timestamp_s)
            if f.torso_angle_deg >= analyzer.lying_angle or f.aspect_ratio >= analyzer.lying_ar:
                s += 0.5
                reasons.append("person horizontal/low posture")
                break
        scores.append((obs.timestamp_s, s, reasons))

    for ev in analyzer.analyze():
        scores.append((ev.start_s, 1.0, [f"scan-level {ev.kind} candidate"]))
    peaks = [(t, s, r) for t, s, r in scores if s > 0.3]
    peaks.sort(key=lambda x: -x[1])
    windows: list[CandidateWindow] = []
    for t, s, reasons in peaks:
        start, end = max(0.0, t - pad), min(duration_s, t + pad + 1.0)
        merged = False
        for w in windows:
            if start <= w.end_s and end >= w.start_s:
                w.start_s, w.end_s = min(w.start_s, start), max(w.end_s, end)
                w.score = max(w.score, s)
                for r in reasons:
                    if r not in w.reasons:
                        w.reasons.append(r)
                merged = True
                break
        if not merged:
            windows.append(CandidateWindow(f"window_{len(windows) + 1:02d}", start, end, s, list(reasons)))
        if len(windows) >= max_windows:
            break
    if not windows:

        if duration_s <= 20:
            windows.append(CandidateWindow("window_01", 0.0, duration_s, 0.2, ["short clip full coverage"]))
        else:
            for i in range(min(3, max_windows)):
                c = duration_s * (i + 1) / 4
                windows.append(CandidateWindow(f"window_{i + 1:02d}", max(0, c - pad), min(duration_s, c + pad),
                                               0.1, ["coverage sample"]))
    windows.sort(key=lambda w: w.start_s)

    return windows

def _merge_human_events(dense: list[HumanStateEvent], scan: list[HumanStateEvent]) -> list[HumanStateEvent]:
    merged = list(dense)
    for ev in scan:
        if not any(e.kind == ev.kind and e.track_id == ev.track_id and abs(e.start_s - ev.start_s) < 4 for e in merged):
            merged.append(ev)
    merged.sort(key=lambda e: e.start_s)
    return merged

def _weapon_sightings(frames: list[FrameObservation]) -> list[WeaponSighting]:
    """N-of-M temporal persistence per (holder track, label)."""
    by_key: dict[tuple[int | None, str], WeaponSighting] = {}
    for obs in frames:
        for det in obs.detections:
            if not _weapon_like(det):
                continue
            label = det.label.lower()
            key = (det.track_id, label)
            sight = by_key.get(key)
            if sight is None:
                by_key[key] = WeaponSighting(det.track_id, label, obs.timestamp_s, obs.timestamp_s, 1, 0,
                                             det.confidence, det.track_id, [obs.frame_index],
                                             best_frame=obs.frame_index, best_box=det.box_xyxy)
            else:
                sight.last_s = obs.timestamp_s
                sight.frames_seen += 1
                if det.confidence > sight.max_conf:
                    sight.max_conf = det.confidence
                    sight.best_frame, sight.best_box = obs.frame_index, det.box_xyxy
                if len(sight.evidence_frames) < 6:
                    sight.evidence_frames.append(obs.frame_index)
    for sight in by_key.values():
        sight.frames_possible = sum(1 for o in frames if sight.first_s <= o.timestamp_s <= sight.last_s) or 1
    return [s for s in by_key.values() if s.frames_seen >= 2 or s.max_conf > 0.8]

def _fallen_sightings(frames: list[FrameObservation]) -> list[dict[str, Any]]:
    """The fallen-person detector (melihuzunoglu YOLO11) as an independent fall signal: N-of-M per person track."""
    by_track: dict[int | None, dict[str, Any]] = {}
    for obs in frames:
        for det in obs.detections:
            if det.source != "fallen_detector":
                continue
            key = det.track_id
            sight = by_track.get(key)
            if sight is None:
                by_track[key] = {"track_id": key, "first_s": obs.timestamp_s, "last_s": obs.timestamp_s, "frames_seen": 1,
                                 "max_conf": det.confidence, "best_frame": obs.frame_index, "frames": [obs.frame_index]}
            else:
                sight["last_s"] = obs.timestamp_s
                sight["frames_seen"] += 1
                if det.confidence > sight["max_conf"]:
                    sight["max_conf"], sight["best_frame"] = det.confidence, obs.frame_index
                if len(sight["frames"]) < 6:
                    sight["frames"].append(obs.frame_index)
    out = []
    for sight in by_track.values():
        possible = sum(1 for o in frames if sight["first_s"] <= o.timestamp_s <= sight["last_s"]) or 1
        sight["frames_possible"] = possible
        sight["persistence"] = sight["frames_seen"] / possible

        upright_before = False
        for o in frames:
            if o.timestamp_s >= sight["first_s"]:
                break
            for p in o.poses:
                if p.track_id == sight["track_id"]:
                    from .pose import compute_features

                    if compute_features(p, o.timestamp_s).torso_angle_deg < 35:
                        upright_before = True
        sight["upright_before"] = upright_before
        if sight["frames_seen"] >= 2 or sight["max_conf"] >= 0.75:
            out.append(sight)
    return out

def best_sighting_time(frames: list[FrameObservation], sight: WeaponSighting) -> float:
    for o in frames:
        if o.frame_index == sight.best_frame:
            return o.timestamp_s
    return (sight.first_s + sight.last_s) / 2

def _run_audio(media_path: Path, kind: str, config: dict[str, Any], fixture: dict[str, Any],
               run_dir: Path, result: PerceptionResult, progress: ProgressFn, telemetry: Any | None) -> None:
    acfg = config.get("audio", {})
    analyzer = create_audio_analyzer(acfg, fixture)
    result.backends["audio"] = analyzer.name
    wav = run_dir / "audio.wav"
    if analyzer.name == "fixture_audio":
        if kind == "audio":
            result.duration_s = max(result.duration_s, audio_duration_s(media_path) or float(fixture.get("duration_s", 0.0)))
        segments, acoustic, risk = analyzer.analyze(media_path)
    else:
        extracted = extract_audio(media_path, wav)
        if extracted is None:
            result.counters["audio_seconds"] = 0
            return
        if kind == "audio":
            result.duration_s = max(result.duration_s, audio_duration_s(extracted))
        progress("AUDIO", {"detail": f"Transcribing/translating with {analyzer.name}"})
        with _span(telemetry, "audio_analysis"):
            segments, acoustic, risk = analyzer.analyze(extracted)
    result.audio_segments = segments
    result.acoustic_events = acoustic
    result.text_risk = risk
    music_s = sum(a.end_s - a.start_s for a in acoustic if a.label in {"music", "singing"})
    total = max(result.duration_s, 1e-3)
    if music_s >= 0.5 * total and music_s > 3:
        result.audio_kind = "music"
    elif segments and music_s > 3:
        result.audio_kind = "mixed"
    elif segments:
        result.audio_kind = "speech"
    result.counters["audio_seconds"] = round(result.duration_s, 2)
    result.counters["audio_segments"] = len(segments)

class _NullSpan:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

def _span(telemetry: Any | None, name: str):
    if telemetry is None:
        return _NullSpan()
    return telemetry.span(name)
