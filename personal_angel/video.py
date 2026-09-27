"""Video/audio ingestion and hierarchical frame sampling.

Stage 1 (cheap): decode at `scan_fps`, compute motion + scene-change scores on
downscaled grey frames. Stage 2 (dense): re-decode only candidate windows at
`dense_fps` for detection/pose. Expensive VLM calls see only selected evidence
frames. This is the "perceive only what is necessary" principle.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def media_kind(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in VIDEO_EXT:
        return "video"
    if suffix in AUDIO_EXT:
        return "audio"
    if suffix in IMAGE_EXT:
        return "image"
    raise ValueError(f"Unsupported media type: {suffix}")

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

@dataclass
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    has_audio: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()

def _has_audio_stream(path: Path) -> bool:
    if shutil.which("ffprobe") is None:
        return False
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return "audio" in out.stdout
    except Exception:
        return False

class VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if fps <= 0 or fps != fps:
            fps = 25.0
        duration = frame_count / fps if frame_count > 0 else 0.0
        self.metadata = VideoMetadata(
            path=str(self.path), fps=fps, frame_count=frame_count, width=width,
            height=height, duration_s=duration, has_audio=_has_audio_stream(self.path),
        )

    def iter_frames(self, target_fps: float, start_s: float = 0.0,
                    end_s: float | None = None, max_frames: int | None = None
                    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_index, timestamp_s, bgr_frame) at approximately target_fps."""
        meta = self.metadata
        end_s = meta.duration_s if end_s is None else min(end_s, meta.duration_s)
        stride = max(1, int(round(meta.fps / max(target_fps, 0.01))))
        first = int(start_s * meta.fps)
        last = int(end_s * meta.fps) if end_s > 0 else meta.frame_count
        wanted = list(range(first, max(first + 1, last), stride))
        if max_frames and len(wanted) > max_frames:

            idx = np.linspace(0, len(wanted) - 1, max_frames).round().astype(int)
            wanted = [wanted[i] for i in idx]
        cap = cv2.VideoCapture(str(self.path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            current = first
            wanted_set = set(wanted)
            last_wanted = wanted[-1] if wanted else first
            while current <= last_wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                if current in wanted_set:
                    yield current, current / meta.fps, frame
                current += 1
        finally:
            cap.release()

    def frame_at(self, timestamp_s: float) -> np.ndarray | None:
        index = int(round(timestamp_s * self.metadata.fps))
        cap = cv2.VideoCapture(str(self.path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
            ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()

def motion_and_scene_scores(prev_small: np.ndarray | None, small: np.ndarray
                            ) -> tuple[float, float]:
    """Cheap per-frame signals on a 96px grey thumbnail."""
    if prev_small is None:
        return 0.0, 0.0
    diff = cv2.absdiff(prev_small, small)
    motion = float(diff.mean()) / 255.0
    hist_a = cv2.calcHist([prev_small], [0], None, [32], [0, 256]).ravel()
    hist_b = cv2.calcHist([small], [0], None, [32], [0, 256]).ravel()
    hist_a /= max(hist_a.sum(), 1.0)
    hist_b /= max(hist_b.sum(), 1.0)
    scene = float(0.5 * np.abs(hist_a - hist_b).sum())
    return motion, scene

def thumbnail(frame: np.ndarray, side: int = 96) -> np.ndarray:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape[:2]
    scale = side / max(h, w)
    return cv2.resize(grey, (max(1, int(w * scale)), max(1, int(h * scale))))

WEAPON_WORDS = {"gun", "pistol", "handgun", "rifle", "knife", "firearm", "weapon", "scissors", "baseball bat"}
COLOR_WEAPON = (40, 40, 230)
COLOR_PERSON = (230, 200, 60)
COLOR_DOWN = (30, 150, 255)
COLOR_OTHER = (160, 160, 160)

def _label_box(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.45, 0.55 * scale)
    th = max(1, int(round(1.5 * scale)))
    (tw, tt), base = cv2.getTextSize(text, font, fs, th)
    y0 = max(tt + base + 4, y)
    cv2.rectangle(image, (x, y0 - tt - base - 4), (x + tw + 8, y0), color, -1)
    cv2.putText(image, text, (x + 4, y0 - base - 2), font, fs, (255, 255, 255), th, cv2.LINE_AA)

def annotate_frame(frame: np.ndarray, detections: list | None = None, poses: list | None = None,
                   timestamp_s: float | None = None, max_side: int = 960, banner: str | None = None) -> np.ndarray:
    """Draw detector boxes the way an operator expects: weapons in thick red with confidence and
    size, people in cyan, fallen/horizontal people in orange; optional pose skeleton and time stamp."""
    image = frame.copy()
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    h, w = image.shape[:2]
    line = max(1, int(round(2.2 * min(1.0, w / 960))))
    persons = [d for d in (detections or []) if d.label == "person"]
    ref_h = max((d.box_xyxy[3] - d.box_xyxy[1]) for d in persons) * scale if persons else None
    for det in detections or []:
        x1, y1, x2, y2 = [int(v * scale) for v in det.box_xyxy]
        lab = det.label.lower()
        weapon = det.source == "weapon_detector" or lab in WEAPON_WORDS
        if weapon:
            color, thick = COLOR_WEAPON, line + 2
        elif lab in {"fallen", "person_down"}:
            color, thick = COLOR_DOWN, line + 1
        elif lab == "person":
            color, thick = COLOR_PERSON, line
        else:
            color, thick = COLOR_OTHER, max(1, line - 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thick)
        tag = det.label.upper() if weapon else det.label
        if det.track_id is not None:
            tag += f" #{det.track_id}"
        text = f"{tag} {det.confidence:.2f}"
        if weapon:
            bw, bh = x2 - x1, y2 - y1
            text += f" | {bw}x{bh}px"
            if ref_h:

                cm = 170.0 * max(bw, bh) / max(ref_h, 1.0)
                text += f" ~{cm:.0f}cm"

            t = max(6, min(bw, bh) // 4)
            for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                cv2.line(image, (cx, cy), (cx + dx * t, cy), color, thick + 1)
                cv2.line(image, (cx, cy), (cx, cy + dy * t), color, thick + 1)
        _label_box(image, text, x1, y1 - 2, color, min(1.0, w / 960))
    for pose in poses or []:
        kp = np.asarray(getattr(pose, "keypoints", []), dtype=np.float32).reshape(-1, 3)
        if kp.shape[0] != 17:
            continue
        for a, b in ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)):
            if kp[a, 2] > 0.4 and kp[b, 2] > 0.4:
                cv2.line(image, (int(kp[a, 0] * scale), int(kp[a, 1] * scale)), (int(kp[b, 0] * scale), int(kp[b, 1] * scale)),
                         (120, 255, 120), max(1, line - 1), cv2.LINE_AA)
    if timestamp_s is not None or banner:
        text = (f"t={timestamp_s:6.2f}s" if timestamp_s is not None else "") + (f"  {banner}" if banner else "")
        _label_box(image, text.strip(), 6, h - 8, (30, 30, 30), min(1.0, w / 960))
    return image

def save_evidence_frame(frame: np.ndarray, path: Path, max_side: int = 960,
                        detections: list | None = None, poses: list | None = None,
                        timestamp_s: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = annotate_frame(frame, detections, poses, timestamp_s, max_side)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return path

class AnnotatedClipWriter:
    """Writes annotated frames of one candidate window to an MP4 (H.264 via ffmpeg when available,
    so the browser can play it); used by the UI as the 'evidence replay'."""

    def __init__(self, path: Path, fps: float, max_side: int = 720) -> None:
        self.path = Path(path)
        self.fps = max(2.0, float(fps))
        self.max_side = max_side
        self.tmp = self.path.with_suffix(".raw.mp4")
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self.frames = 0

    def add(self, frame: np.ndarray, detections: list | None, poses: list | None, timestamp_s: float,
            banner: str | None = None) -> None:
        image = annotate_frame(frame, detections, poses, timestamp_s, self.max_side, banner)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = image.shape[:2]
            self.size = (max(2, w - w % 2), max(2, h - h % 2))
            self.writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
            if not self.writer.isOpened():
                self.writer = None
                raise RuntimeError("cv2.VideoWriter could not open the clip file")
        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size)
        self.writer.write(image)
        self.frames += 1

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.tmp.unlink(missing_ok=True)

    def close(self) -> Path | None:
        if self.writer is None:
            return None
        self.writer.release()
        self.writer = None
        if not self.tmp.exists() or self.frames == 0:
            self.tmp.unlink(missing_ok=True)
            return None
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(self.tmp), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "24",
                   "-movflags", "+faststart", str(self.path)]
            try:
                done = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
                if done.returncode == 0 and self.path.exists():
                    self.tmp.unlink(missing_ok=True)
                    return self.path
            except subprocess.TimeoutExpired:
                pass
        self.tmp.replace(self.path)
        return self.path

def extract_audio(media_path: str | Path, out_wav: str | Path, sample_rate: int = 16000
                  ) -> Path | None:
    """Extract a mono 16 kHz WAV with ffmpeg. Returns None when no audio track."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        return None
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(media_path), "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "wav", str(out_wav)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size < 1000:
        return None
    return out_wav

def audio_duration_s(path: str | Path) -> float:
    if shutil.which("ffprobe") is None:
        return 0.0
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
         "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=30,
        check=False,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0
