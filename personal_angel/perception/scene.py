"""Scene understanding: *infer* the environment instead of assuming it.

Three cheap zero-shot checks with one local CLIP model (ViT-B/32, ~600 MB,
runs on CPU in ~0.1 s per crop):

  1. scene      – where is this? (home room, nursery, car cabin, office, street, ...)
  2. age group  – who is this person? (baby/toddler, child, adult, elderly)
  3. object     – second opinion on every weapon-like detector box
                  (real handgun / real knife / toy / phone / household object)

The results feed the event builder (severity priors), the master agent prompt
(context) and the UI ("Scene understanding" card). A fixture backend keeps the
whole system testable without the model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SCENE_PROMPTS: list[tuple[str, str, str]] = [

    ("HOME_ROOM", "home interior", "a photo of a living room, bedroom or hallway inside a home"),
    ("NURSERY", "nursery / child's room", "a photo of a nursery or a child's room with a crib or toys"),
    ("KITCHEN", "kitchen", "a photo of a kitchen"),
    ("CAR_CABIN", "vehicle cabin", "a photo taken inside a car showing the seats, a passenger or the dashboard"),
    ("OFFICE", "office / classroom", "a photo of an office, classroom or meeting room"),
    ("CORRIDOR", "corridor / stairs", "a photo of a corridor, staircase or lobby of a building"),
    ("RETAIL", "shop / gas station", "a photo of the inside of a shop, store, bank or gas station"),
    ("OUTDOOR", "street / outdoor", "a photo of a street, parking lot, park or other outdoor area"),
    ("HOSPITAL", "hospital / care room", "a photo of a hospital ward or care-home room"),
]
AGE_PROMPTS: list[tuple[str, str]] = [
    ("baby", "a photo of a baby or toddler crawling, sitting or lying on the floor"),
    ("child", "a photo of a young child"),
    ("adult", "a photo of an adult person"),
    ("elderly", "a photo of an elderly person with grey hair"),
]
OBJECT_PROMPTS: list[tuple[str, str]] = [
    ("handgun", "a close-up photo of a real handgun or pistol"),
    ("rifle", "a close-up photo of a real rifle or shotgun"),
    ("knife", "a close-up photo of a real knife with a metal blade"),
    ("toy", "a close-up photo of a colorful plastic toy"),
    ("phone", "a close-up photo of a mobile phone or remote control"),
    ("hand", "a close-up photo of an empty hand or an arm"),
    ("household", "a close-up photo of a household object such as a bottle, cup, tool or bag"),
]
WEAPON_KEYS = {"handgun", "rifle", "knife"}
ACTIVITY_PROMPTS: list[tuple[str, str]] = [
    ("fight", "a photo of two people fighting, punching, kicking or attacking each other"),
    ("shove", "a photo of a person grabbing, shoving or wrestling another person"),
    ("weapon", "a photo of a person pointing a gun or holding a knife toward someone"),
    ("fallen", "a photo of a person collapsed or lying on the floor"),
    ("calm", "a photo of people standing, walking or sitting calmly"),
    ("talk", "a photo of people talking, hugging or working together"),
    ("play", "a photo of people playing or dancing"),
]
VIOLENT_KEYS = {"fight", "shove", "weapon"}
_SCENE_WORDS = [
    ("CAR_CABIN", ("car", "vehicle", "cabin", "taxi", "dashboard", "back seat", "backseat", "passenger seat", "driver")),
    ("NURSERY", ("nursery", "crib", "cot", "baby room", "changing table")),
    ("KITCHEN", ("kitchen",)),
    ("HOSPITAL", ("hospital", "ward", "clinic", "care home")),
    ("RETAIL", ("restaurant", "bar", "shop", "store", "market", "gas station", "petrol", "cafe", "counter")),
    ("OFFICE", ("office", "classroom", "meeting room", "desk")),
    ("CORRIDOR", ("corridor", "hallway", "staircase", "stairs", "lobby", "elevator")),
    ("OUTDOOR", ("street", "outdoor", "outside", "parking", "sidewalk", "park", "road", "garden", "yard")),
    ("HOME_ROOM", ("living room", "bedroom", "home", "apartment", "house", "couch", "sofa", "wooden floor", "indoor")),
]

def location_from_text(text: str | None) -> str | None:
    """Map a free-text scene description (from the VLM) to a location code."""
    if not text:
        return None
    t = text.lower()
    for code, words in _SCENE_WORDS:
        if any(w in t for w in words):
            return code
    return None

def _as_tensor(out):
    """transformers returns a tensor (<=4.x) or a model output object (5.x) from get_*_features."""
    import torch

    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(out, attr, None)
        if val is not None and torch.is_tensor(val):
            return val
    if isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(f"unexpected CLIP output type {type(out).__name__}")
MIN_AGE_CROP_PX = 56
MIN_AGE_CROP_FRACTION = 0.14
LOCATION_LABEL = {code: label for code, label, _ in SCENE_PROMPTS}
LOCATION_LABEL.update({"ROOM_01": "room", "PHONE_LINE": "phone line / voicemail", "UNKNOWN": "unknown"})

@dataclass
class PersonProfile:
    track_id: int | None
    age_group: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"track_id": self.track_id, "age_group": self.age_group, "confidence": round(self.confidence, 3),
                "scores": {k: round(v, 3) for k, v in self.scores.items()}, "note": self.note}

@dataclass
class ObjectCheck:
    label: str
    frame_index: int
    verdict: str
    weapon_probability: float
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "frame_index": self.frame_index, "verdict": self.verdict,
                "weapon_probability": round(self.weapon_probability, 3),
                "scores": {k: round(v, 3) for k, v in self.scores.items()}}

@dataclass
class SceneUnderstanding:
    location: str = "UNKNOWN"
    label: str = "unknown"
    confidence: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)
    people: list[PersonProfile] = field(default_factory=list)
    object_checks: list[ObjectCheck] = field(default_factory=list)
    operator_note: str | None = None
    backend: str = "none"
    frames_used: int = 0
    activity: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    vlm_scene: str | None = None

    def update_from_text(self, text: str | None) -> bool:
        """Use the VLM's own description of the environment when the zero-shot model failed or was unsure."""
        loc = location_from_text(text)
        if not loc:
            return False
        self.vlm_scene = (text or "")[:160]
        if self.location in {"UNKNOWN", ""} or self.confidence < 0.45:
            self.location, self.label, self.confidence = loc, LOCATION_LABEL.get(loc, loc.lower()), max(self.confidence, 0.6)
            self.backend = (self.backend + "+vlm").replace("none+vlm", "vlm")
            return True
        return False

    def description(self) -> str:
        who = ", ".join((f"PERSON_{p.track_id:02d}: " if p.track_id is not None else "person: ")
                        + (f"{p.age_group} ({p.confidence:.0%})" if p.age_group != "unknown" else "age not judged (figure too small)")
                        for p in self.people) or "no person profiled"
        objs = "; ".join(f"{o.label}@{o.frame_index}: {o.verdict} (weapon p={o.weapon_probability:.2f})"
                         for o in self.object_checks[:4])
        text = f"{self.label} (scene confidence {self.confidence:.0%}); {who}"
        if objs:
            text += f"; object second opinion: {objs}"
        if self.activity:
            top = sorted(self.activity.items(), key=lambda kv: -kv[1])[:2]
            text += "; activity zero-shot: " + ", ".join(f"{k} {v:.2f}" for k, v in top)
        if self.error:
            text += f"; scene model error: {self.error[:80]}"
        if self.operator_note:
            text += f"; operator note: {self.operator_note[:120]}"
        return text

    def age_of(self, track_id: int | None) -> str | None:
        for p in self.people:
            if p.track_id == track_id:
                return None if p.age_group == "unknown" else p.age_group
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"location": self.location, "label": self.label, "confidence": round(self.confidence, 3),
                "scores": {k: round(v, 3) for k, v in self.scores.items()},
                "people": [p.to_dict() for p in self.people],
                "object_checks": [o.to_dict() for o in self.object_checks],
                "operator_note": self.operator_note, "backend": self.backend, "frames_used": self.frames_used,
                "activity": {k: round(v, 3) for k, v in self.activity.items()}, "error": self.error,
                "vlm_scene": self.vlm_scene, "description": self.description()}

class SceneAnalyzer:
    name = "abstract"

    def scene(self, frames: list[np.ndarray]) -> tuple[str, str, float, dict[str, float]]:
        raise NotImplementedError

    def activity(self, frames: list[np.ndarray]) -> dict[str, float]:
        raise NotImplementedError

    def activity_per_frame(self, frames: list[np.ndarray]) -> list[dict[str, float]]:
        return [self.activity([f]) for f in frames]

    def selftest(self) -> dict[str, Any]:
        return {"ok": True, "backend": self.name}

    def age_group(self, crop: np.ndarray) -> tuple[str, float, dict[str, float]]:
        raise NotImplementedError

    def object_check(self, crop: np.ndarray, label: str) -> ObjectCheck:
        raise NotImplementedError

class FixtureSceneAnalyzer(SceneAnalyzer):
    """Replays what a fixture declares (location, ages, object verdicts)."""
    name = "fixture_scene"

    def __init__(self, fixture: dict[str, Any], note: str | None) -> None:
        self.fixture = fixture
        self.note = note

    def scene(self, frames: list[np.ndarray]) -> tuple[str, str, float, dict[str, float]]:
        loc = str(self.fixture.get("location") or note_location(self.note) or "ROOM_01")
        return loc, LOCATION_LABEL.get(loc, loc.lower()), 0.9, {loc: 0.9}

    def age_group(self, crop: np.ndarray) -> tuple[str, float, dict[str, float]]:
        return "adult", 0.8, {"adult": 0.8}

    def object_check(self, crop: np.ndarray, label: str) -> ObjectCheck:
        verdict = str(self.fixture.get("object_verdicts", {}).get(label, "weapon"))
        p = 0.85 if verdict == "weapon" else 0.15
        return ObjectCheck(label, 0, verdict, p, {verdict: 0.85})

    def activity(self, frames: list[np.ndarray]) -> dict[str, float]:
        return dict(self.fixture.get("activity", {"calm": 0.9}))

    def activity_per_frame(self, frames: list[np.ndarray]) -> list[dict[str, float]]:
        return [self.activity([f]) for f in frames]

    def selftest(self) -> dict[str, Any]:
        return {"ok": True, "backend": self.name}

class ClipSceneAnalyzer(SceneAnalyzer):
    """Zero-shot CLIP (transformers). Text embeddings are cached per prompt set."""
    name = "clip_zero_shot"

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "auto",
                 project_root: Path | None = None) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        local = (project_root / "models" / "clip") if project_root else None
        source = str(local) if local is not None and local.exists() else model_id
        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.model = CLIPModel.from_pretrained(source).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(source)
        self._text_cache: dict[str, Any] = {}
        self.name = f"clip_zero_shot({Path(source).name})"

    def _to_device(self, batch):
        try:
            return batch.to(self.device)
        except Exception:
            return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in dict(batch).items()}

    def _embed_text(self, inputs):
        """Projected text embeddings, computed through the sub-modules so it works on transformers 4.x and 5.x
        (5.x returns BaseModelOutputWithPooling from get_text_features)."""
        m = self.model
        try:
            out = m.text_model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
            pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
            return m.text_projection(pooled)
        except Exception:
            return _as_tensor(m.get_text_features(**inputs))

    def _embed_images(self, inputs):
        m = self.model
        try:
            out = m.vision_model(pixel_values=inputs["pixel_values"])
            pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
            return m.visual_projection(pooled)
        except Exception:
            return _as_tensor(m.get_image_features(**inputs))

    def _text(self, key: str, prompts: list[str]):
        if key not in self._text_cache:
            with self.torch.no_grad():
                inputs = self._to_device(self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True))
                emb = self._embed_text(inputs).float()
                self._text_cache[key] = emb / emb.norm(dim=-1, keepdim=True)
        return self._text_cache[key]

    def _scores(self, images: list[np.ndarray], key: str, prompts: list[str]) -> np.ndarray:
        from PIL import Image

        pil = [Image.fromarray(np.ascontiguousarray(im[:, :, ::-1])) for im in images]
        with self.torch.no_grad():
            inputs = self._to_device(self.processor(images=pil, return_tensors="pt"))
            emb = self._embed_images(inputs).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            text = self._text(key, prompts)
            if emb.shape[-1] != text.shape[-1]:
                raise RuntimeError(f"CLIP embedding size mismatch: image {emb.shape[-1]} vs text {text.shape[-1]}")
            logits = 100.0 * emb @ text.T
            probs = logits.softmax(dim=-1).mean(dim=0)
        return probs.float().cpu().numpy()

    def activity(self, frames: list[np.ndarray]) -> dict[str, float]:
        """Per-frame activity scores averaged over the given frames (fight / shove / weapon / fallen / calm ...)."""
        probs = self._scores(frames, "activity", [p for _, p in ACTIVITY_PROMPTS])
        return {k: float(p) for (k, _), p in zip(ACTIVITY_PROMPTS, probs)}

    def activity_per_frame(self, frames: list[np.ndarray]) -> list[dict[str, float]]:
        return [self.activity([f]) for f in frames]

    def selftest(self) -> dict[str, Any]:
        """Run every prompt set on a synthetic image so a broken install fails loudly (health check / CLI)."""
        import time

        t0 = time.perf_counter()
        img = np.full((224, 224, 3), 128, dtype=np.uint8)
        self.scene([img]); self.age_group(img); self.object_check(img, "gun"); self.activity([img])
        return {"ok": True, "backend": self.name, "device": self.device, "ms": round((time.perf_counter() - t0) * 1000)}

    def scene(self, frames: list[np.ndarray]) -> tuple[str, str, float, dict[str, float]]:
        if not frames:
            return "UNKNOWN", "unknown", 0.0, {}
        probs = self._scores(frames[:4], "scene", [p for _, _, p in SCENE_PROMPTS])
        scores = {code: float(p) for (code, _, _), p in zip(SCENE_PROMPTS, probs)}
        best = max(scores, key=scores.get)
        return best, LOCATION_LABEL[best], scores[best], scores

    def age_group(self, crop: np.ndarray) -> tuple[str, float, dict[str, float]]:
        probs = self._scores([crop], "age", [p for _, p in AGE_PROMPTS])
        scores = {k: float(p) for (k, _), p in zip(AGE_PROMPTS, probs)}
        best = max(scores, key=scores.get)
        return best, scores[best], scores

    def object_check(self, crop: np.ndarray, label: str) -> ObjectCheck:
        probs = self._scores([crop], "object", [p for _, p in OBJECT_PROMPTS])
        scores = {k: float(p) for (k, _), p in zip(OBJECT_PROMPTS, probs)}
        weapon_p = sum(v for k, v in scores.items() if k in WEAPON_KEYS)
        best = max(scores, key=scores.get)
        if weapon_p >= 0.5:
            verdict = "weapon"
        elif best in WEAPON_KEYS:
            verdict = "uncertain"
        else:
            verdict = best
        return ObjectCheck(label, 0, verdict, weapon_p, scores)

def note_location(note: str | None) -> str | None:
    """Only a *strong* operator note can pin the location (used when the visual scene is ambiguous)."""
    if not note:
        return None
    n = note.lower()
    if any(k in n for k in ("car cabin", "robotaxi", "driverless", "taxi", "vehicle cabin", "in the car", "passenger")):
        return "CAR_CABIN"
    if any(k in n for k in ("voicemail", "phone call", "phone message")):
        return "PHONE_LINE"
    if any(k in n for k in ("nursery", "daycare", "childcare")):
        return "NURSERY"
    return None

def create_scene_analyzer(config: dict[str, Any], project_root: Path, fixture: dict[str, Any] | None,
                          note: str | None) -> SceneAnalyzer | None:
    backend = str(config.get("backend", "fixture"))
    if backend == "fixture":
        return FixtureSceneAnalyzer(fixture or {}, note)
    if backend == "none":
        return None
    if backend == "clip":
        try:
            from .registry import cached

            return cached("scene", config, lambda: ClipSceneAnalyzer(str(config.get("model", "openai/clip-vit-base-patch32")),
                                                                      str(config.get("device", "auto")), project_root), str(project_root))
        except Exception as error:
            log.warning("CLIP scene analyzer unavailable (%s); scene will be inferred from the VLM / operator note", error, exc_info=True)
            return None
    raise ValueError(f"Unknown scene backend: {backend}")

def expand_box(box: tuple[float, float, float, float], w: int, h: int, factor: float = 1.3, min_side: int = 48
               ) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max((x2 - x1) * factor, min_side), max((y2 - y1) * factor, min_side)
    return (int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2)), int(min(w, cx + bw / 2)), int(min(h, cy + bh / 2)))

def understand_scene(analyzer: SceneAnalyzer | None, frames: dict[int, np.ndarray], observations: list,
                     weapon_boxes: list[tuple[int, str, tuple[float, float, float, float]]],
                     note: str | None, fixture_location: str | None = None,
                     scales: dict[int, float] | None = None) -> SceneUnderstanding:
    """Run the three checks. `frames` maps frame_index -> BGR frame; when a frame was downscaled,
    `scales[frame_index]` is the factor that maps original detector coordinates onto it."""
    scales = scales or {}

    def scaled(box, frame_index):
        k = float(scales.get(frame_index, 1.0))
        return tuple(v * k for v in box)

    su = SceneUnderstanding(operator_note=note)
    if fixture_location:
        su.location, su.label, su.confidence = fixture_location, LOCATION_LABEL.get(fixture_location, fixture_location.lower()), 1.0
    if analyzer is None:
        loc = fixture_location or note_location(note) or "ROOM_01"
        su.location, su.label, su.confidence, su.backend = loc, LOCATION_LABEL.get(loc, loc.lower()), 0.3, "note_only"
        return su
    su.backend = analyzer.name
    ordered = [frames[i] for i in sorted(frames)]
    if not ordered:
        return su
    su.frames_used = min(4, len(ordered))
    picks = [ordered[int(i)] for i in np.linspace(0, len(ordered) - 1, su.frames_used)]
    try:
        loc, label, conf, scores = analyzer.scene(picks)
        su.scores = scores
        strong_note = note_location(note)
        if fixture_location:
            pass
        elif strong_note and conf < 0.55:
            su.location, su.label, su.confidence = strong_note, LOCATION_LABEL.get(strong_note, strong_note), conf
        else:
            su.location, su.label, su.confidence = loc, label, conf
    except Exception as error:
        log.warning("scene classification failed: %s", error, exc_info=True)
        su.error = f"{type(error).__name__}: {error}"

    best_crop: dict[int | None, tuple[float, np.ndarray]] = {}
    for obs in observations:
        frame = frames.get(obs.frame_index)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        for det in obs.detections:
            if det.label != "person":
                continue
            box = scaled(det.box_xyxy, obs.frame_index)
            area = (box[2] - box[0]) * (box[3] - box[1])
            if det.track_id in best_crop and best_crop[det.track_id][0] >= area:
                continue
            x1, y1, x2, y2 = expand_box(box, w, h, 1.1)
            crop = frame[y1:y2, x1:x2]
            if crop.size and min(crop.shape[:2]) >= 12:
                best_crop[det.track_id] = (area, crop)

    frame_h = max((f.shape[0] for f in frames.values()), default=0)
    min_h = max(MIN_AGE_CROP_PX, int(MIN_AGE_CROP_FRACTION * frame_h))
    ranked_people = sorted(best_crop.items(), key=lambda kv: -kv[1][0])[:4]
    for track_id, (_, crop) in ranked_people:
        if crop.shape[0] < min_h:
            su.people.append(PersonProfile(track_id, "unknown", 0.0, {}, f"figure only {crop.shape[0]} px tall (min {min_h})"))
            continue
        try:
            group, conf, scores = analyzer.age_group(crop)
            if group in {"baby", "child"} and conf < 0.6:
                group = "unknown"
            su.people.append(PersonProfile(track_id, group, conf, scores))
        except Exception as error:
            log.warning("age-group check failed: %s", error, exc_info=True)
            su.error = su.error or f"{type(error).__name__}: {error}"
    for frame_index, label, box in weapon_boxes[:6]:
        frame = frames.get(frame_index)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = expand_box(scaled(box, frame_index), w, h, 1.4, 64)
        crop = frame[y1:y2, x1:x2]
        if not crop.size or min(crop.shape[:2]) < 8:
            continue
        try:
            check = analyzer.object_check(crop, label)
            check.frame_index = frame_index
            su.object_checks.append(check)
        except Exception as error:
            log.warning("object check failed: %s", error, exc_info=True)
            su.error = su.error or f"{type(error).__name__}: {error}"
    return su
