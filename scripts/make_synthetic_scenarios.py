"""Generate clearly-synthetic rehearsal media + fixture sidecars for the five
demo scenarios. Used by the tests and by the UI rehearsal mode. These are
software tests of the agent loop, NOT model-accuracy evidence.

  python scripts/make_synthetic_scenarios.py --out tests/.fixtures
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import wave
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

W, H, FPS = 640, 480, 15

def _draw_person(img, box, posture, color=(40, 120, 220)):
    x1, y1, x2, y2 = [int(v) for v in box]
    from personal_angel.perception.fixtures import synth_keypoints

    pts = synth_keypoints([x1, y1, x2, y2], posture)
    bones = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
    for a, b in bones:
        cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), color, 6)
    cv2.circle(img, (int(pts[0][0]), int(pts[0][1])), 14, color, -1)

def _write_video(path: Path, seconds: float, tracks: list[dict], background=(30, 30, 30), props=None):
    from personal_angel.perception.fixtures import _interp_box

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(seconds * FPS)
    for i in range(n):
        t = i / FPS
        img = np.full((H, W, 3), background, dtype=np.uint8)
        cv2.putText(img, "SYNTHETIC REHEARSAL CLIP - not real footage", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        for p in props or []:
            cv2.rectangle(img, p[0], p[1], p[2], -1)
        for tr in tracks:
            interp = _interp_box(tr["keyframes"], t)
            if interp is None:
                continue
            box, posture = interp
            if tr.get("label", "person") == "person":
                _draw_person(img, box, posture, tr.get("color", (40, 120, 220)))
            else:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), tr.get("color", (0, 0, 255)), -1)
        cv2.putText(img, f"t={t:5.1f}s", (W - 110, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
        writer.write(img)
    writer.release()
    _to_h264(path)

def _to_h264(path: Path) -> None:
    """Browsers cannot play OpenCV's mp4v stream; re-encode with ffmpeg when available."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return
    tmp = path.with_suffix(".h264.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-preset", "veryfast", "-crf", "23", "-movflags", "+faststart", str(tmp)]
    if subprocess.run(cmd, check=False).returncode == 0 and tmp.exists():
        tmp.replace(path)

def _write_wav(path: Path, seconds: float, tone_hz: float = 220.0):
    sr = 16000
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(int(sr * seconds)):
            v = int(6000 * math.sin(2 * math.pi * tone_hz * i / sr) * (0.5 + 0.5 * math.sin(i / 4000)))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))

def scenario_car_pregnant(out: Path) -> Path:
    tracks = [{"track_id": 1, "label": "person", "keyframes": [
        {"t": 0.0, "box": [230, 90, 430, 470], "posture": "seated"},
        {"t": 7.0, "box": [230, 90, 430, 470], "posture": "seated"},
        {"t": 9.5, "box": [200, 170, 450, 478], "posture": "slumped"},
        {"t": 24.0, "box": [200, 170, 450, 478], "posture": "slumped"}]}]
    path = out / "car_pregnant_passenger.mp4"
    _write_video(path, 24, tracks, background=(45, 40, 35), props=[((0, 300), (640, 480), (70, 60, 55))])
    fixture = {"location": "CAR_CABIN", "seated_context": True, "tracks": tracks, "user_answer": None,
               "audio": {"segments": [
                   {"start_s": 6.0, "end_s": 8.5, "text": "No me siento bien... me duele mucho el vientre, creo que son contracciones.",
                    "language": "es", "translation_en": "I don't feel well... my belly hurts a lot, I think these are contractions.", "confidence": 0.8}],
                   "acoustic": [{"start_s": 9.0, "end_s": 13.0, "label": "moaning_in_pain", "confidence": 0.62}]},
               "vlm": {"support": {"slump_unresponsive": 0.6, "distress_speech": 0.6},
                       "description": {"slump_unresponsive": "A seated passenger has slid sideways against the door, head tilted down, one hand on the abdomen, eyes closed; no driver is present."}},
               "critic": {"verdict": {"slump_unresponsive": "supported", "distress_speech": "supported"},
                          "alternative": {"slump_unresponsive": "The passenger could have fallen asleep in an awkward position; however the spoken complaint of pain makes sleep unlikely."}}}
    path.with_suffix(".mp4.fixture.json").write_text(json.dumps(fixture, indent=1))
    return path

def scenario_nursery_abuse(out: Path) -> Path:
    tracks = [
        {"track_id": 1, "label": "person", "color": (60, 60, 200), "keyframes": [
            {"t": 0.0, "box": [80, 60, 240, 440], "posture": "upright"},
            {"t": 3.0, "box": [180, 60, 340, 440], "posture": "upright"},
            {"t": 3.6, "box": [190, 60, 350, 440], "posture": "striking"},
            {"t": 4.2, "box": [190, 60, 350, 440], "posture": "upright"},
            {"t": 4.8, "box": [190, 60, 350, 440], "posture": "striking"},
            {"t": 5.4, "box": [190, 60, 350, 440], "posture": "upright"},
            {"t": 6.0, "box": [190, 60, 350, 440], "posture": "striking"},
            {"t": 6.6, "box": [190, 60, 350, 440], "posture": "upright"},
            {"t": 12.0, "box": [120, 60, 280, 440], "posture": "upright"}]},
        {"track_id": 2, "label": "person", "color": (200, 160, 60), "keyframes": [
            {"t": 0.0, "box": [380, 300, 470, 420], "posture": "infant"},
            {"t": 12.0, "box": [380, 300, 470, 420], "posture": "infant"}]},
    ]
    path = out / "nursery_caregiver_baby.mp4"
    _write_video(path, 12, tracks, background=(50, 45, 60), props=[((360, 280), (500, 440), (110, 90, 120))])
    fixture = {"location": "NURSERY", "tracks": tracks,
               "audio": {"segments": [], "acoustic": [{"start_s": 4.0, "end_s": 11.0, "label": "baby_crying", "confidence": 0.81}]},
               "vlm": {"support": {"aggressive_interaction": 0.7},
                       "description": {"aggressive_interaction": "An adult leans over a crib and makes repeated fast downward arm movements toward the infant; the infant's limbs are flailing. This is not consistent with gentle care."}},
               "critic": {"verdict": {"aggressive_interaction": "supported"},
                          "alternative": {"aggressive_interaction": "Vigorous but benign actions (patting to burp, changing) could look similar in low-resolution frames; the persistence of crying and the speed of motion argue against it."}}}
    path.with_suffix(".mp4.fixture.json").write_text(json.dumps(fixture, indent=1))
    return path

def scenario_spanish_voicemail(out: Path) -> Path:
    path = out / "spanish_threat_voicemail.wav"
    _write_wav(path, 14.0)
    fixture = {"location": "PHONE_LINE", "duration_s": 14.0, "audio": {"segments": [
        {"start_s": 0.5, "end_s": 6.0, "text": "Escúchame bien, sé dónde vives y sé a qué hora sales.", "language": "es",
         "translation_en": "Listen carefully, I know where you live and I know what time you leave.", "confidence": 0.85},
        {"start_s": 6.5, "end_s": 13.0, "text": "Si no me pagas mañana te voy a matar, a ti y a tu familia. Es tu última advertencia.", "language": "es",
         "translation_en": "If you don't pay me tomorrow I am going to kill you, you and your family. This is your last warning.", "confidence": 0.86}],
        "acoustic": [], "risk": {"threat": 0.93, "hate": 0.1, "toxicity": 0.9, "model": "fixture_detoxify_multilingual"}},
        "critic": {"verdict": {"threatening_speech": "supported"},
                   "alternative": {"threatening_speech": "Could be a quotation or a movie line; but the direct address, deadline and target reference make a genuine threat far more likely."}}}
    path.with_suffix(".wav.fixture.json").write_text(json.dumps(fixture, indent=1))
    return path

def scenario_knife_in_car(out: Path) -> Path:
    tracks = [
        {"track_id": 1, "label": "person", "keyframes": [
            {"t": 0.0, "box": [250, 80, 450, 470], "posture": "seated"}, {"t": 16.0, "box": [250, 80, 450, 470], "posture": "seated"}]},
        {"track_id": 2, "label": "person", "keyframes": [
            {"t": 0.0, "box": [40, 100, 230, 470], "posture": "seated"}, {"t": 16.0, "box": [40, 100, 230, 470], "posture": "seated"}]},
        {"track_id": 9, "label": "gun", "source": "weapon_detector", "confidence": 0.83, "holder": 1, "color": (0, 0, 255), "keyframes": [
            {"t": 5.0, "box": [300, 330, 350, 380]}, {"t": 12.0, "box": [305, 335, 355, 385]}]},
    ]
    path = out / "car_gun_in_pocket.mp4"
    _write_video(path, 16, tracks, background=(45, 40, 35), props=[((0, 300), (640, 480), (70, 60, 55))])
    fixture = {"location": "CAR_CABIN", "seated_context": True, "tracks": tracks, "audio": {"segments": [], "acoustic": []},
               "vlm": {"support": {"weapon_visible": 0.65},
                       "description": {"weapon_visible": "The grip of a black handgun protrudes from the right jacket pocket of the passenger on the right; the passenger's hand rests on it. It does not look like a phone."}},
               "critic": {"verdict": {"weapon_visible": "supported"},
                          "alternative": {"weapon_visible": "A phone or a toy replica in a pocket can produce a similar silhouette; the detector's 83% confidence and 100% persistence over 7 seconds reduce but do not remove that possibility."}}}
    path.with_suffix(".mp4.fixture.json").write_text(json.dumps(fixture, indent=1))
    return path

def scenario_elder_fall(out: Path) -> Path:
    tracks = [{"track_id": 1, "label": "person", "keyframes": [
        {"t": 0.0, "box": [120, 60, 260, 440], "posture": "upright"},
        {"t": 4.0, "box": [260, 60, 400, 440], "posture": "upright"},
        {"t": 4.6, "box": [250, 200, 420, 450], "posture": "crouched"},
        {"t": 5.2, "box": [200, 330, 520, 450], "posture": "lying"},
        {"t": 18.0, "box": [200, 330, 520, 450], "posture": "lying"}]}]
    path = out / "elder_fall_livingroom.mp4"
    _write_video(path, 18, tracks, background=(40, 48, 44))
    fixture = {"location": "ROOM_01", "tracks": tracks, "audio": {"segments": [], "acoustic": [{"start_s": 5.0, "end_s": 6.0, "label": "scream", "confidence": 0.55}]},
               "vlm": {"support": {"fall": 0.7}, "description": {"fall": "An elderly person walking with a cane loses balance, drops to the floor and remains motionless on their side."}},
               "critic": {"verdict": {"fall": "supported"}, "alternative": {"fall": "Deliberately lying down to exercise or rest would show a controlled descent; here the descent is abrupt (0.9 body-heights/s) and the person stays still."}}}
    path.with_suffix(".mp4.fixture.json").write_text(json.dumps(fixture, indent=1))
    return path

def scenario_normal_room(out: Path) -> Path:
    tracks = [{"track_id": 1, "label": "person", "keyframes": [
        {"t": 0.0, "box": [100, 60, 240, 440], "posture": "upright"}, {"t": 10.0, "box": [400, 60, 540, 440], "posture": "upright"}]}]
    path = out / "normal_room_walk.mp4"
    _write_video(path, 10, tracks)
    fixture = {"location": "ROOM_01", "tracks": tracks, "audio": {"segments": [], "acoustic": []}}
    path.with_suffix(".mp4.fixture.json").write_text(json.dumps(fixture, indent=1))
    return path

ALL = {"car_pregnant": scenario_car_pregnant, "nursery_abuse": scenario_nursery_abuse,
       "spanish_voicemail": scenario_spanish_voicemail, "knife_in_car": scenario_knife_in_car,
       "elder_fall": scenario_elder_fall, "normal_room": scenario_normal_room}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/.fixtures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in ALL.items():
        print(name, "->", fn(out))

if __name__ == "__main__":
    main()
