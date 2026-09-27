"""Fetch REAL public recordings for the demo library (data/demo) — no synthetic blobs.

Every item below is a direct, key-less download (Hugging Face `resolve` links, GitHub raw,
the UR Fall Detection server, Pexels' public download redirect). Clips are re-encoded to
H.264 720p (browser-playable), trimmed to --max-seconds, and listed in data/demo/library.json
with title, source and license so the UI can show attribution.

    python scripts/fetch_real_demo_clips.py --out data/demo            # everything (~250 MB)
    python scripts/fetch_real_demo_clips.py --only falls,weapons        # a subset
    python scripts/fetch_real_demo_clips.py --list                      # show the manifest

Licenses: UR Fall = CC BY-NC-SA 4.0 (Kwolek & Kepski 2014); ESC-50 = CC BY-NC; Surveillance
Fight Dataset = MIT; Pexels = Pexels License (free to use); HF Spaces samples = as published by
their authors (research/demo use). Keep the attribution when you show the clips.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PersonalAngel-demo-fetch/1.0"}

MANIFEST: list[dict] = [
    {"group": "falls", "file": "urfd_fall_01.mp4", "title": "Fall #01 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-01-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "urfd_fall_05.mp4", "title": "Fall #05 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-05-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "urfd_fall_12.mp4", "title": "Fall #12 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-12-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "urfd_fall_02.mp4", "title": "Fall #02 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-02-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "urfd_fall_03.mp4", "title": "Fall #03 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-03-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "urfd_fall_08.mp4", "title": "Fall #08 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-08-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "urfd_fall_20.mp4", "title": "Fall #20 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-20-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "urfd_fall_25.mp4", "title": "Fall #25 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-25-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "urfd_fall_30.mp4", "title": "Fall #30 — real fall at home (UR Fall Detection, cam0)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/fall-30-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "ALERT: fall -> asks 'are you okay?' -> silence -> 911 + notify owner (simulated)"},
    {"group": "falls", "file": "unidata_fall_sample_1.mp4", "title": "Fall at home — UniDataPro sample 1 (1080p, real)", "url": "https://huggingface.co/datasets/UniDataPro/fall-detection/resolve/main/sample_1.mp4", "source": "UniDataPro/fall-detection (HF)", "license": "CC BY-NC-ND", "expected": "ALERT: fall -> question -> 911 (simulated)"},
    {"group": "falls", "file": "unidata_fall_sample_2.mp4", "title": "Fall at home — UniDataPro sample 2 (1080p, real)", "url": "https://huggingface.co/datasets/UniDataPro/fall-detection/resolve/main/sample_2.mp4", "source": "UniDataPro/fall-detection (HF)", "license": "CC BY-NC-ND", "expected": "ALERT: fall -> question -> 911 (simulated)"},
    {"group": "falls", "file": "urfd_adl_01_normal.mp4", "title": "Normal activity #01 — daily activity, no fall (UR Fall ADL, negative control)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/adl-01-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "CLEAR: no incident, no VLM spent"},
    {"group": "falls", "file": "urfd_adl_05_normal.mp4", "title": "Normal activity #05 — daily activity, no fall (UR Fall ADL, negative control)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/adl-05-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "CLEAR: no incident, no VLM spent"},
    {"group": "falls", "file": "urfd_adl_10_normal.mp4", "title": "Normal activity #10 — daily activity, no fall (UR Fall ADL, negative control)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/adl-10-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "CLEAR: no incident, no VLM spent"},
    {"group": "falls", "file": "urfd_adl_20_normal.mp4", "title": "Normal activity #20 — daily activity, no fall (UR Fall ADL, negative control)", "url": "http://fenix.ur.edu.pl/~mkepski/ds/data/adl-20-cam0.mp4", "source": "UR Fall Detection Dataset (Kwolek & Kepski 2014)", "license": "CC BY-NC-SA 4.0", "expected": "CLEAR: no incident, no VLM spent"},
    {"group": "weapons", "file": "pexels_gun_in_car_8102787.mp4", "title": "Gun in a car — man with a handgun in the driver seat (Pexels)", "url": "https://www.pexels.com/download/video/8102787/", "source": "Pexels #8102787", "license": "Pexels License", "expected": "ALERT: weapon_visible -> no question -> 911 (police) + location + security (simulated)"},
    {"group": "weapons", "file": "pexels_pistol_indoors_6091316.mp4", "title": "Man aiming a pistol indoors (Pexels)", "url": "https://www.pexels.com/download/video/6091316/", "source": "Pexels #6091316", "license": "Pexels License", "expected": "ALERT: weapon_visible -> 911 (police) + location (simulated)"},
    {"group": "weapons", "file": "hf_knife_sample.mp4", "title": "Knife held in hand — detection sample", "url": "https://huggingface.co/spaces/Dricz/Weapon_Detection_YOLOv8-4/resolve/main/video/Knife.mp4", "source": "HF Space Dricz/Weapon_Detection_YOLOv8-4", "license": "as published", "expected": "ALERT: knife -> security + police (simulated)"},
    {"group": "weapons", "file": "hf_rifle_sample.mp4", "title": "Rifle — detection sample", "url": "https://huggingface.co/spaces/Dricz/Weapon_Detection_YOLOv8-4/resolve/main/video/ExampleRifle.mp4", "source": "HF Space Dricz/Weapon_Detection_YOLOv8-4", "license": "as published", "expected": "ALERT: weapon_visible"},
    {"group": "weapons", "file": "hf_with_guns_sample.mp4", "title": "People with guns — detection sample", "url": "https://huggingface.co/spaces/AbhishekShrimali/Weapon_Detection/resolve/main/sample/WithGuns.mp4", "source": "HF Space AbhishekShrimali/Weapon_Detection", "license": "as published", "expected": "ALERT: weapon_visible"},
    {"group": "weapons", "file": "hf_criminal_threatens.mp4", "title": "Armed threat — detection sample", "url": "https://huggingface.co/spaces/AbhishekShrimali/Weapon_Detection/resolve/main/sample/CriminalThreatens.mp4", "source": "HF Space AbhishekShrimali/Weapon_Detection", "license": "as published", "expected": "ALERT: weapon_visible + aggression"},
    {"group": "aggression", "file": "surv_fight_001.mp4", "title": "Fight #001 — real CCTV fight (Surveillance Camera Fight Dataset)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/fight/fi001.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "ALERT/WATCH: aggressive_interaction -> 911 (police) + security (simulated)"},
    {"group": "aggression", "file": "surv_fight_002.mp4", "title": "Fight #002 — real CCTV fight (Surveillance Camera Fight Dataset)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/fight/fi002.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "ALERT/WATCH: aggressive_interaction -> 911 (police) + security (simulated)"},
    {"group": "aggression", "file": "surv_fight_003.mp4", "title": "Fight #003 — real CCTV fight (Surveillance Camera Fight Dataset)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/fight/fi003.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "ALERT/WATCH: aggressive_interaction -> 911 (police) + security (simulated)"},
    {"group": "aggression", "file": "surv_fight_004.mp4", "title": "Fight #004 — real CCTV fight (Surveillance Camera Fight Dataset)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/fight/fi004.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "ALERT/WATCH: aggressive_interaction -> 911 (police) + security (simulated)"},
    {"group": "aggression", "file": "surv_fight_005.mp4", "title": "Fight #005 — real CCTV fight (Surveillance Camera Fight Dataset)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/fight/fi005.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "ALERT/WATCH: aggressive_interaction -> 911 (police) + security (simulated)"},
    {"group": "aggression", "file": "surv_fight_006.mp4", "title": "Fight #006 — real CCTV fight (Surveillance Camera Fight Dataset)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/fight/fi006.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "ALERT/WATCH: aggressive_interaction -> 911 (police) + security (simulated)"},
    {"group": "aggression", "file": "surv_fight_007.mp4", "title": "Fight #007 — real CCTV fight (Surveillance Camera Fight Dataset)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/fight/fi007.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "ALERT/WATCH: aggressive_interaction -> 911 (police) + security (simulated)"},
    {"group": "aggression", "file": "surv_fight_008.mp4", "title": "Fight #008 — real CCTV fight (Surveillance Camera Fight Dataset)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/fight/fi008.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "ALERT/WATCH: aggressive_interaction -> 911 (police) + security (simulated)"},
    {"group": "aggression", "file": "surv_nofight_001.mp4", "title": "No fight #001 — real CCTV, people moving normally (negative control)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/noFight/nofi001.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "CLEAR"},
    {"group": "aggression", "file": "surv_nofight_002.mp4", "title": "No fight #002 — real CCTV, people moving normally (negative control)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/noFight/nofi002.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "CLEAR"},
    {"group": "aggression", "file": "surv_nofight_003.mp4", "title": "No fight #003 — real CCTV, people moving normally (negative control)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/noFight/nofi003.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "CLEAR"},
    {"group": "aggression", "file": "surv_nofight_004.mp4", "title": "No fight #004 — real CCTV, people moving normally (negative control)", "url": "https://raw.githubusercontent.com/sayibet/fight-detection-surv-dataset/master/noFight/nofi004.mp4", "source": "sayibet/fight-detection-surv-dataset", "license": "MIT", "expected": "CLEAR"},
    {"group": "aggression", "file": "rwf2000_fight_val_0.avi", "title": "Fight — RWF-2000 real-world surveillance clip", "url": "https://huggingface.co/datasets/A1mal/RWF-2000-Dataset/resolve/main/data/RWF-2000%20Sliced/val/Fight/0Ow4cotKOuw_0.avi", "source": "RWF-2000 (Cheng et al.) via A1mal/RWF-2000-Dataset", "license": "research use", "expected": "ALERT/WATCH: aggressive_interaction"},
    {"group": "nursery", "file": "pexels_baby_in_crib_16418098.mp4", "title": "Baby lying in a crib, calm (Pexels; negative control)", "url": "https://www.pexels.com/download/video/16418098/", "source": "Pexels #16418098", "license": "Pexels License", "expected": "CLEAR (baby profiled, no distress)"},
    {"group": "nursery", "file": "pexels_baby_crying_6848985.mp4", "title": "Baby crying (Pexels; video has no audio track)", "url": "https://www.pexels.com/download/video/6848985/", "source": "Pexels #6848985", "license": "Pexels License", "expected": "WATCH: infant_distress from the vision model -> call/notify family (simulated)"},
    {"group": "nursery", "file": "esc50_baby_crying_1.wav", "title": "Baby crying — ESC-50 clip 1 (audio only)", "url": "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-187207-A-20.wav", "source": "ESC-50 (crying_baby)", "license": "CC BY-NC", "expected": "acoustic baby_crying -> notify caregiver"},
    {"group": "nursery", "file": "esc50_baby_crying_2.wav", "title": "Baby crying — ESC-50 clip 2 (audio only)", "url": "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/2-50665-A-20.wav", "source": "ESC-50 (crying_baby)", "license": "CC BY-NC", "expected": "acoustic baby_crying -> notify caregiver"},
    {"group": "vehicle", "file": "pexels_man_unwell_in_car_8641684.mp4", "title": "Car cabin — man sitting in a car, talking (Pexels; negative control)", "url": "https://www.pexels.com/download/video/8641684/", "source": "Pexels #8641684", "license": "Pexels License", "expected": "CLEAR (scene: vehicle cabin)"},
    {"group": "vehicle", "file": "pexels_couple_in_car_6788032.mp4", "title": "Car cabin — two people sitting in a car (Pexels; negative control)", "url": "https://www.pexels.com/download/video/6788032/", "source": "Pexels #6788032", "license": "Pexels License", "expected": "CLEAR (scene: vehicle cabin)"},

    {"group": "audio", "file": "song_en_same_boat_josh_woodward.wav", "title": "Song — 'Same Boat', Josh Woodward (English vocals, CC BY 4.0)",
     "url": "https://archive.org/download/JoshWoodward-TheShadeFromOurTrees/JoshWoodward-TheShadeFromOurTrees-04-SameBoat.mp3", "source": "Internet Archive / Josh Woodward", "license": "CC BY 4.0",
     "expected": "CLEAR: audio classified as music/singing; lyrics transcribed, no threat", "audio_seconds": 45},
    {"group": "audio", "file": "song_en_over_there_1917.wav", "title": "Song — 'Over There', Billy Murray 1917 (public domain 78 rpm)",
     "url": "https://archive.org/download/78_over-there_billy-murray-george-m-cohan_gbia0088690a/Over%20There%20-%20Billy%20Murray%20-%20George%20M.%20Cohan.mp3", "source": "Internet Archive (Great 78 Project)", "license": "public domain",
     "expected": "CLEAR: music/singing — war-song lyrics are not a threat directed at a person", "audio_seconds": 45},
    {"group": "audio", "file": "song_es_la_paloma_1903.wav", "title": "Song — 'La Paloma', Zélie de Lussan 1903 (Spanish vocals, public domain)",
     "url": "https://archive.org/download/78_la-paloma_zelie-de-lussan-yradier_gbia0077653b/La%20Paloma%20-%20Zelie%20de%20Lussan%20-%20Yradier.mp3", "source": "Internet Archive (Great 78 Project)", "license": "public domain",
     "expected": "CLEAR: music/singing in Spanish, translated lyrics, no threat", "audio_seconds": 45},
    {"group": "audio", "file": "song_es_cosas_tan_bellas.wav", "title": "Song — 'Cosas tan bellas', Los Rocianeros (Spanish vocals, CC BY 3.0)",
     "url": "https://archive.org/download/jamendo-244865/01-1242768-Los%20Rocianeros-cosas%20tan%20bellas.mp3", "source": "Internet Archive / Jamendo", "license": "CC BY 3.0",
     "expected": "CLEAR: music/singing", "audio_seconds": 45},
    {"group": "audio", "file": "instrumental_same_boat_novox.wav", "title": "Instrumental — 'Same Boat (NoVox)', no vocals (CC BY 4.0)",
     "url": "https://archive.org/download/JoshWoodward-TheShadeFromOurTrees-NoVox/JoshWoodward-SameBoat-NoVox.mp3", "source": "Internet Archive / Josh Woodward", "license": "CC BY 4.0",
     "expected": "CLEAR: music, no speech at all", "audio_seconds": 40},
    {"group": "audio", "file": "argument_acted_taming_of_the_shrew.wav", "title": "Argument (acted, public domain) — Petruchio vs Katherina quarrel, LibriVox full cast",
     "url": "https://archive.org/download/tamingoftheshrew_1009_librivox/tamingoftheshrew_2_shakespeare_64kb.mp3", "source": "LibriVox (public domain)", "license": "public domain",
     "expected": "raised voices + insults -> abusive_speech -> asks 'do you want me to report it?' (no threat to life -> no 911)", "audio_seconds": 75, "audio_offset": 640},
    {"group": "voicemail", "file": "hatecheck_spanish_test.csv", "title": "HateCheck-ES threat sentences (text, CC BY 4.0)", "url": "https://huggingface.co/datasets/Paul/hatecheck-spanish/resolve/main/test.csv", "source": "Paul/hatecheck-spanish (HF)", "license": "CC BY 4.0", "keep_raw": True},
]

VOICEMAIL_EN = ("Listen carefully. I know where you live and I know where your kids go to school. "
                "If you show up at work tomorrow, I will hurt you. This is not a joke.")

def download(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            if dest.stat().st_size > 2000:
                return True
        except Exception as error:
            print(f"    attempt {attempt + 1} failed: {error}")
            time.sleep(2)
    return False

def transcode_audio(src: Path, dest: Path, seconds: float, offset: float = 0.0) -> bool:
    if shutil.which("ffmpeg") is None:
        shutil.copy(src, dest)
        return True
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(offset), "-i", str(src), "-t", str(seconds), "-vn", "-ac", "1", "-ar", "16000", str(dest)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False).returncode == 0 and dest.exists()

def transcode(src: Path, dest: Path, max_seconds: float) -> bool:
    if shutil.which("ffmpeg") is None:
        shutil.copy(src, dest)
        return True
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(max_seconds),
           "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(dest)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return done.returncode == 0 and dest.exists()

def make_voicemails(out: Path, csv_path: Path | None) -> list[dict]:
    """Voice real threat sentences (HateCheck-ES, CC BY 4.0) with the local Kokoro TTS and give them a
    phone-line sound. Skipped (with a hint) when kokoro is not installed."""
    items: list[dict] = []
    try:
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline
    except Exception as error:
        print(f"  (voicemail synthesis skipped: {error}; pip install kokoro soundfile — espeak-ng is needed for Spanish)")
        return items
    spanish = ["Escúchame bien. Sé dónde vives y a qué hora sales. Si vuelves a hablar con la policía te voy a hacer daño. No es una broma."]
    if csv_path and csv_path.exists():
        import csv

        with open(csv_path, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)]
        threats = [r for r in rows if "threat" in str(r.get("functionality", "")).lower() and str(r.get("label_gold", "")).lower() == "hateful"]
        if threats:
            spanish = [" ".join(r["test_case"] for r in threats[:3]) + " No es una broma."] + spanish
    jobs = [("es", "ef_dora", spanish[0], "voicemail_spanish_threat.wav", "Voicemail — threatening message in Spanish (HateCheck-ES sentences, local TTS)"),
            ("a", "am_adam", VOICEMAIL_EN, "voicemail_english_threat.wav", "Voicemail — threatening message in English (local TTS)"),
            ("es", "em_alex", "Hola, soy Marcos del taller. Su coche ya está listo, puede pasar a recogerlo mañana por la mañana. Gracias.",
             "voicemail_spanish_benign.wav", "Voicemail — benign message in Spanish (negative control)")]
    for lang, voice, text, name, title in jobs:
        try:
            pipe = KPipeline(lang_code=lang)
            chunks = [a for _, _, a in pipe(text, voice=voice) if a is not None]
            audio = np.concatenate([np.asarray(a, dtype=np.float32) for a in chunks])

            from numpy.fft import irfft, rfft

            spec = rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1 / 24000)
            spec[(freqs < 300) | (freqs > 3400)] = 0
            audio = irfft(spec, n=len(audio)).astype(np.float32)
            audio = np.tanh(2.2 * audio) * 0.8
            sf.write(out / name, audio, 24000)
            items.append({"file": name, "title": title, "group": "voicemail", "source": "HateCheck-ES text (CC BY 4.0) + Kokoro-82M TTS",
                          "license": "CC BY 4.0 / Apache-2.0"})
            print(f"  + {name}")
        except Exception as error:
            print(f"  (voicemail {name} failed: {error})")
    return items

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "demo")
    parser.add_argument("--only", default="", help="comma-separated groups: falls,weapons,aggression,nursery,vehicle,audio,voicemail")
    parser.add_argument("--max-seconds", type=float, default=40.0)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    groups = {g.strip() for g in args.only.split(",") if g.strip()}
    if args.list:
        for m in MANIFEST:
            print(f"{m['group']:<11} {m['file']:<40} {m['source']} [{m['license']}]")
        return
    out: Path = args.out
    raw = out / ".raw"
    out.mkdir(parents=True, exist_ok=True)
    library: list[dict] = []
    ok = fail = 0
    csv_path = None
    for m in MANIFEST:
        if groups and m["group"] not in groups:
            continue
        dest = out / m["file"]
        print(f"[{m['group']}] {m['title']}")
        if dest.exists() and dest.stat().st_size > 2000:
            print("  already present")
        else:
            tmp = raw / m["file"]
            if not download(m["url"], tmp):
                print("  ✖ download failed (skipped)")
                fail += 1
                continue
            if m.get("audio_seconds"):
                if not transcode_audio(tmp, dest, float(m["audio_seconds"]), float(m.get("audio_offset", 0.0))):
                    shutil.copy(tmp, dest.with_suffix(Path(m["url"]).suffix.split("?")[0] or ".mp3"))
            elif m.get("keep_raw") or dest.suffix.lower() in {".wav", ".csv"}:
                shutil.copy(tmp, dest)
            elif dest.suffix.lower() == ".avi":
                mp4 = dest.with_suffix(".mp4")
                if transcode(tmp, mp4, args.max_seconds):
                    dest = mp4
                    m["file"] = mp4.name
                else:
                    shutil.copy(tmp, dest)
            elif not transcode(tmp, dest, args.max_seconds):
                shutil.copy(tmp, dest)
            print(f"  ✔ {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        ok += 1
        if m["file"].endswith(".csv"):
            csv_path = dest
            continue
        library.append({k: m[k] for k in ("file", "title", "group", "source", "license", "expected") if k in m})
    if not groups or "voicemail" in groups:
        print("[voicemail] multilingual voicemails + dubbed demo videos -> scripts/make_multilingual_voicemails.py")
        (out / "library.json").write_text(json.dumps({"items": library}, indent=2), encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_multilingual_voicemails.py"), "--out", str(out)], check=False)
        try:
            library = json.loads((out / "library.json").read_text(encoding="utf-8")).get("items", library)
        except Exception:
            pass
    (out / "library.json").write_text(json.dumps({"items": library, "generated": time.strftime("%Y-%m-%d %H:%M")}, indent=2), encoding="utf-8")
    shutil.rmtree(raw, ignore_errors=True)
    print(f"\ndone: {ok} fetched, {fail} failed -> {out}  (library.json written; open 'Example recordings' in the app)")
    if fail:
        print("Failed items are optional — re-run later or drop your own recording into the app.")

if __name__ == "__main__":
    sys.exit(main())
