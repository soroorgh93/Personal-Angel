"""Download every pretrained model the real profiles use (run once per machine).

  python scripts/download_models.py            # all
  python scripts/download_models.py --only cv  # cv | audio | text | tts

Sources (verified 2026-09-22):
  * Ultralytics YOLO11n / YOLO11n-pose            (AGPL-3.0, auto-downloaded by ultralytics)
  * cosgun99/gun-knife-yolo11n  best.pt           (MIT; Gun/Knife, mAP50 0.964)
  * melihuzunoglu/human-fall-detection best.pt    (AGPL-3.0; Fallen/Sitting/Standing)
  * Systran/faster-whisper-{small,medium,large-v3} (MIT; cached by faster-whisper on first use)
  * unitary/multilingual-toxic-xlm-roberta (Detoxify multilingual, Apache-2.0) — `threat` head
  * insiktml/threat_detection_xmlRoberta_ES       (OpenRAIL; Spanish threat/non-threat)
  * laion/clap-htsat-unfused                       (Apache-2.0; zero-shot audio events)
  * hexgrad/Kokoro-82M                             (Apache-2.0; Spanish TTS for the demo voicemail)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

def hf_file(repo: str, filename: str, dest: Path, repo_type: str = "model") -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo, filename=filename, repo_type=repo_type)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, dest)
    print(f"  ✔ {repo}/{filename} → {dest}")
    return dest

def cv() -> None:
    print("[cv] Ultralytics base weights")
    from ultralytics import YOLO

    for name in ("yolo11n.pt", "yolo11n-pose.pt", "yolo11s.pt", "yolo11s-pose.pt", "yolo11m.pt", "yolo11m-pose.pt"):
        if (MODELS / name).exists():
            print(f"  ✔ {name} (cached)")
            continue
        YOLO(name)
        src = Path(name)
        if src.exists():
            MODELS.mkdir(exist_ok=True)
            shutil.move(str(src), MODELS / name)
        print(f"  ✔ {name}")
    print("[cv] weapon + fallen detectors from Hugging Face")
    hf_file("cosgun99/gun-knife-yolo11n", "best.pt", MODELS / "gun-knife-yolo11n.pt")
    hf_file("melihuzunoglu/human-fall-detection", "best.pt", MODELS / "human-fall-detection-yolo11.pt")

def audio(asr_size: str) -> None:
    print(f"[audio] faster-whisper {asr_size} (downloads to the HF cache)")
    from faster_whisper import WhisperModel

    WhisperModel(asr_size, device="cpu", compute_type="int8")
    print("  ✔ whisper ready")
    print("[audio] CLAP zero-shot audio classifier")
    from transformers import pipeline

    pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused")
    print("  ✔ CLAP ready")

def text() -> None:
    print("[text] Detoxify multilingual (threat/toxicity)")
    from detoxify import Detoxify

    Detoxify("multilingual")
    print("  ✔ detoxify ready")
    try:
        from transformers import pipeline

        pipeline("text-classification", model="insiktml/threat_detection_xmlRoberta_ES")
        print("  ✔ Spanish threat classifier ready")
    except Exception as error:
        print(f"  (optional) Spanish threat classifier skipped: {error}")

def scene() -> None:
    """CLIP ViT-B/32 for scene understanding (environment, age group, weapon second opinion). Saved
    under models/clip so the app never needs the network afterwards."""
    print("[scene] CLIP ViT-B/32 zero-shot (openai/clip-vit-base-patch32, ~600 MB)")
    from transformers import CLIPModel, CLIPProcessor

    dest = MODELS / "clip"
    if (dest / "config.json").exists():
        print(f"  ✔ already at {dest}")
        return
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    proc.save_pretrained(dest)
    print(f"  ✔ CLIP saved to {dest}")

def tts() -> None:
    print("[tts] Kokoro-82M (Spanish voices ef_dora / em_alex)")
    try:
        from kokoro import KPipeline

        KPipeline(lang_code="e")
        print("  ✔ kokoro ready")
    except Exception as error:
        print(f"  (optional) kokoro skipped: {error} — pip install kokoro>=0.9.2 soundfile")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["cv", "audio", "text", "scene", "tts"], default=None)
    parser.add_argument("--asr", default="small", help="faster-whisper size: tiny|base|small|medium|large-v3")
    args = parser.parse_args()
    steps = {"cv": cv, "audio": lambda: audio(args.asr), "text": text, "scene": scene, "tts": tts}
    for name, fn in steps.items():
        if args.only and args.only != name:
            continue
        try:
            fn()
        except Exception as error:
            print(f"  ✖ {name} failed: {type(error).__name__}: {error}")
    print("\nModels directory:", MODELS)

if __name__ == "__main__":
    main()
