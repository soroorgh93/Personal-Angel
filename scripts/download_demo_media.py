"""Fetch public demo/eval media with plain HTTP (no accounts):

  python scripts/download_demo_media.py --set urfd --n 6        # UR Fall Detection (CC BY-NC-SA 4.0)
  python scripts/download_demo_media.py --set gmncsa --n 6      # GMNCSA24-FO fall clips on HF (MIT)
  python scripts/download_demo_media.py --set fleurs-es         # Spanish speech samples (CC BY)
  python scripts/download_demo_media.py --set pexels --ids 8104918 8104919 9550236 29097485
  python scripts/download_demo_media.py --set weapons-subset    # OD-WeaponDetection HF mirror (CC BY 4.0), 1 part

Everything lands in data/downloads/<set>/ with a manifest.json (source, license, sha256).
Licenses are non-commercial for URFD; Pexels clips are free to use under the Pexels license.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "downloads"

URFD = "https://fenix.ur.edu.pl/~mkepski/ds/data/{kind}-{n:02d}-cam0.mp4"
GMNCSA = "https://huggingface.co/datasets/Voxel51/GMNCSA24-FO/resolve/main/data/{n:02d}.mp4"
PEXELS = "https://www.pexels.com/download/video/{id}/"

def fetch(url: str, dest: Path, headers: dict | None = None) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return {"url": url, "path": str(dest), "cached": True, "sha256": sha256(dest)}
    with requests.get(url, stream=True, timeout=120, headers=headers or {"User-Agent": "PersonalAngel/1.0"}) as r:
        r.raise_for_status()
        with open(dest, "wb") as handle:
            for chunk in r.iter_content(1 << 20):
                handle.write(chunk)
    print(f"  ✔ {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return {"url": url, "path": str(dest), "sha256": sha256(dest)}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True, choices=["urfd", "gmncsa", "fleurs-es", "pexels", "weapons-subset"])
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--ids", nargs="*", default=[])
    args = parser.parse_args()
    out = OUT / args.set
    manifest: dict = {"set": args.set, "files": []}
    if args.set == "urfd":
        manifest["license"] = "CC BY-NC-SA 4.0 — Kwolek & Kepski 2014; research/demo use only"
        for n in range(1, args.n + 1):
            manifest["files"].append(fetch(URFD.format(kind="fall", n=n), out / f"fall-{n:02d}-cam0.mp4"))
            manifest["files"].append(fetch(URFD.format(kind="adl", n=n), out / f"adl-{n:02d}-cam0.mp4"))
    elif args.set == "gmncsa":
        manifest["license"] = "MIT (Voxel51 mirror of GMDCSA24)"
        for n in range(1, args.n + 1):
            manifest["files"].append(fetch(GMNCSA.format(n=n), out / f"gmncsa_{n:02d}.mp4"))
    elif args.set == "fleurs-es":
        manifest["license"] = "CC BY 4.0 (google/fleurs es_419)"
        from datasets import load_dataset
        import soundfile as sf

        ds = load_dataset("google/fleurs", "es_419", split="test", streaming=True)
        out.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(ds):
            if i >= args.n:
                break
            path = out / f"fleurs_es_{i:02d}.wav"
            sf.write(str(path), row["audio"]["array"], row["audio"]["sampling_rate"])
            manifest["files"].append({"path": str(path), "transcript": row["transcription"], "sha256": sha256(path)})
            print(f"  ✔ {path.name}: {row['transcription'][:60]}")
    elif args.set == "pexels":
        manifest["license"] = "Pexels License (free to use); verify each clip page before publishing"
        for pid in args.ids or ["8104918", "8104919", "9550236", "29097485", "7644974", "9340841"]:
            try:
                manifest["files"].append(fetch(PEXELS.format(id=pid), out / f"pexels_{pid}.mp4", {"User-Agent": "Mozilla/5.0"}))
            except Exception as error:
                print(f"  ✖ pexels {pid}: {error}")
    elif args.set == "weapons-subset":
        manifest["license"] = "CC BY 4.0 (shravya11/weapon-detection-dataset mirror of OD-WeaponDetection)"
        from huggingface_hub import snapshot_download

        path = snapshot_download("shravya11/weapon-detection-dataset", repo_type="dataset",
                                 allow_patterns=["Pistol detection/part_1/*", "Knife_detection/part_1/*", "README.md", "License.md"],
                                 local_dir=str(out))
        manifest["files"].append({"path": path})
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("manifest:", out / "manifest.json")

if __name__ == "__main__":
    main()
