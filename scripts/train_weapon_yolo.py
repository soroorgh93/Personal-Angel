"""Fine-tune the visible-weapon detector on public data (runs on any CUDA GPU).

Data (all public, direct downloads, no key):
  * PranomVignesh/HandGuns (HF, MIT)          8,968 images already in YOLO layout  -> --fetch handguns
  * fcakyon/gun-object-detection (HF, CC BY 4.0) 4,666 images, COCO json           -> --fetch coco-guns
  * Simuletic/cctv-weapon-dataset (HF, CC BY 4.0) 141 CCTV images                   -> --fetch cctv
  * OD-WeaponDetection (Pascal-VOC XML; pistol/knife + hard negatives phone/wallet) -> --prepare <dir>
Init: cosgun99/gun-knife-yolo11n (MIT) so a short schedule already gives a strong model.

  python scripts/train_weapon_yolo.py --fetch handguns --out data/weapons_yolo
  python scripts/train_weapon_yolo.py --train data/weapons_yolo/dataset.yaml --epochs 30 --imgsz 960 --device 0

The resulting best.pt is copied to models/gun-knife-yolo11n.pt (the profile's weapon_model).
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["gun", "knife"]
ALIASES = {"pistol": "gun", "handgun": "gun", "gun": "gun", "rifle": "gun", "knife": "knife", "cuchillo": "knife"}

def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    tree = ET.parse(xml_path)
    lines = []
    for obj in tree.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        cls = ALIASES.get(name)
        if cls is None:
            continue
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        cx, cy, w, h = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h
        lines.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines

def prepare(src: Path, out: Path, val_fraction: float = 0.15, seed: int = 0) -> Path:
    from PIL import Image

    images = [p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.Random(seed).shuffle(images)
    n_val = int(len(images) * val_fraction)
    for split, items in (("val", images[:n_val]), ("train", images[n_val:])):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in items:
            xml = img.with_suffix(".xml")
            if not xml.exists():
                candidates = list(img.parent.parent.rglob(img.stem + ".xml"))
                xml = candidates[0] if candidates else None
            with Image.open(img) as im:
                w, h = im.size
            lines = voc_to_yolo(xml, w, h) if xml and xml.exists() else []
            dest = out / "images" / split / f"{img.parent.name}_{img.name}"
            shutil.copy(img, dest)
            (out / "labels" / split / (dest.stem + ".txt")).write_text("\n".join(lines))
    yaml = out / "dataset.yaml"
    yaml.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: gun\n  1: knife\n")
    print(f"prepared {len(images)} images ({n_val} val) → {yaml}")
    return yaml

def train(dataset_yaml: Path, epochs: int, imgsz: int, device: str, init: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(init)
    results = model.train(data=str(dataset_yaml), epochs=epochs, imgsz=imgsz, device=device, batch=16, workers=4,
                          project=str(ROOT / "runs" / "train"), name="weapon_yolo11n", exist_ok=True,
                          mosaic=1.0, mixup=0.1, close_mosaic=5, patience=10, pretrained=True)
    best = Path(results.save_dir) / "weights" / "best.pt"
    metrics = model.val(data=str(dataset_yaml), imgsz=imgsz, device=device)
    print("mAP50:", float(metrics.box.map50), "mAP50-95:", float(metrics.box.map))
    (ROOT / "models").mkdir(exist_ok=True)
    shutil.copy(best, ROOT / "models" / "gun-knife-yolo11n.pt")
    card = ROOT / "models" / "gun-knife-yolo11n.MODEL_CARD.md"
    card.write_text(f"# Visible-weapon detector\n\n- init: {init}\n- data: {dataset_yaml}\n- epochs: {epochs}, imgsz {imgsz}\n"
                    f"- mAP50 {float(metrics.box.map50):.3f}, mAP50-95 {float(metrics.box.map):.3f}\n"
                    "- classes: gun, knife (visible parts only; concealed objects are out of scope)\n"
                    "- hard negatives: phones, wallets, cards from the Sohas subset\n")
    print("saved", ROOT / "models" / "gun-knife-yolo11n.pt")
    return best

def fetch(which: str, out: Path) -> Path:
    """Download a public weapon dataset from Hugging Face and write dataset.yaml (YOLO layout)."""
    from huggingface_hub import snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    if which == "handguns":

        path = Path(snapshot_download("PranomVignesh/HandGuns", repo_type="dataset"))
        for split in ("train", "valid", "test"):
            for sub in ("images", "labels"):
                src = path / split / sub
                if src.exists():
                    dst = out / split / sub
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in src.iterdir():
                        shutil.copy(f, dst / f.name)
        (out / "dataset.yaml").write_text(f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\nnames: {{0: gun}}\n")
    elif which in {"coco-guns", "cctv"}:
        repo = "fcakyon/gun-object-detection" if which == "coco-guns" else "Simuletic/cctv-weapon-dataset"
        path = Path(snapshot_download(repo, repo_type="dataset"))
        import json
        import zipfile

        for z in path.rglob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out / z.stem)

        for ann in list(out.rglob("*.json")):
            try:
                data = json.loads(ann.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "images" not in data or "annotations" not in data:
                continue
            cats = {c["id"]: c["name"].lower() for c in data.get("categories", [])}
            imgs = {i["id"]: i for i in data["images"]}
            split = "valid" if "val" in ann.stem.lower() or "valid" in str(ann.parent).lower() else "train"
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "labels").mkdir(parents=True, exist_ok=True)
            per_img: dict[int, list[str]] = {}
            for a in data["annotations"]:
                name = cats.get(a["category_id"], "")
                cls = ALIASES.get(name, "gun" if "gun" in name or "pistol" in name else ("knife" if "knife" in name else None))
                if cls is None:
                    continue
                im = imgs[a["image_id"]]
                x, y, w, h = a["bbox"]
                per_img.setdefault(a["image_id"], []).append(
                    f"{CLASSES.index(cls)} {(x + w / 2) / im['width']:.6f} {(y + h / 2) / im['height']:.6f} {w / im['width']:.6f} {h / im['height']:.6f}")
            for img_id, lines in per_img.items():
                im = imgs[img_id]
                src = next(ann.parent.rglob(Path(im["file_name"]).name), None)
                if src is None:
                    continue
                shutil.copy(src, out / split / "images" / src.name)
                (out / split / "labels" / (src.stem + ".txt")).write_text("\n".join(lines))
        (out / "dataset.yaml").write_text(f"path: {out.resolve()}\ntrain: train/images\nval: valid/images\nnames: {{0: gun, 1: knife}}\n")
    else:
        raise SystemExit(f"unknown --fetch {which}")
    print("dataset ready:", out / "dataset.yaml")
    return out / "dataset.yaml"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", choices=["handguns", "coco-guns", "cctv"], help="download a public weapon dataset from HF")
    parser.add_argument("--prepare", type=Path, help="source folder with images + VOC xml")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "weapons_yolo")
    parser.add_argument("--train", type=Path, help="dataset.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--init", default=str(ROOT / "models" / "gun-knife-yolo11n.pt"))
    parser.add_argument("--export-trt", action="store_true", help="export a TensorRT FP16 engine")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.fetch, args.out)
    if args.prepare:
        prepare(args.prepare, args.out)
    if args.train:
        best = train(args.train, args.epochs, args.imgsz, args.device, args.init)
        if args.export_trt:
            from ultralytics import YOLO

            YOLO(str(best)).export(format="engine", half=True, imgsz=args.imgsz, device=args.device)

if __name__ == "__main__":
    main()
