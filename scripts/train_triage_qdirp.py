"""Train QDiRP-CompassNet (S. Ghandali's NPU-friendly CNN) as the Stage-1 frame
triage model on frames from public clips, following the recipe of the
technical note (RandAugment 2/7, MixUp 0.2 / CutMix 1.0, label smoothing 0.1,
random erasing 0.15, AdamW wd 0.03, cosine LR peak 2.5e-3 with 5 warm-up
epochs, EMA, aux heads + KL consistency, drop-path 0.08). Then fuses the
mixers, re-heads the classifier and exports a fixed-shape ONNX.

  python scripts/train_triage_qdirp.py --check                       # param count + fuse equivalence
  python scripts/train_triage_qdirp.py --build-frames data/downloads --out data/triage_frames
  python scripts/train_triage_qdirp.py --train data/triage_frames --epochs 60 --device 0

Frame labels come from the clip folder/file name: 'fall'→person_down (frames after
the fall onset), 'adl'/'normal'→normal, 'gun'/'knife'/'weapon'→weapon_visible,
'distress'/'slump'→distress. Keep a held-out test folder that is touched once.
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["normal", "person_down", "weapon_visible", "distress"]

def label_for(name: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ("gun", "knife", "weapon", "pistol")):
        return "weapon_visible"
    if any(k in n for k in ("distress", "slump", "faint")):
        return "distress"
    if "fall" in n:
        return "person_down"
    if any(k in n for k in ("adl", "normal", "walk")):
        return "normal"
    return None

def build_frames(src: Path, out: Path, fps: float = 2.0, fall_onset_fraction: float = 0.45) -> None:
    """Sample frames from clips; for 'fall' clips only the last part is labelled person_down."""
    import cv2

    count = {k: 0 for k in LABELS}
    for video in sorted(src.rglob("*.mp4")):
        label = label_for(video.stem) or label_for(video.parent.name)
        if label is None:
            continue
        cap = cv2.VideoCapture(str(video))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 25
        stride = max(1, int(vfps / fps))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frame_label = label
                if label == "person_down" and i < fall_onset_fraction * n:
                    frame_label = "normal"
                dest = out / frame_label / f"{video.stem}_{i:06d}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                h, w = frame.shape[:2]
                scale = 288 / min(h, w)
                cv2.imwrite(str(dest), cv2.resize(frame, (int(w * scale), int(h * scale))), [cv2.IMWRITE_JPEG_QUALITY, 90])
                count[frame_label] += 1
            i += 1
        cap.release()
    for img in sorted(src.rglob("*.jpg")) + sorted(src.rglob("*.png")):
        label = label_for(img.parent.name) or label_for(img.stem)
        if label:
            dest = out / label / img.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dest)
            count[label] += 1
    print("frames per class:", count)

def check() -> None:
    import torch

    from personal_angel.perception.qdirp import QDiRPCompassNet, count_parameters

    model = QDiRPCompassNet(num_classes=20)
    print(f"training-form parameters: {count_parameters(model):,} (note reports 4,382,108 incl. aux heads)")
    model.eval()
    x = torch.randn(2, 3, 240, 240)
    with torch.no_grad():
        y0 = model(x)
        model.fuse()
        y1 = model(x)
    print(f"deploy-form parameters: {count_parameters(model):,}; max |Δ| after fusion+re-head = {(y0 - y1).abs().max().item():.2e}")

def train(root: Path, epochs: int, device: str, out: Path, batch: int = 16, accum: int = 4) -> None:
    import torch
    import torch.nn.functional as F
    from torchvision import datasets, transforms

    from personal_angel.perception.qdirp import QDiRPCompassNet, qdirp_loss

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([transforms.RandomResizedCrop(240, scale=(0.55, 1.0), ratio=(3 / 4, 4 / 3)),
                                   transforms.RandomHorizontalFlip(), transforms.RandAugment(2, 7), transforms.ToTensor(), norm,
                                   transforms.RandomErasing(p=0.15)])
    val_tf = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(240),
                                 transforms.ToTensor(), norm])
    full = datasets.ImageFolder(str(root), transform=train_tf)
    classes = full.classes
    idx = np.arange(len(full))
    rng = np.random.default_rng(25520)
    rng.shuffle(idx)
    n_val = max(1, int(0.15 * len(idx)))
    val_ds = torch.utils.data.Subset(datasets.ImageFolder(str(root), transform=val_tf), idx[:n_val].tolist())
    train_ds = torch.utils.data.Subset(full, idx[n_val:].tolist())
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=4, drop_last=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=4)
    model = QDiRPCompassNet(num_classes=len(classes)).to(device)
    ema = QDiRPCompassNet(num_classes=len(classes)).to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, betas=(0.9, 0.999), weight_decay=0.03)
    steps_per_epoch = len(train_dl) // accum
    total = epochs * steps_per_epoch
    warm = 5 * steps_per_epoch
    lr_at = lambda u: 2.5e-3 * u / warm if u < warm else 1e-5 + (2.5e-3 - 1e-5) * 0.5 * (1 + math.cos(math.pi * (u - warm) / max(total - warm, 1)))
    scaler = torch.amp.GradScaler(enabled=device != "cpu")
    update = 0
    best = 0.0
    n_cls = len(classes)
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            target = F.one_hot(y, n_cls).float() * 0.9 + 0.1 / n_cls
            if rng.random() < 0.8:
                lam = float(rng.beta(0.2, 0.2)) if rng.random() < 0.5 else float(rng.beta(1.0, 1.0))
                perm = torch.randperm(x.size(0), device=device)
                if rng.random() < 0.5:
                    x = lam * x + (1 - lam) * x[perm]
                else:
                    h, w = x.shape[2:]
                    rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
                    cy, cx = int(rng.integers(h)), int(rng.integers(w))
                    y1, y2, x1, x2 = max(cy - rh // 2, 0), min(cy + rh // 2, h), max(cx - rw // 2, 0), min(cx + rw // 2, w)
                    x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
                    lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
                target = lam * target + (1 - lam) * target[perm]
            with torch.autocast(device_type="cuda" if device != "cpu" else "cpu", enabled=device != "cpu"):
                loss = qdirp_loss(model(x), target) / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(update)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                update += 1
                d = 0.9998 * (1 - math.exp(-update / 2000))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
        ema.eval()
        correct = total_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = ema(x.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum()); total_n += len(y)
        acc = correct / max(total_n, 1)
        print(f"epoch {epoch + 1}/{epochs} loss {loss.item() * accum:.3f} EMA val acc {acc:.4f}")
        if acc >= best:
            best = acc
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": ema.state_dict(), "classes": classes, "val_acc": acc}, out)
    print(f"best EMA val acc {best:.4f} → {out}")
    export_onnx(out, out.with_suffix(".onnx"), len(classes))

def export_onnx(checkpoint: Path, onnx_path: Path, n_cls: int) -> None:
    import torch

    from personal_angel.perception.qdirp import QDiRPCompassNet

    model = QDiRPCompassNet(num_classes=n_cls, aux=False)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"], strict=False)
    model.fuse()
    torch.onnx.export(model, torch.zeros(1, 3, 240, 240), str(onnx_path), input_names=["input"], output_names=["scores"], opset_version=18, dynamo=False)
    print("exported fused ONNX →", onnx_path, "(convs, BN, ReLU, add, GAP only — no concat/flatten/gemm)")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--build-frames", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "triage_frames")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "triage_qdirp.pt")
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(ROOT))
    if args.check:
        check()
    if args.build_frames:
        build_frames(args.build_frames, args.out)
    if args.train:
        train(args.train, args.epochs, args.device, args.checkpoint)

if __name__ == "__main__":
    main()
