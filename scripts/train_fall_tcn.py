"""Train the learned fall head: YOLO11n-pose keypoints → 32-frame windows
[x, y, conf, vx, vy] per joint (normalized by bbox) → small temporal CNN
(depthwise TCN + attention pooling) → fall / no-fall. Saves TorchScript for
`TemporalFallHead` (pose.py) and reports held-out metrics by *video*, never by frame.

  python scripts/download_demo_media.py --set urfd --n 30
  python scripts/train_fall_tcn.py --extract data/downloads/urfd --out data/fall_windows.npz
  python scripts/train_fall_tcn.py --train data/fall_windows.npz --epochs 40 --device 0
Clips whose filename contains 'fall' are positives; 'adl' negatives (URFD); GMNCSA24 uses its CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
T = 32

def extract(src: Path, out: Path, stride: int = 4) -> None:
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "models" / "yolo11n-pose.pt")) if (ROOT / "models" / "yolo11n-pose.pt").exists() else YOLO("yolo11n-pose.pt")
    X, y, groups = [], [], []
    for video in sorted(src.glob("*.mp4")):
        label = 1 if "fall" in video.stem.lower() else 0
        cap = cv2.VideoCapture(str(video))
        seq = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(frame, conf=0.3, verbose=False)[0]
            if r.keypoints is None or len(r.boxes) == 0:
                seq.append(np.zeros((17, 3), dtype=np.float32))
                continue
            i = int(np.argmax(r.boxes.conf.cpu().numpy()))
            kp = r.keypoints.data[i].cpu().numpy()
            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
            w, h = max(x2 - x1, 1), max(y2 - y1, 1)
            kp[:, 0] = (kp[:, 0] - x1) / w
            kp[:, 1] = (kp[:, 1] - y1) / h
            seq.append(kp.astype(np.float32))
        cap.release()
        seq = np.stack(seq) if seq else np.zeros((0, 17, 3), dtype=np.float32)
        for start in range(0, max(len(seq) - T, 0) + 1, stride):
            win = seq[start:start + T]
            if len(win) < T:
                continue
            vel = np.diff(win[:, :, :2], axis=0, prepend=win[:1, :, :2])
            X.append(np.concatenate([win, vel], axis=2))
            y.append(label)
            groups.append(video.stem)
        print(f"  {video.name}: {len(seq)} frames → label {label}")
    np.savez_compressed(out, X=np.stack(X), y=np.array(y), groups=np.array(groups))
    print(f"saved {len(X)} windows → {out}")

def build_model(n_joints: int = 17, feats: int = 5, hidden: int = 96):
    import torch.nn as nn

    class FallTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = n_joints * feats
            self.net = nn.Sequential(
                nn.Conv1d(c, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2, groups=hidden), nn.Conv1d(hidden, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4, groups=hidden), nn.Conv1d(hidden, hidden, 1), nn.BatchNorm1d(hidden), nn.ReLU(),
            )
            self.attn = nn.Conv1d(hidden, 1, 1)
            self.fc = nn.Linear(hidden, 2)

        def forward(self, x):
            b, t, j, f = x.shape
            h = self.net(x.reshape(b, t, j * f).transpose(1, 2))
            w = self.attn(h).softmax(dim=-1)
            return self.fc((h * w).sum(-1))

    return FallTCN()

def train(npz: Path, epochs: int, device: str, out: Path) -> None:
    import torch
    import torch.nn.functional as F

    data = np.load(npz, allow_pickle=True)
    X, y, groups = data["X"], data["y"], data["groups"]
    uniq = sorted(set(groups))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    val_groups = set(uniq[: max(1, len(uniq) // 5)])
    val_mask = np.array([g in val_groups for g in groups])
    Xt, yt = torch.tensor(X[~val_mask]), torch.tensor(y[~val_mask])
    Xv, yv = torch.tensor(X[val_mask]), torch.tensor(y[val_mask])
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos_w = torch.tensor([1.0, float((yt == 0).sum() / max((yt == 1).sum(), 1))], device=device)
    best, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            xb = xb + 0.01 * torch.randn_like(xb)
            loss = F.cross_entropy(model(xb), yb, weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xv.to(device)).argmax(1).cpu()
        tp = int(((pred == 1) & (yv == 1)).sum()); fp = int(((pred == 1) & (yv == 0)).sum()); fn = int(((pred == 0) & (yv == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        if f1 >= best:
            best, best_state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch + 1}/{epochs} loss {loss.item():.3f} val P {prec:.2f} R {rec:.2f} F1 {f1:.2f} (videos held out: {len(val_groups)})")
    model.load_state_dict(best_state)
    model.eval().cpu()
    scripted = torch.jit.trace(model, torch.zeros(1, T, 17, 5))
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))
    print(f"saved TorchScript → {out} (best val F1 {best:.2f}); enable with pose.fall.temporal_checkpoint in the profile")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "fall_windows.npz")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / "fall_tcn.ts")
    args = parser.parse_args()
    if args.extract:
        extract(args.extract, args.out)
    if args.train:
        train(args.train, args.epochs, args.device, args.checkpoint)

if __name__ == "__main__":
    main()
