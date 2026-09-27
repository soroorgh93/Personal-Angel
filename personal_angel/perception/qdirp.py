"""QDiRP-CompassNet — re-implementation of S. Ghandali's technical note
(21 Sep 2026): re-parameterizable directional depthwise CNN with fixed
Compass neighbour aggregation and a pyramid head. Used here as the cheap
Stage-1 frame-triage classifier (normal / person_down / weapon_visible /
distress) inside the adaptive perception cascade, trained on a GPU.

Architecture (input 240×240×3):
  stem 3×3 s2 → 120×120×32
  stage1: 1×1 s1 → 48ch, 2 QDiRP blocks (e=2, k=3,5)
  stage2: 3×3 s2 → 80ch, 3 blocks (e=3, k=5,3,5)
  stage3: 3×3 s2 → 144ch, 4 blocks (e=3, k=3,5,3,5)
  stage4: 3×3 s2 → 224ch, 4 blocks (e=3, k=5,3,5,3) + Compass
  stage5: 3×3 s2 → 320ch, 2 blocks (e=2, k=5,3) + Compass
  head: 1×1→160 (+BN+ReLU) on stages 3/4/5, GAP, concat 480, dropout .2, Linear→classes
Training form uses 5 parallel depthwise paths (k×k, 1×k, k×1, 1×1, identity)
sharing one BatchNorm; `fuse()` merges them into a single k×k kernel and
rewrites the head as three 1×1 convs + add (NPU-friendly, no concat/flatten).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGES = [
    (48, 1, 1, 2, 2, [3, 5], False),
    (80, 3, 2, 3, 3, [5, 3, 5], False),
    (144, 3, 2, 4, 3, [3, 5, 3, 5], False),
    (224, 3, 2, 4, 3, [5, 3, 5, 3], True),
    (320, 3, 2, 2, 2, [5, 3], True),
]

def _round8(v: float) -> int:
    return int((v + 7) // 8 * 8)

class DirectionalMixer(nn.Module):
    """Five depthwise paths → sum → shared BN (training form); fused k×k depthwise (deploy form)."""

    def __init__(self, ch: int, k: int) -> None:
        super().__init__()
        self.ch, self.k = ch, k
        p = k // 2
        self.square = nn.Conv2d(ch, ch, k, padding=p, groups=ch, bias=False)
        self.horizontal = nn.Conv2d(ch, ch, (1, k), padding=(0, p), groups=ch, bias=False)
        self.vertical = nn.Conv2d(ch, ch, (k, 1), padding=(p, 0), groups=ch, bias=False)
        self.point = nn.Conv2d(ch, ch, 1, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.fused: nn.Conv2d | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused is not None:
            return self.bn(self.fused(x))
        return self.bn(self.square(x) + self.horizontal(x) + self.vertical(x) + self.point(x) + x)

    @torch.no_grad()
    def fuse(self) -> None:
        k, p = self.k, self.k // 2
        kernel = self.square.weight.clone()
        kernel[:, :, p:p + 1, :] += self.horizontal.weight
        kernel[:, :, :, p:p + 1] += self.vertical.weight
        kernel[:, 0, p, p] += self.point.weight[:, 0, 0, 0] + 1.0
        fused = nn.Conv2d(self.ch, self.ch, k, padding=p, groups=self.ch, bias=False)
        fused.weight.copy_(kernel)
        self.fused = fused
        for name in ("square", "horizontal", "vertical", "point"):
            delattr(self, name)

class QDiRPBlock(nn.Module):
    def __init__(self, ch: int, expansion: int, k: int, drop_path: float = 0.0) -> None:
        super().__init__()
        hidden = _round8(ch * expansion)
        self.expand = nn.Sequential(nn.Conv2d(ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True))
        self.mixer = DirectionalMixer(hidden, k)
        self.project = nn.Sequential(nn.Conv2d(hidden, ch, 1, bias=False), nn.BatchNorm2d(ch))
        nn.init.zeros_(self.project[1].weight)
        nn.init.zeros_(self.project[1].bias)
        self.drop_path = drop_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(F.relu(self.mixer(self.expand(x)), inplace=True))
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            y = y * keep / (1 - self.drop_path)
        return F.relu(x + y, inplace=True)

class CompassBlock(nn.Module):
    """Fixed axial (+) and diagonal (×) 3×3 averaging kernels, learned 1×1 projections, residual."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        axial = torch.zeros(ch, 1, 3, 3)
        axial[:, 0, 0, 1] = axial[:, 0, 2, 1] = axial[:, 0, 1, 0] = axial[:, 0, 1, 2] = 0.25
        diag = torch.zeros(ch, 1, 3, 3)
        diag[:, 0, 0, 0] = diag[:, 0, 0, 2] = diag[:, 0, 2, 0] = diag[:, 0, 2, 2] = 0.25

        self.axial = nn.Parameter(axial, requires_grad=False)
        self.diag = nn.Parameter(diag, requires_grad=False)
        self.p_axial = nn.Conv2d(ch, ch, 1, bias=False)
        self.p_diag = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)
        self.ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.conv2d(x, self.axial, padding=1, groups=self.ch)
        d = F.conv2d(x, self.diag, padding=1, groups=self.ch)
        return F.relu(x + self.bn(self.p_axial(a) + self.p_diag(d)), inplace=True)

class QDiRPCompassNet(nn.Module):
    def __init__(self, num_classes: int = 20, aux: bool = True, drop_path_max: float = 0.08, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        stages = []
        in_ch = 32
        total_blocks = sum(s[3] for s in STAGES)
        block_idx = 0
        for out_ch, tk, stride, n_blocks, exp, ks, compass in STAGES:
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, tk, stride=stride, padding=tk // 2, bias=False),
                                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
            for b in range(n_blocks):
                dp = drop_path_max * block_idx / max(total_blocks - 1, 1)
                layers.append(QDiRPBlock(out_ch, exp, ks[b], dp))
                block_idx += 1
            if compass:
                layers.append(CompassBlock(out_ch))
            stages.append(nn.Sequential(*layers))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.head_proj = nn.ModuleList([nn.Sequential(nn.Conv2d(c, 160, 1, bias=False), nn.BatchNorm2d(160), nn.ReLU(inplace=True))
                                        for c in (144, 224, 320)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(480, num_classes)
        self.aux = aux
        if aux:
            self.aux3 = nn.Linear(144, num_classes)
            self.aux4 = nn.Linear(224, num_classes)
        self.deploy_heads: nn.ModuleList | None = None
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        feats = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= 2:
                feats.append(x)
        pooled = [F.adaptive_avg_pool2d(self.head_proj[i](f), 1) for i, f in enumerate(feats)]
        if self.deploy_heads is not None:
            logits = sum(h(v) for h, v in zip(self.deploy_heads, pooled)) + self.deploy_bias.view(1, -1, 1, 1)
            return logits.flatten(1)
        v = torch.cat([p.flatten(1) for p in pooled], dim=1)
        logits = self.classifier(self.dropout(v))
        if self.training and self.aux:
            a3 = self.aux3(F.adaptive_avg_pool2d(feats[0], 1).flatten(1))
            a4 = self.aux4(F.adaptive_avg_pool2d(feats[1], 1).flatten(1))
            return logits, a3, a4
        return logits

    @torch.no_grad()
    def fuse(self) -> "QDiRPCompassNet":
        """Deployment form: fused mixers + re-headed classifier (identical outputs)."""
        self.eval()
        for m in self.modules():
            if isinstance(m, DirectionalMixer) and m.fused is None:
                m.fuse()
        w, b = self.classifier.weight, self.classifier.bias
        heads = []
        for i in range(3):
            conv = nn.Conv2d(160, w.shape[0], 1, bias=False)
            conv.weight.copy_(w[:, i * 160:(i + 1) * 160].reshape(w.shape[0], 160, 1, 1))
            heads.append(conv)
        self.deploy_heads = nn.ModuleList(heads)
        self.register_buffer("deploy_bias", b.clone())
        self.aux = False
        return self

def qdirp_loss(outputs, target_soft: torch.Tensor, T: float = 2.0) -> torch.Tensor:
    """CE(main) + 0.2·CE(aux3) + 0.1·CE(aux4) + 0.08·T²·[KL(main‖aux3)+KL(main‖aux4)] with the main
    prediction as a fixed teacher (Section 3.3 of the note)."""
    logits, a3, a4 = outputs
    ce = lambda z: -(target_soft * F.log_softmax(z, dim=1)).sum(1).mean()
    teacher = F.softmax(logits.detach() / T, dim=1)
    kl = lambda z: F.kl_div(F.log_softmax(z / T, dim=1), teacher, reduction="batchmean")
    return ce(logits) + 0.2 * ce(a3) + 0.1 * ce(a4) + 0.08 * T * T * (kl(a3) + kl(a4))

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
