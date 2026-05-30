"""
model.py — MobileFaceNet backbone + ArcFace additive angular margin loss head.

Architecture:
  Input (3×112×112)
    -> Initial Conv (64 filters)
    -> Depthwise Separable Bottleneck blocks (inverted residuals)
    -> Linear Bottleneck -> 128-D L2-normalised embedding
    -> ArcFace head (training only — not exported to TFLite)

References:
  - MobileFaceNet: https://arxiv.org/abs/1804.07573
  - ArcFace:       https://arxiv.org/abs/1801.07698
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIM  = 128
ARCFACE_MARGIN = 0.5   # m — angular margin in radians
ARCFACE_SCALE  = 64.0  # s — feature scale factor


# ─── Building Blocks ──────────────────────────────────────────────────────────

class ConvBnAct(nn.Module):
    """Conv2d -> BatchNorm2d -> PReLU (or identity when act=False)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        """Initialise conv block."""
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)
        self.act  = nn.PReLU(out_ch) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise + Pointwise conv (no residual — used in early stages)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """Initialise depthwise separable block."""
        super().__init__()
        self.dw = ConvBnAct(in_ch, in_ch,  kernel=3, stride=stride, groups=in_ch)
        self.pw = ConvBnAct(in_ch, out_ch, kernel=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (MobileNetV2 style) with optional residual skip.
    Expansion -> Depthwise -> Linear Pointwise (no activation on output).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: int = 2,
    ) -> None:
        """Initialise inverted residual block."""
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        self.layers = nn.Sequential(
            # Expand
            ConvBnAct(in_ch,  hidden, kernel=1, padding=0),
            # Depthwise
            ConvBnAct(hidden, hidden, kernel=3, stride=stride, groups=hidden),
            # Project (linear — no activation)
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual add."""
        out = self.layers(x)
        return x + out if self.use_residual else out


def _make_bottleneck_stage(
    in_ch: int,
    out_ch: int,
    num_blocks: int,
    stride: int,
    expand_ratio: int = 2,
) -> nn.Sequential:
    """Stack InvertedResidual blocks; only the first block uses given stride."""
    blocks = [InvertedResidual(in_ch, out_ch, stride=stride, expand_ratio=expand_ratio)]
    for _ in range(1, num_blocks):
        blocks.append(InvertedResidual(out_ch, out_ch, stride=1, expand_ratio=expand_ratio))
    return nn.Sequential(*blocks)


# ─── MobileFaceNet Backbone ───────────────────────────────────────────────────

class MobileFaceNet(nn.Module):
    """
    MobileFaceNet: lightweight CNN producing 128-D L2-normalised face embeddings.

    Input:  (B, 3, 112, 112)
    Output: (B, 128)  — L2-normalised embedding vector
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        """Build network layers."""
        super().__init__()

        # Stage 0 — initial conv
        self.conv1 = ConvBnAct(3, 64, kernel=3, stride=2, padding=1)   # -> 64×56×56

        # Stage 1 — single DW-Sep block
        self.dw1   = DepthwiseSeparableBlock(64, 64, stride=1)          # -> 64×56×56

        # Stage 2–5 — Inverted Residual bottleneck stages
        # (out_ch, num_blocks, stride, expand_ratio)
        stage_cfg: List[Tuple[int, int, int, int]] = [
            (64,  5, 2, 2),   # -> 64×28×28
            (128, 1, 2, 4),   # -> 128×14×14
            (128, 6, 1, 2),   # -> 128×14×14
            (128, 1, 2, 4),   # -> 128×7×7
            (128, 2, 1, 2),   # -> 128×7×7
        ]
        in_ch = 64
        stages = []
        for out_ch, n, s, e in stage_cfg:
            stages.append(_make_bottleneck_stage(in_ch, out_ch, n, s, e))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Stage 6 — 1×1 conv to increase depth before GDC
        self.conv2  = ConvBnAct(128, 512, kernel=1, stride=1, padding=0)  # -> 512×7×7

        # Global Depthwise Conv (GDC) — feature aggregation
        self.gdc    = ConvBnAct(512, 512, kernel=7, stride=1, padding=0, groups=512, act=False)
        # -> 512×1×1

        # Linear layer — project to embedding (no bias, no activation)
        self.flatten  = nn.Flatten()
        self.bn_gdc   = nn.BatchNorm1d(512, eps=1e-5, momentum=0.1)
        self.dropout  = nn.Dropout(p=0.4)
        self.fc       = nn.Linear(512, embedding_dim, bias=False)
        self.bn_emb   = nn.BatchNorm1d(embedding_dim, eps=1e-5, momentum=0.1, affine=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform init for conv layers, ones for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce L2-normalised 128-D embedding."""
        x = self.conv1(x)
        x = self.dw1(x)
        x = self.stages(x)
        x = self.conv2(x)
        x = self.gdc(x)
        x = self.flatten(x)
        x = self.bn_gdc(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_emb(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalise


# ─── ArcFace Head ─────────────────────────────────────────────────────────────

class ArcFaceHead(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) classification head.

    Given L2-normalised embeddings and class labels, computes logits with
    an angular margin penalty m applied to the target class angle θ:
        logit_target = s * cos(θ + m)
        logit_others = s * cos(θ)

    This forces tight intra-class and wide inter-class angular clustering.
    """

    def __init__(
        self,
        embedding_dim: int  = EMBEDDING_DIM,
        num_classes:   int  = 10000,
        margin:        float = ARCFACE_MARGIN,
        scale:         float = ARCFACE_SCALE,
        easy_margin:   bool  = False,
    ) -> None:
        """Initialise ArcFace parameters."""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes
        self.margin        = margin
        self.scale         = scale
        self.easy_margin   = easy_margin

        # Weight matrix W: (num_classes × embedding_dim) — L2-normalised at forward
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin trig constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th    = math.cos(math.pi - margin)   # cos(π - m)
        self.mm    = math.sin(math.pi - margin) * margin  # sin(π-m)*m

    def forward(
        self,
        embeddings: torch.Tensor,  # (B, D) — already L2-normalised
        labels:     torch.Tensor,  # (B,)   — ground-truth class indices
    ) -> torch.Tensor:
        """Compute ArcFace logits."""
        # Normalise weight matrix columns
        w_norm = F.normalize(self.weight, p=2, dim=1)   # (C, D)

        # cos θ = embeddings · W^T  (both L2-normed -> dot = cos)
        cos_theta = F.linear(embeddings, w_norm)         # (B, C)
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        # sin θ via trig identity: sin²θ + cos²θ = 1
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2) + 1e-8)

        # cos(θ + m) = cosθ·cosm − sinθ·sinm
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            # Safeguard: if θ > π−m, fall back to cosθ − mm
            cos_theta_m = torch.where(
                cos_theta > self.th, cos_theta_m, cos_theta - self.mm
            )

        # Build one-hot mask and apply margin only to the target class
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.scale
        return logits  # (B, C) — raw logits for CrossEntropyLoss


# ─── Combined Training Wrapper ────────────────────────────────────────────────

class ArcFaceMobileFaceNet(nn.Module):
    """
    Full training model: MobileFaceNet spine + ArcFace head.

    During inference / export, use only the `backbone` sub-module.
    """

    def __init__(
        self,
        num_classes:   int   = 10000,
        embedding_dim: int   = EMBEDDING_DIM,
        margin:        float = ARCFACE_MARGIN,
        scale:         float = ARCFACE_SCALE,
    ) -> None:
        """Initialise backbone + head."""
        super().__init__()
        self.backbone = MobileFaceNet(embedding_dim)
        self.head     = ArcFaceHead(embedding_dim, num_classes, margin, scale)

    def forward(
        self,
        images: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Training: pass labels -> returns ArcFace logits.
        Inference: labels=None -> returns L2-normed embeddings.
        """
        embeddings = self.backbone(images)
        if labels is not None:
            return self.head(embeddings, labels)
        return embeddings

    def get_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw embeddings (inference / export mode)."""
        return self.backbone(images)


# ─── Quick Sanity Check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ArcFaceMobileFaceNet(num_classes=500).to(device)
    dummy_imgs   = torch.randn(4, 3, 112, 112, device=device)
    dummy_labels = torch.randint(0, 500, (4,), device=device)

    # Training forward
    logits = model(dummy_imgs, dummy_labels)
    print(f"Logits shape:    {logits.shape}")      # (4, 500)

    # Embedding forward
    embs = model.get_embedding(dummy_imgs)
    print(f"Embedding shape: {embs.shape}")         # (4, 128)
    print(f"Embedding norms: {embs.norm(dim=1)}")   # should be ≈ 1.0

    # Parameter count
    n_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"Backbone params: {n_params / 1e6:.2f} M")
