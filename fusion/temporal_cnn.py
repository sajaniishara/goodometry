"""1D temporal CNN for sensor fusion — R3D-18-inspired but with no spatial dims.

Replaces the (T, H, W) volume of R3D-18 with a (T,) sensor sequence. Each
modality is first embedded per-timestep to a common channel space, the
embeddings are concatenated along the channel axis, then a stack of
BasicBlock1D residual blocks runs along time with progressive downsampling
(R3D-18 stage layout: 4 stages, each 2 blocks, channel doubling and stride-2
temporal downsampling between stages).

Inputs match fusion_v2:
    kin (B, T, kin_in)   — 31 (v2) or 35 (v3) D
    ins (B, T, ins_in)   — 10 D
    vo  (B, T, vo_in)    — 7 D, optional

Output: (B, 6) body-frame velocity.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BasicBlock1D(nn.Module):
    """R3D-18 BasicBlock, but 1D — Conv1d-BN-ReLU-Conv1d-BN + residual."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class FusionTCN(nn.Module):
    def __init__(
        self,
        kin_in: int = 31,
        ins_in: int = 10,
        vo_in:  int | None = None,        # None → kin+ins; set to 7 for fusion_v2-like 3-modality input
        kin_emb: int = 32,
        ins_emb: int = 16,
        vo_emb:  int = 16,
        stage_channels: tuple[int, int, int, int] = (64, 64, 128, 256),
        blocks_per_stage: int = 2,
        clip_len: int = 40,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_vo = vo_in is not None
        self.clip_len = clip_len

        # Per-modality per-timestep linear embedders.
        self.embed_kin = nn.Sequential(nn.Linear(kin_in, kin_emb), nn.GELU())
        self.embed_ins = nn.Sequential(nn.Linear(ins_in, ins_emb), nn.GELU())
        if self.use_vo:
            self.embed_vo = nn.Sequential(nn.Linear(vo_in, vo_emb), nn.GELU())

        in_ch = kin_emb + ins_emb + (vo_emb if self.use_vo else 0)

        # Stem: project channel-concat to first stage width with a small temporal context.
        c0, c1, c2, c3 = stage_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, c0, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(c0),
            nn.ReLU(inplace=True),
        )

        # Four stages — stage 1 keeps T, stages 2/3/4 halve T each.
        def make_stage(ic: int, oc: int, stride: int) -> nn.Sequential:
            blocks = [BasicBlock1D(ic, oc, stride=stride)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(BasicBlock1D(oc, oc, stride=1))
            return nn.Sequential(*blocks)

        self.stage1 = make_stage(c0, c0, stride=1)
        self.stage2 = make_stage(c0, c1, stride=2)
        self.stage3 = make_stage(c1, c2, stride=2)
        self.stage4 = make_stage(c2, c3, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(c3, 6)

    def forward(
        self,
        kin: torch.Tensor,                       # (B, T, kin_in)
        ins: torch.Tensor,                       # (B, T, ins_in)
        vo:  torch.Tensor | None = None,         # (B, T, vo_in)  required when use_vo=True
        kin_mask: torch.Tensor | None = None,    # (B,) 0/1 — modality dropout
        ins_mask: torch.Tensor | None = None,
        vo_mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = kin.shape

        k = self.embed_kin(kin)      # (B, T, kin_emb)
        s = self.embed_ins(ins)      # (B, T, ins_emb)
        if kin_mask is not None:
            k = k * kin_mask.view(B, 1, 1)
        if ins_mask is not None:
            s = s * ins_mask.view(B, 1, 1)
        feats = [k, s]

        if self.use_vo:
            assert vo is not None, "vo tensor required when model was built with vo_in"
            v = self.embed_vo(vo)
            if vo_mask is not None:
                v = v * vo_mask.view(B, 1, 1)
            feats.append(v)

        x = torch.cat(feats, dim=-1)         # (B, T, C)
        x = x.transpose(1, 2)                # (B, C, T)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)         # (B, c3)
        x = self.dropout(x)
        return self.head(x)                  # (B, 6)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
