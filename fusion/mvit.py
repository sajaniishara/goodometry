"""MViT-inspired multiscale transformer for sensor fusion (no images).

Mirrors the MViT V2 design at a smaller scale, adapted to 1D sensor sequences:
  - per-modality embedders project (kin, ins, vo) per-timestep to a token
  - tokens are arranged as (B, T, M, D) — M = number of modalities
  - 3 stages with progressive temporal pooling (T → T/2 → T/4) and channel
    expansion (D → 1.5·D → 2·D), each containing N factorized modal+temporal
    attention blocks
  - readout: mean-pool over (T, M), linear head → 6

The factorized block reuses the modal-then-temporal-attention pattern of
goodometry/fusion/model.py::FactorizedBlock; no causal mask (we read out a
single regression at the clip end, not a per-timestep stream).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


def sinusoidal_pe(T: int, d: int, device) -> torch.Tensor:
    pe = torch.zeros(T, d, device=device)
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(-i * math.log(10000.0) / d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class FactorizedAttnBlock(nn.Module):
    """Modal self-attn (over M) → temporal self-attn (over T) → FFN. Pre-norm."""
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.modal_norm = nn.LayerNorm(d_model)
        self.modal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temp_norm  = nn.LayerNorm(d_model)
        self.temp_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm   = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M, D)
        B, T, M, D = x.shape
        xm = x.reshape(B * T, M, D)
        xm_n = self.modal_norm(xm)
        xm = xm + self.modal_attn(xm_n, xm_n, xm_n, need_weights=False)[0]

        xt = xm.reshape(B, T, M, D).permute(0, 2, 1, 3).reshape(B * M, T, D)
        xt_n = self.temp_norm(xt)
        xt = xt + self.temp_attn(xt_n, xt_n, xt_n, need_weights=False)[0]

        xt = xt + self.ffn(self.ffn_norm(xt))
        return xt.reshape(B, M, T, D).permute(0, 2, 1, 3)


class StageDownsample(nn.Module):
    """MViT-style transition: temporal avg-pool (stride 2 over T) + channel expand."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M, D_in)
        B, T, M, D = x.shape
        # Temporal avg-pool over pairs of timesteps (stride 2, kernel 2).
        if T % 2 == 1:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, 1))   # pad T to even
            T = T + 1
        x = x.reshape(B, T // 2, 2, M, D).mean(dim=2)      # (B, T/2, M, D)
        x = self.proj(x)
        x = self.norm(x)
        return x


class FusionMViT(nn.Module):
    def __init__(
        self,
        kin_in: int = 31,
        ins_in: int = 10,
        vo_in:  int | None = None,
        stage_dims: tuple[int, int, int] = (64, 96, 128),
        blocks_per_stage: tuple[int, int, int] = (2, 2, 2),
        n_heads: int = 4,
        ffn_mult: int = 2,
        clip_len: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_vo = vo_in is not None
        self.clip_len = clip_len

        d0 = stage_dims[0]
        self.embed_kin = nn.Sequential(nn.Linear(kin_in, d0), nn.GELU(), nn.Linear(d0, d0))
        self.embed_ins = nn.Sequential(nn.Linear(ins_in, d0), nn.GELU(), nn.Linear(d0, d0))
        if self.use_vo:
            self.embed_vo = nn.Sequential(nn.Linear(vo_in, d0), nn.GELU(), nn.Linear(d0, d0))

        n_modalities = 3 if self.use_vo else 2
        self.modality_emb = nn.Parameter(torch.zeros(n_modalities, d0))
        nn.init.normal_(self.modality_emb, std=0.02)

        # Three stages — temporal pool happens at the start of stages 2 and 3.
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_d = d0
        for s_idx, (d, n_blocks) in enumerate(zip(stage_dims, blocks_per_stage)):
            if s_idx > 0:
                self.downsamples.append(StageDownsample(prev_d, d))
            blocks = nn.ModuleList([
                FactorizedAttnBlock(d, n_heads, ffn_dim=d * ffn_mult, dropout=dropout)
                for _ in range(n_blocks)
            ])
            self.stages.append(blocks)
            prev_d = d

        d_last = stage_dims[-1]
        self.head_norm = nn.LayerNorm(d_last)
        self.head = nn.Linear(d_last, 6)

    def forward(
        self,
        kin: torch.Tensor,
        ins: torch.Tensor,
        vo:  torch.Tensor | None = None,
        kin_mask: torch.Tensor | None = None,
        ins_mask: torch.Tensor | None = None,
        vo_mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = kin.shape

        k = self.embed_kin(kin) + self.modality_emb[0]
        s = self.embed_ins(ins) + self.modality_emb[1]
        if kin_mask is not None:
            k = k * kin_mask.view(B, 1, 1)
        if ins_mask is not None:
            s = s * ins_mask.view(B, 1, 1)

        d0 = k.size(-1)
        pe = sinusoidal_pe(T, d0, kin.device).unsqueeze(0)
        k = k + pe
        s = s + pe
        tokens = [k, s]

        if self.use_vo:
            assert vo is not None, "vo tensor required when model was built with vo_in"
            v = self.embed_vo(vo) + self.modality_emb[2]
            if vo_mask is not None:
                v = v * vo_mask.view(B, 1, 1)
            v = v + pe
            tokens.append(v)

        x = torch.stack(tokens, dim=2)       # (B, T, M, D)

        for s_idx, blocks in enumerate(self.stages):
            if s_idx > 0:
                x = self.downsamples[s_idx - 1](x)
            for blk in blocks:
                x = blk(x)

        # Mean over (T, M) → (B, D)
        x = x.mean(dim=(1, 2))
        x = self.head_norm(x)
        return self.head(x)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
