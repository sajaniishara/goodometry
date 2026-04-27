"""Factorized modal-then-temporal causal transformer for Go2 state estimation.

Per `go2-concrete-algorithms.md` §5.2. Two or three modality tokens per timestep
(kinematics, INS, optional VO). Architecture is agnostic to M.

Shapes (M=2, no VO):
    input: kin (B, T, 31) + ins (B, T, 10)
    project each to 128-D via a 2-layer MLP → per-modality tokens (B, T, 128)
    stack modality tokens:  x (B, T, M=2, 128)
    block: modal self-attn across M, then causal temporal self-attn across T
    head:  mean over M, take last-timestep embedding, linear → 6

Shapes (M=3, with VO):
    input: kin (B, T, 31) + ins (B, T, 10) + vo (B, T, 7)
    same flow with x (B, T, M=3, 128)

Output: 6-D body-frame velocity [vx, vy, vz, ωx, ωy, ωz].
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEmbedder(nn.Module):
    """2-layer MLP per the concrete-algorithms recipe; operates per-timestep."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim) → (B, T, out_dim)
        return self.net(x)


def sinusoidal_pe(T: int, d: int, device) -> torch.Tensor:
    """(T, d) sinusoidal positional encoding."""
    pe = torch.zeros(T, d, device=device)
    pos = torch.arange(T, device=device).unsqueeze(1).float()
    i = torch.arange(0, d, 2, device=device).float()
    div = torch.exp(-i * math.log(10000.0) / d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class FactorizedBlock(nn.Module):
    """One modal self-attn, then one causal temporal self-attn. Pre-norm."""
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.modal_norm  = nn.LayerNorm(d_model)
        self.modal_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temp_norm   = nn.LayerNorm(d_model)
        self.temp_attn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn_norm    = nn.LayerNorm(d_model)
        self.ffn         = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, M, D)
        B, T, M, D = x.shape

        # Modal self-attn over M tokens — each (B*T, M, D).
        xm = x.reshape(B * T, M, D)
        xm_n = self.modal_norm(xm)
        xm = xm + self.modal_attn(xm_n, xm_n, xm_n, need_weights=False)[0]

        # Temporal causal self-attn over T — each (B*M, T, D).
        xt = xm.reshape(B, T, M, D).permute(0, 2, 1, 3).reshape(B * M, T, D)
        xt_n = self.temp_norm(xt)
        xt = xt + self.temp_attn(xt_n, xt_n, xt_n,
                                 attn_mask=causal_mask, need_weights=False)[0]

        # FFN per token.
        xt = xt + self.ffn(self.ffn_norm(xt))

        # Back to (B, T, M, D).
        return xt.reshape(B, M, T, D).permute(0, 2, 1, 3)


class FusionTransformer(nn.Module):
    def __init__(
        self,
        kin_in: int = 31,
        ins_in: int = 10,
        vo_in:  int | None = None,    # None → M=2 (no VO); set to VO_DIM=7 for fusion_v2
        d_model: int = 128,
        n_blocks: int = 2,
        n_heads: int = 4,
        ffn_dim: int = 256,
        clip_len: int = 40,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.clip_len = clip_len
        self.d_model = d_model
        self.use_vo = vo_in is not None

        self.embed_kin = ModalityEmbedder(kin_in, d_model)
        self.embed_ins = ModalityEmbedder(ins_in, d_model)
        if self.use_vo:
            self.embed_vo = ModalityEmbedder(vo_in, d_model)

        n_modalities = 3 if self.use_vo else 2
        self.modality_emb = nn.Parameter(torch.zeros(n_modalities, d_model))
        nn.init.normal_(self.modality_emb, std=0.02)

        self.blocks = nn.ModuleList([
            FactorizedBlock(d_model, n_heads, ffn_dim, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 6)           # [vx vy vz ωx ωy ωz]

    def forward(
        self,
        kin: torch.Tensor,                       # (B, T, kin_in)
        ins: torch.Tensor,                       # (B, T, ins_in)
        vo:  torch.Tensor | None = None,         # (B, T, vo_in)  — required when use_vo=True
        kin_mask: torch.Tensor | None = None,    # (B,) 0/1 — modality dropout at training
        ins_mask: torch.Tensor | None = None,
        vo_mask:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = kin.shape

        k = self.embed_kin(kin) + self.modality_emb[0]     # (B, T, D)
        s = self.embed_ins(ins) + self.modality_emb[1]     # (B, T, D)

        if kin_mask is not None:
            k = k * kin_mask.view(B, 1, 1)
        if ins_mask is not None:
            s = s * ins_mask.view(B, 1, 1)

        # Add temporal positional encoding.
        pe = sinusoidal_pe(T, self.d_model, kin.device).unsqueeze(0)  # (1, T, D)
        k = k + pe
        s = s + pe

        tokens = [k, s]

        if self.use_vo:
            assert vo is not None, "vo tensor required when model was built with vo_in"
            v = self.embed_vo(vo) + self.modality_emb[2]   # (B, T, D)
            if vo_mask is not None:
                v = v * vo_mask.view(B, 1, 1)
            v = v + pe
            tokens.append(v)

        # Stack into (B, T, M, D).
        x = torch.stack(tokens, dim=2)

        # Causal mask for the temporal attention (T, T).
        causal = torch.triu(torch.ones(T, T, device=kin.device, dtype=torch.bool), diagonal=1)

        for block in self.blocks:
            x = block(x, causal_mask=causal)

        # Readout: mean over modalities, take the last timestep, then a head.
        x = x.mean(dim=2)                                  # (B, T, D)
        x = self.head_norm(x[:, -1])                       # (B, D)
        return self.head(x)                                # (B, 6)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
