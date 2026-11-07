"""Cross-attention over modality tokens, with attention logits shifted by log r_m.

At each BEV cell there is one token per stream (lidar, camera). A learned query
(plus the cell's position) attends over those tokens:

    a_m = softmax_m( q . k_m / sqrt(d)  +  log r_m  +  log(valid_m) )

so a stream with r_m -> 0, or a camera token that saw no pixels, drops out of
the softmax. The output is only the attention-weighted mixture: there is no
residual path that carries the lidar features around the gate, otherwise
gating the lidar out would have no effect.
"""

import math

import torch
import torch.nn as nn

LOG_EPS = math.log(1e-4)


class GatedModalityAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2, ffn_mult: int = 2):
        super().__init__()
        assert dim % num_heads == 0
        self.h, self.dh = num_heads, dim // num_heads
        self.query = nn.Parameter(torch.zeros(dim))
        self.pos = nn.Sequential(nn.Linear(2, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(dim), "norm_kv": nn.LayerNorm(dim),
                "q": nn.Linear(dim, dim), "k": nn.Linear(dim, dim), "v": nn.Linear(dim, dim),
                "o": nn.Linear(dim, dim),
                "norm_ffn": nn.LayerNorm(dim),
                "ffn": nn.Sequential(nn.Linear(dim, ffn_mult * dim), nn.GELU(), nn.Linear(ffn_mult * dim, dim)),
            }))

    def forward(self, tokens: torch.Tensor, stream_bias: torch.Tensor, pos_xy: torch.Tensor):
        """tokens (B, N, M, D); stream_bias (B, N, M) additive logits (log r + log valid);
        pos_xy (N, 2) normalised cell positions. Returns fused (B, N, D) and the
        head-averaged attention weights of the last layer (B, N, M)."""
        B, N, M, D = tokens.shape
        q = (self.query + self.pos(pos_xy)).unsqueeze(0).expand(B, -1, -1)       # (B, N, D)
        bias = stream_bias.unsqueeze(2)                                           # (B, N, 1, M)
        weights = None
        for L in self.layers:
            kv = L["norm_kv"](tokens)
            Q = L["q"](L["norm_q"](q)).view(B, N, self.h, self.dh)
            K = L["k"](kv).view(B, N, M, self.h, self.dh)
            V = L["v"](kv).view(B, N, M, self.h, self.dh)
            logits = torch.einsum("bnhd,bnmhd->bnhm", Q, K) / math.sqrt(self.dh)
            weights = torch.softmax(logits.float() + bias.float(), dim=-1).to(V.dtype)
            out = torch.einsum("bnhm,bnmhd->bnhd", weights, V).reshape(B, N, D)
            q = q + L["o"](out)
            q = q + L["ffn"](L["norm_ffn"](q))
        return q, weights.float().mean(dim=2)


def stream_bias(r: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """r (B, M) reliability in (0, 1); valid (B, N, M) bool or None -> (B, N, M) logit bias."""
    b = torch.log(r.clamp(min=1e-4)).unsqueeze(1)                                  # (B, 1, M)
    if valid is None:
        return b
    return b + torch.where(valid, torch.zeros_like(b), torch.full_like(b, LOG_EPS))
