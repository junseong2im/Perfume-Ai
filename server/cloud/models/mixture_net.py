"""MixtureNet - Set Transformer for Fragrance Mixture Prediction
===============================================================
Input: N ingredients x 31d (odor[22] + conc[1] + time[1] + reserved[7])
Output: mixture[22], temporal[66], synergy[1], accord[12], hedonic[1]
~500K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InducedSetAttention(nn.Module):
    """ISAB: Induced Set Attention Block (O(N*M) instead of O(N^2))"""
    def __init__(self, d_model=128, n_inducing=8, nhead=4):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.attn1 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B = x.size(0)
        ind = self.inducing.expand(B, -1, -1)
        h, _ = self.attn1(ind, x, x, key_padding_mask=mask)
        h = self.norm1(h + ind)
        out, _ = self.attn2(x, h, h)
        return self.norm2(out + x)


class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, d_model=128, n_seeds=1, nhead=4):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B = x.size(0)
        s = self.seeds.expand(B, -1, -1)
        out, _ = self.attn(s, x, x, key_padding_mask=mask)
        return self.norm(out + s)


class PairwiseInteraction(nn.Module):
    """Pairwise ingredient interaction module"""
    def __init__(self, d_model=128):
        super().__init__()
        self.proj_q = nn.Linear(d_model, 64)
        self.proj_k = nn.Linear(d_model, 64)
        self.score_net = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Tanh()
        )

    def forward(self, x, mask=None):
        B, N, D = x.shape
        q = self.proj_q(x)
        k = self.proj_k(x)
        scores = []
        for i in range(N):
            for j in range(i + 1, N):
                diff = q[:, i] - k[:, j]
                s = self.score_net(diff)
                scores.append(s)
        if scores:
            return torch.cat(scores, dim=-1).mean(dim=-1)
        return torch.zeros(B, device=x.device)


class WeberFechner(nn.Module):
    """Concentration -> perceptual intensity (Weber-Fechner law)"""
    def __init__(self, d_out=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d_out))

    def forward(self, concentration):
        log_c = torch.log1p(concentration.clamp(min=0))
        return self.mlp(log_c)


class MixtureNet(nn.Module):
    """Set Transformer for fragrance mixture prediction"""
    def __init__(self, ingredient_dim=31, d_model=128, n_isab=2, n_inducing=8):
        super().__init__()
        self.input_proj = nn.Linear(ingredient_dim, d_model)
        self.weber = WeberFechner(d_out=8)
        self.conc_proj = nn.Linear(8, d_model)

        # Condition embeddings (mood/season)
        self.mood_emb = nn.Embedding(16, d_model)
        self.season_emb = nn.Embedding(4, d_model)

        # Set Transformer encoder
        self.isab_layers = nn.ModuleList(
            [InducedSetAttention(d_model, n_inducing) for _ in range(n_isab)]
        )
        self.pma = PMA(d_model, n_seeds=1)
        self.pairwise = PairwiseInteraction(d_model)

        # Output heads
        self.head_mixture = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 22), nn.Sigmoid()
        )
        self.head_temporal = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 66), nn.Sigmoid()
        )
        self.head_synergy = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.head_accord = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 12), nn.Softmax(dim=-1)
        )
        self.head_hedonic = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Linear(32, 1), nn.Tanh()
        )

    def forward(self, ingredients, mask=None, mood_idx=None, season_idx=None):
        B, N, D = ingredients.shape
        h = self.input_proj(ingredients)
        conc = ingredients[:, :, 22:23]
        wf = self.weber(conc)
        h = h + self.conc_proj(wf)
        if mood_idx is not None:
            h = h + self.mood_emb(mood_idx).unsqueeze(1)
        if season_idx is not None:
            h = h + self.season_emb(season_idx).unsqueeze(1)
        for isab in self.isab_layers:
            h = isab(h, mask=mask)
        pair_scores = self.pairwise(h, mask)
        pooled = self.pma(h, mask).squeeze(1)
        return {
            'mixture': self.head_mixture(pooled),
            'temporal': self.head_temporal(pooled),
            'synergy': self.head_synergy(pooled),
            'accord': self.head_accord(pooled),
            'hedonic': self.head_hedonic(pooled),
            'pair_scores': pair_scores,
        }


def compute_mixture_loss(pred, target, masks=None):
    L_mix = F.mse_loss(pred['mixture'], target['mixture'])
    return L_mix, {'mixture': L_mix}
