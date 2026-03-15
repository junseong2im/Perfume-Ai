"""MixtureNet v3 — Set Transformer + Pairwise Interaction + Temporal + Conditions
================================================================================
설계안 v3 Model 2 구현

입력: [(odor_22d, ratio, phys_7d, time_t) × N재료 + condition(mood+season)]
출력: 혼합 향 22d, TMB 66d, 시너지, 아코드, 쾌적도

Architecture:
  Ingredient Encoder [31d→128d]
  → ISAB × 2 (Set Transformer, 순서 무관)
  → PMA (→ set-level summary)
  → Condition Fusion (mood/season)
  → Pairwise Interaction (모든 쌍 시너지/길항 계산)
  → 5 Heads

Features:
  - Weber-Fechner 비선형 농도 변환
  - 시간축 입력 (t=0~480min)
  - 무드/시즌 조건부 생성
  - Competitive binding 기반 pairwise interaction

~500K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ================================================================
# Set Transformer Components (Lee et al. 2019)
# ================================================================

class MultiheadAttentionBlock(nn.Module):
    """MAB: MultiheadAttention + LayerNorm + FFN"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, Q, K, mask=None):
        """Q attends to K"""
        attn_out, _ = self.mha(Q, K, K, key_padding_mask=mask)
        Q = self.norm1(Q + attn_out)
        Q = self.norm2(Q + self.ffn(Q))
        return Q


class InducedSetAttentionBlock(nn.Module):
    """ISAB: Set Transformer with inducing points (O(nm) instead of O(n²))"""
    def __init__(self, d_model, n_heads, n_inducing=16, dropout=0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)
        self.mab1 = MultiheadAttentionBlock(d_model, n_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, n_heads, dropout)

    def forward(self, x, mask=None):
        """x: [B, N, d_model] → [B, N, d_model]"""
        B = x.size(0)
        I = self.inducing.expand(B, -1, -1)  # [B, n_inducing, d]
        H = self.mab1(I, x, mask=mask)        # [B, n_inducing, d]
        return self.mab2(x, H)                 # [B, N, d]


class PoolingByMultiheadAttention(nn.Module):
    """PMA: Compress set to k seed vectors"""
    def __init__(self, d_model, n_heads, n_seeds=1, dropout=0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.mab = MultiheadAttentionBlock(d_model, n_heads, dropout)

    def forward(self, x, mask=None):
        """x: [B, N, d] → [B, n_seeds, d]"""
        B = x.size(0)
        S = self.seeds.expand(B, -1, -1)
        return self.mab(S, x, mask=mask)


# ================================================================
# Weber-Fechner 농도 변환
# ================================================================

def weber_fechner_transform(concentrations):
    """Weber-Fechner law: perceived ∝ log(C)
    Args: concentrations: [B, N] or [B] — 0~100%
    Returns: same shape, log-scaled
    """
    return torch.log1p(concentrations)


# ================================================================
# Pairwise Interaction Module
# ================================================================

class PairwiseInteraction(nn.Module):
    """모든 재료 쌍의 상호작용 계산 (시너지/길항)"""
    def __init__(self, d_model=128):
        super().__init__()
        self.pair_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Tanh(),  # -1 (길항) ~ +1 (시너지)
        )
        self.interaction_proj = nn.Linear(d_model + 1, d_model)

    def forward(self, x, mask=None):
        """
        x: [B, N, d_model] — ingredient representations
        mask: [B, N] bool — True = padding (ignore)
        Returns: [B, d_model] — interaction-enhanced mixture representation
        """
        B, N, D = x.shape

        # Build per-sample real ingredient counts
        if mask is not None:
            # real_mask: True = real ingredient, False = padding
            real_mask = ~mask  # [B, N]
        else:
            real_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)

        pair_scores = torch.zeros(B, device=x.device)
        pair_counts = torch.zeros(B, device=x.device)

        if N >= 2:
            for i in range(N):
                for j in range(i+1, N):
                    # ★ Only compute interaction for pairs where BOTH are real
                    valid = real_mask[:, i] & real_mask[:, j]  # [B] bool
                    if not valid.any():
                        continue
                    pair = torch.cat([x[:, i], x[:, j]], dim=-1)  # [B, 2D]
                    score = self.pair_net(pair).squeeze(-1)  # [B]
                    pair_scores += score * valid.float()
                    pair_counts += valid.float()

            pair_scores = pair_scores / pair_counts.clamp(min=1)

        # Masked mean of set (only real ingredients)
        real_mask_f = real_mask.unsqueeze(-1).float()  # [B, N, 1]
        set_mean = (x * real_mask_f).sum(dim=1) / real_mask_f.sum(dim=1).clamp(min=1)  # [B, D]
        pair_feat = pair_scores.unsqueeze(-1)  # [B, 1]

        combined = torch.cat([set_mean, pair_feat], dim=-1)  # [B, D+1]
        return self.interaction_proj(combined), pair_scores  # [B, D], [B]


# ================================================================
# Condition Encoder (Mood + Season)
# ================================================================

# 16 moods
MOODS = ['romantic', 'energetic', 'calm', 'mysterious', 'fresh',
         'warm', 'elegant', 'playful', 'sensual', 'powerful',
         'minimalist', 'bohemian', 'royal', 'sporty', 'cozy', 'exotic']

# 4 seasons
SEASONS = ['spring', 'summer', 'autumn', 'winter']

class ConditionEncoder(nn.Module):
    """무드(16d one-hot) + 시즌(4d one-hot) → 128d"""
    def __init__(self, n_moods=16, n_seasons=4, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_moods + n_seasons, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, out_dim), nn.GELU(), nn.LayerNorm(out_dim),
        )
        self.n_moods = n_moods
        self.n_seasons = n_seasons

    def forward(self, mood_idx=None, season_idx=None, condition_vec=None):
        """
        Args:
            mood_idx: [B] int indices (or None)
            season_idx: [B] int indices (or None)
            condition_vec: [B, 20] pre-computed (overrides above)
        """
        if condition_vec is not None:
            return self.net(condition_vec)

        B = mood_idx.size(0) if mood_idx is not None else season_idx.size(0)
        device = mood_idx.device if mood_idx is not None else season_idx.device

        mood_oh = torch.zeros(B, self.n_moods, device=device)
        if mood_idx is not None:
            mood_oh.scatter_(1, mood_idx.unsqueeze(1).clamp(0, self.n_moods-1), 1.0)

        season_oh = torch.zeros(B, self.n_seasons, device=device)
        if season_idx is not None:
            season_oh.scatter_(1, season_idx.unsqueeze(1).clamp(0, self.n_seasons-1), 1.0)

        return self.net(torch.cat([mood_oh, season_oh], dim=-1))


# ================================================================
# MixtureNet Main Model
# ================================================================

class MixtureNet(nn.Module):
    """Set Transformer based mixture predictor

    Input per ingredient: [odor_22d, ratio_1d, phys_7d, time_1d] = 31d
    Condition: [mood_16d + season_4d] = 20d
    """
    def __init__(self, ing_dim=31, d_model=128, n_heads=4, n_inducing=16,
                 n_moods=16, n_seasons=4, dropout=0.1, max_ingredients=20):
        super().__init__()
        self.d_model = d_model
        self.max_ingredients = max_ingredients

        # === Ingredient Encoder ===
        self.ing_encoder = nn.Sequential(
            nn.Linear(ing_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )

        # === Set Transformer ===
        self.isab1 = InducedSetAttentionBlock(d_model, n_heads, n_inducing, dropout)
        self.isab2 = InducedSetAttentionBlock(d_model, n_heads, n_inducing, dropout)

        # === Pooling ===
        self.pma = PoolingByMultiheadAttention(d_model, n_heads, n_seeds=1, dropout=dropout)

        # === Pairwise Interaction ===
        self.pairwise = PairwiseInteraction(d_model)

        # === Condition Encoder & Fusion ===
        self.cond_encoder = ConditionEncoder(n_moods, n_seasons, d_model)
        self.cond_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2), nn.GELU(), nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.LayerNorm(d_model),
        )

        # === Output Heads ===
        self.head_mixture = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(64, 22), nn.Sigmoid(),
        )
        self.head_temporal = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 66), nn.Sigmoid(),  # Top(22)+Mid(22)+Base(22)
        )
        self.head_synergy = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
        self.head_accord = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 12),  # 12 accord categories (softmax at loss)
        )
        self.head_hedonic = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Tanh(),
        )

    def forward(self, ingredients, mask=None, condition_vec=None,
                mood_idx=None, season_idx=None):
        """
        Args:
            ingredients: [B, N, 31] — (odor_22d, ratio_1d, phys_7d, time_1d) per ingredient
                         ratio → Weber-Fechner log(1+ratio) applied before input
            mask: [B, N] bool — True = padding (ignore)
            condition_vec: [B, 20] — mood+season one-hot (optional)
            mood_idx: [B] — mood index (optional, alternative to condition_vec)
            season_idx: [B] — season index (optional)

        Returns:
            dict with mixture(22), temporal(66), synergy(1), accord(12), hedonic(1)
        """
        B, N, D = ingredients.shape

        # Encode each ingredient
        ing_feat = self.ing_encoder(ingredients)  # [B, N, 128]

        # Set Transformer
        h = self.isab1(ing_feat, mask=mask)
        h = self.isab2(h, mask=mask)

        # Pool to single vector
        pooled = self.pma(h, mask=mask).squeeze(1)  # [B, 128]

        # Pairwise interactions
        pair_feat, pair_scores = self.pairwise(h, mask=mask)  # [B, 128], [B]

        # Condition encoding
        if condition_vec is not None or mood_idx is not None or season_idx is not None:
            cond = self.cond_encoder(mood_idx, season_idx, condition_vec)
        else:
            cond = torch.zeros(B, self.d_model, device=ingredients.device)

        # Fuse: pooled + pairwise + condition
        fused = self.cond_fusion(torch.cat([pooled, pair_feat, cond], dim=-1))  # [B, 128]

        # Heads
        return {
            'mixture': self.head_mixture(fused),       # [B, 22]
            'temporal': self.head_temporal(fused),      # [B, 66]
            'synergy': self.head_synergy(fused),        # [B, 1]
            'accord': self.head_accord(fused),          # [B, 12]
            'hedonic': self.head_hedonic(fused),        # [B, 1]
            'pair_scores': pair_scores,                  # [B]
        }

    def prepare_ingredients(self, odor_vecs, ratios, phys_props=None, time_t=0.0):
        """Helper: 개별 텐서 → [B, N, 31] 입력 포맷으로 변환

        Args:
            odor_vecs: [B, N, 22] — OdorPredictor 출력
            ratios: [B, N] — 농도 (%)
            phys_props: [B, N, 7] — MW,LogP,VP,BP,TPSA,HBD,HBA (optional)
            time_t: float or [B] — 시간 (분), 0=초기
        """
        B, N, _ = odor_vecs.shape
        device = odor_vecs.device

        # Weber-Fechner transform on ratios
        wf_ratios = weber_fechner_transform(ratios).unsqueeze(-1)  # [B, N, 1]

        # Physical props (7d, zeros if not provided)
        if phys_props is None:
            phys_props = torch.zeros(B, N, 7, device=device)

        # Time (1d)
        if isinstance(time_t, (int, float)):
            time_feat = torch.full((B, N, 1), time_t / 480.0, device=device)  # Normalize by 8h
        else:
            time_feat = (time_t / 480.0).unsqueeze(-1).unsqueeze(-1).expand(B, N, 1)

        # Concat
        return torch.cat([odor_vecs, wf_ratios, phys_props, time_feat], dim=-1)  # [B, N, 31]


def compute_mixture_loss(pred, target, masks=None):
    """MixtureNet loss computation"""
    losses = {}

    # HA: Mixture MSE + CosSim
    L_mix = F.mse_loss(pred['mixture'], target['mixture'])
    L_cos = 1 - F.cosine_similarity(pred['mixture'], target['mixture']).mean()
    losses['mixture'] = L_mix + 0.5 * L_cos

    # HB: Temporal MSE
    if 'temporal' in target:
        losses['temporal'] = F.mse_loss(pred['temporal'], target['temporal'])
    else:
        losses['temporal'] = torch.tensor(0.0)

    # HC: Synergy MSE
    if 'synergy' in target:
        losses['synergy'] = F.mse_loss(pred['synergy'], target['synergy'])
    else:
        losses['synergy'] = torch.tensor(0.0)

    # HD: Accord CE
    if 'accord' in target:
        losses['accord'] = F.cross_entropy(pred['accord'], target['accord'])
    else:
        losses['accord'] = torch.tensor(0.0)

    # HE: Hedonic MSE
    if 'hedonic' in target:
        losses['hedonic'] = F.mse_loss(pred['hedonic'], target['hedonic'])
    else:
        losses['hedonic'] = torch.tensor(0.0)

    # Weighted total
    total = (1.0 * losses['mixture'] +
             0.3 * losses['temporal'] +
             0.2 * losses['synergy'] +
             0.3 * losses['accord'] +
             0.2 * losses['hedonic'])

    return total, losses


# ================================================================
# Quick Test
# ================================================================

if __name__ == '__main__':
    print("=== MixtureNet Architecture Test ===")

    model = MixtureNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Dummy input: 4 batches, 5 ingredients each
    B, N = 4, 5
    ingredients = torch.randn(B, N, 31)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, 3:] = True  # Last 2 are padding

    # With conditions
    mood_idx = torch.randint(0, 16, (B,))
    season_idx = torch.randint(0, 4, (B,))

    preds = model(ingredients, mask=mask, mood_idx=mood_idx, season_idx=season_idx)
    print("\nOutputs:")
    for name, tensor in preds.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {name}: {tensor.shape}")

    # Test prepare_ingredients
    odor_vecs = torch.randn(B, N, 22)
    ratios = torch.rand(B, N) * 20  # 0-20%
    inp = model.prepare_ingredients(odor_vecs, ratios, time_t=30.0)
    print(f"\nPrepared input shape: {inp.shape}")

    print("\n✅ MixtureNet test passed!")
