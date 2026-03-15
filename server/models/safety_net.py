"""SafetyNet — IFRA/Allergen/Toxicity Prediction
================================================
설계안 v3 Model 3 구현

입력: OdorPredictor backbone 512d (frozen) + 농도(1d) + 카테고리(15d)
출력: IFRA 위반, 26 알레르겐, 4 위험 요소

Focal BCE Loss for imbalanced safety data
~200K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 15 향료 카테고리
SAFETY_CATEGORIES = [
    'floral', 'woody', 'citrus', 'spicy', 'musk',
    'fruity', 'green', 'herbal', 'amber', 'leather',
    'gourmand', 'aquatic', 'aldehyde', 'smoky', 'animalic',
]

# EU 26종 의무 표기 알레르겐
EU_26_ALLERGENS = [
    'Amyl cinnamal', 'Amylcinnamyl alcohol', 'Anise alcohol',
    'Benzyl alcohol', 'Benzyl benzoate', 'Benzyl cinnamate',
    'Benzyl salicylate', 'Cinnamal', 'Cinnamyl alcohol',
    'Citral', 'Citronellol', 'Coumarin',
    'Eugenol', 'Evernia furfuracea', 'Evernia prunastri',
    'Farnesol', 'Geraniol', 'Hexyl cinnamal',
    'HICC', 'Hydroxycitronellal', 'Isoeugenol',
    'Limonene', 'Linalool', 'Methyl 2-octynoate',
    'alpha-Isomethyl ionone', 'Butylphenyl methylpropional',
]


class SafetyNet(nn.Module):
    """분자별 안전성 예측 (IFRA/알레르겐/독성)

    Args:
        backbone_dim: OdorPredictor fusion output dimension (512d)
    """
    def __init__(self, backbone_dim=512, n_categories=15, n_allergens=26, n_hazards=4):
        super().__init__()

        input_dim = backbone_dim + 1 + n_categories  # 512 + 1 + 15 = 528

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
        )

        # IFRA violation probability (분자별)
        self.head_ifra = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

        # EU 26 allergens
        self.head_allergen = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(64, n_allergens), nn.Sigmoid(),
        )

        # Hazard types: [skin_irritation, sensitization, phototoxicity, environmental]
        self.head_hazard = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(32, n_hazards), nn.Sigmoid(),
        )

        # Max concentration prediction (regression, 0-100%)
        self.head_max_conc = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),  # × 100 for percentage
        )

    def forward(self, backbone_features, concentration, category_onehot):
        """
        Args:
            backbone_features: [B, 512] from OdorPredictor (frozen)
            concentration: [B, 1] 현재 농도 (%)
            category_onehot: [B, 15] 향료 카테고리

        Returns:
            dict: ifra_prob, allergen_probs, hazard_probs, max_concentration
        """
        x = torch.cat([backbone_features, concentration, category_onehot], dim=-1)
        h = self.encoder(x)

        return {
            'ifra_violation': self.head_ifra(h),       # [B, 1] 0-1
            'allergen': self.head_allergen(h),          # [B, 26] 0-1
            'hazard': self.head_hazard(h),             # [B, 4] 0-1
            'max_concentration': self.head_max_conc(h) * 100,  # [B, 1] 0-100%
        }


def focal_bce_loss(pred, target, gamma=2.0, alpha=0.75):
    """Focal Binary Cross Entropy (soft-label safe)
    
    Uses continuous formulation instead of hard `target == 1` comparison,
    so soft targets like 0.95, 0.8 are handled correctly.
    """
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    # Continuous soft-label pt: when target≈1 → pt≈pred, when target≈0 → pt≈(1-pred)
    pt = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal = alpha_t * ((1 - pt) ** gamma) * bce
    return focal.mean()


def compute_safety_loss(pred, target, masks=None):
    """SafetyNet loss"""
    L_ifra = focal_bce_loss(pred['ifra_violation'], target['ifra_violation'], gamma=2, alpha=0.75)
    L_aller = focal_bce_loss(pred['allergen'], target['allergen'], gamma=2, alpha=0.7)
    L_hazard = focal_bce_loss(pred['hazard'], target['hazard'], gamma=2, alpha=0.8)
    # Normalize concentration to 0-1 range to prevent 10,000x loss scale domination
    pred_conc_norm = pred['max_concentration'] / 100.0
    tgt_conc_norm = target['max_concentration'] / 100.0
    L_conc = F.mse_loss(pred_conc_norm, tgt_conc_norm)

    total = 1.0 * L_ifra + 0.5 * L_aller + 0.3 * L_hazard + 0.2 * L_conc
    return total, {'ifra': L_ifra, 'allergen': L_aller, 'hazard': L_hazard, 'conc': L_conc}


if __name__ == '__main__':
    print("=== SafetyNet Architecture Test ===")
    model = SafetyNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    B = 4
    backbone = torch.randn(B, 512)
    conc = torch.rand(B, 1) * 10
    cat = torch.zeros(B, 15)
    cat[:, 0] = 1  # floral

    preds = model(backbone, conc, cat)
    for name, tensor in preds.items():
        print(f"  {name}: {tensor.shape}")

    print("\n✅ SafetyNet test passed!")
