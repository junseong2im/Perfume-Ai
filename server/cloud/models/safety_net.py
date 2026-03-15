"""SafetyNet - IFRA/Allergen/Toxicity Prediction
================================================
Input: OdorPredictor backbone 512d + concentration(1d) + category(15d)
Output: IFRA violation, 26 allergens, 4 hazards, max_concentration
Focal BCE Loss for imbalanced safety data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SAFETY_CATEGORIES = [
    'floral', 'woody', 'citrus', 'spicy', 'musk',
    'fruity', 'green', 'herbal', 'amber', 'leather',
    'gourmand', 'aquatic', 'aldehyde', 'smoky', 'animalic',
]

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
    def __init__(self, backbone_dim=512, n_categories=15, n_allergens=26, n_hazards=4):
        super().__init__()
        input_dim = backbone_dim + 1 + n_categories
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.LayerNorm(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(0.1),
        )
        self.head_ifra = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Dropout(0.05), nn.Linear(32, 1), nn.Sigmoid())
        self.head_allergen = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.05), nn.Linear(64, n_allergens), nn.Sigmoid())
        self.head_hazard = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Dropout(0.05), nn.Linear(32, n_hazards), nn.Sigmoid())
        self.head_max_conc = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, backbone_features, concentration, category_onehot):
        x = torch.cat([backbone_features, concentration, category_onehot], dim=-1)
        h = self.encoder(x)
        return {
            'ifra_violation': self.head_ifra(h),
            'allergen': self.head_allergen(h),
            'hazard': self.head_hazard(h),
            'max_concentration': self.head_max_conc(h) * 100,
        }


def focal_bce_loss(pred, target, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * ((1 - pt) ** gamma) * bce).mean()


def compute_safety_loss(pred, target, masks=None):
    L_ifra = focal_bce_loss(pred['ifra_violation'], target['ifra_violation'])
    L_aller = focal_bce_loss(pred['allergen'], target['allergen'])
    L_hazard = focal_bce_loss(pred['hazard'], target['hazard'])
    pred_conc_norm = pred['max_concentration'] / 100.0
    tgt_conc_norm = target['max_concentration'] / 100.0
    L_conc = F.mse_loss(pred_conc_norm, tgt_conc_norm)
    total = 1.0 * L_ifra + 0.5 * L_aller + 0.3 * L_hazard + 0.2 * L_conc
    return total, {'ifra': L_ifra, 'allergen': L_aller, 'hazard': L_hazard, 'conc': L_conc}
