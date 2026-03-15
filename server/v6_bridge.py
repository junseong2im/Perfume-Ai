"""
v6_bridge.py — v6 모델 ↔ 기존 시스템 통합 브릿지
================================================
OdorPredictor v6, MixtureNet, SafetyNet, RecipeVAE를
기존 odor_engine.py 인터페이스에 드롭인 연결합니다.

기존 시스템 호환:
    OdorGNN.encode(smiles) → np.array[22]

v6 업그레이드:
    OdorEngineV6.encode(smiles) → np.array[22]  (동일 인터페이스)
    OdorEngineV6.encode_full(smiles) → {'odor': [22], 'top': [22], ...}
    OdorEngineV6.predict_mixture(smiles_list, ratios) → np.array[22]
    OdorEngineV6.check_safety(smiles) → {'ifra': float, ...}
    OdorEngineV6.generate_recipe(mood, season) → [{'ingredient':..., 'ratio':...}]
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'models'))

from models.odor_predictor_v6 import OdorPredictorV6, smiles_to_graph_v6, extract_phys_props, EMA
from models.mixture_net import MixtureNet
from models.safety_net import SafetyNet
try:
    from train_models import BackboneMixtureNet
    HAS_BACKBONE_MIX = True
except ImportError:
    HAS_BACKBONE_MIX = False
from models.recipe_vae import RecipeVAE

try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',
]


class OdorEngineV6:
    """v6 통합 엔진 — 기존 OdorGNN 드롭인 대체

    기존 코드에서:
        gnn = OdorGNN(device='cuda')
        vec = gnn.encode('CCO')

    v6로 교체:
        engine = OdorEngineV6(weights_dir='weights/v6')
        vec = engine.encode('CCO')  # 동일 인터페이스
    """

    def __init__(self, weights_dir='weights/v6', device='cuda',
                 use_ensemble=True, n_ensemble=5, n_odor_dim=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.weights_dir = Path(weights_dir)
        self.use_ensemble = use_ensemble
        self.n_odor_dim = n_odor_dim  # None = auto-detect from checkpoint

        # Cache
        self._cache = {}
        self._bert_cache = {}

        # Load BERT cache for SMILES → 384d embedding lookup
        self._load_bert_cache()

        # Load models
        self.odor_models = []
        self.mixture_model = None
        self.safety_model = None
        self.recipe_model = None

        self._load_models(n_ensemble)

    def _load_bert_cache(self):
        """ChemBERTa BERT 캐시 로드 (SMILES → 384d)"""
        cache_paths = [
            self.weights_dir / '..' / '..' / 'data' / 'bert_cache.pt',
            Path('data/bert_cache.pt'),
            Path('server/data/bert_cache.pt'),
        ]
        for cp in cache_paths:
            if cp.exists():
                try:
                    raw = torch.load(cp, map_location='cpu', weights_only=False)
                    if isinstance(raw, dict):
                        for k, v in raw.items():
                            if isinstance(v, torch.Tensor):
                                self._bert_cache[k] = v.numpy()
                            elif isinstance(v, np.ndarray):
                                self._bert_cache[k] = v
                    print(f"  BERT cache loaded: {len(self._bert_cache)} embeddings from {cp}")
                    return
                except Exception as e:
                    print(f"  BERT cache load failed ({cp}): {e}")
        print(f"  ⚠ No BERT cache found (will use zero vectors)")

    def _load_models(self, n_ensemble):
        """학습된 모델 로드 (없으면 초기화만)"""
        # OdorPredictor — try multiple filename patterns
        seeds = [42, 123, 456, 789, 1024][:n_ensemble]
        for seed in seeds:
            # Try different checkpoint filename patterns
            patterns = [
                self.weights_dir / f'odor_v6_best_seed{seed}.pt',
                self.weights_dir / f'odor_predictor_v6_seed{seed}.pt',
                self.weights_dir / f'odor_v6_swa_seed{seed}.pt',
            ]
            ckpt_path = None
            for p in patterns:
                if p.exists():
                    ckpt_path = p
                    break

            # Detect n_odor_dim from checkpoint (for logging only)
            n_dim = self.n_odor_dim
            if ckpt_path:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                if n_dim is None and 'n_odor_dim' in ckpt:
                    n_dim = ckpt['n_odor_dim']
                model = OdorPredictorV6(bert_dim=ckpt.get('bert_dim', 384),
                                        use_lora=False)
                model.load_state_dict(ckpt.get('model_state_dict', ckpt))
                cos_val = ckpt.get('val_cos_sim', None)
                cos_str = f"{cos_val:.4f}" if isinstance(cos_val, (int, float)) else str(cos_val)
                print(f"  Loaded OdorPredictor seed={seed} "
                      f"(val_cos={cos_str}, "
                      f"n_odor_dim={n_dim or 22}) from {ckpt_path.name}")
            else:
                model = OdorPredictorV6(bert_dim=384, use_lora=False)

            model = model.to(self.device).eval()
            self.odor_models.append(model)

        if not self.odor_models:
            # No checkpoints — create single untrained model
            model = OdorPredictorV6(bert_dim=384).to(self.device).eval()
            self.odor_models.append(model)
            print("  OdorPredictor: untrained (new init)")

        # MixtureNet — try v6 MixtureNet first, fallback to legacy TrainableMixtureNet
        self.mixture_model = MixtureNet().to(self.device).eval()
        mix_path = self.weights_dir / 'mixture_net.pt'
        legacy_mix_path = self.weights_dir / '..' / 'mixture_transformer.pt'
        if mix_path.exists():
            ckpt = torch.load(mix_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in ckpt:
                self.mixture_model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded MixtureNet (v6)")
        elif legacy_mix_path.exists():
            # S5 fix: fallback to TrainableMixtureNet from train_models.py
            try:
                from train_models import TrainableMixtureNet
                legacy_model = TrainableMixtureNet().to(self.device).eval()
                ckpt = torch.load(legacy_mix_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in ckpt:
                    legacy_model.load_state_dict(ckpt['model_state_dict'])
                self._legacy_mixture = legacy_model
                print(f"  Loaded TrainableMixtureNet (legacy fallback)")
            except Exception as e:
                print(f"  ⚠ Legacy MixtureNet load failed: {e}")

        # B2: BackboneMixtureNet (128d backbone features → 22d)
        self._backbone_mixture = None
        if HAS_BACKBONE_MIX:
            bb_path = self.weights_dir / 'backbone_mixture.pt'
            if bb_path.exists():
                try:
                    bb_model = BackboneMixtureNet(input_dim=128).to(self.device).eval()
                    ckpt = torch.load(bb_path, map_location=self.device, weights_only=False)
                    if 'model_state_dict' in ckpt:
                        bb_model.load_state_dict(ckpt['model_state_dict'])
                    else:
                        bb_model.load_state_dict(ckpt)
                    self._backbone_mixture = bb_model
                    print(f"  Loaded BackboneMixtureNet (128d backbone)")
                except Exception as e:
                    print(f"  ⚠ BackboneMixtureNet load failed: {e}")

        # SafetyNet
        self.safety_model = SafetyNet().to(self.device).eval()
        safe_path = self.weights_dir / 'safety_net.pt'
        if safe_path.exists():
            ckpt = torch.load(safe_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in ckpt:
                self.safety_model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded SafetyNet")

        # RecipeVAE
        self.recipe_model = RecipeVAE(n_ingredients=200).to(self.device).eval()
        vae_path = self.weights_dir / 'recipe_vae.pt'
        if vae_path.exists():
            ckpt = torch.load(vae_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in ckpt:
                self.recipe_model.load_state_dict(ckpt['model_state_dict'])
            print(f"  Loaded RecipeVAE")

        print(f"  Engine V6 ready: {len(self.odor_models)} OdorPredictor(s), "
              f"device={self.device}")

    def _prepare_inputs(self, smiles):
        """SMILES → (graph_batch, bert, phys) tensor 준비
        NOTE: 캐시는 CPU에 저장하여 device 불일치 방지"""
        if smiles in self._cache:
            gb_cpu, bert_cpu, phys_cpu = self._cache[smiles]
            return (
                gb_cpu.clone().to(self.device),
                bert_cpu.clone().to(self.device),
                phys_cpu.clone().to(self.device),
            )

        # Graph
        graph = smiles_to_graph_v6(smiles, device='cpu', compute_3d=False)
        if graph is None:
            from torch_geometric.data import Data
            graph = Data(
                x=torch.zeros(1, 47),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 13),
            )

        graph_batch = Batch.from_data_list([graph])  # CPU

        # ChemBERTa embedding
        bert = self._bert_cache.get(smiles, np.zeros(384, dtype=np.float32))
        bert = torch.tensor(bert, dtype=torch.float32).unsqueeze(0)  # CPU

        # Physical properties
        phys = torch.tensor(
            extract_phys_props(smiles), dtype=torch.float32
        ).unsqueeze(0)  # CPU

        # Cache on CPU
        self._cache[smiles] = (graph_batch, bert, phys)
        return (
            graph_batch.clone().to(self.device),
            bert.clone().to(self.device),
            phys.clone().to(self.device),
        )

    # ================================================================
    # 기존 인터페이스 호환
    # ================================================================

    def encode(self, smiles):
        """OdorGNN.encode(smiles) 드롭인 대체 → np.array[22]"""
        with torch.no_grad():
            graph_batch, bert, phys = self._prepare_inputs(smiles)

            if self.use_ensemble and len(self.odor_models) > 1:
                preds = []
                for model in self.odor_models:
                    # Clone graph for each model to prevent in-place mutation
                    gb_clone = graph_batch.clone()
                    pred = model(bert, gb_clone, phys, return_aux=False)
                    preds.append(pred)
                ensemble_pred = torch.stack(preds).mean(dim=0)
                return ensemble_pred.squeeze(0).cpu().numpy()
            else:
                pred = self.odor_models[0](bert, graph_batch, phys, return_aux=False)
                return pred.squeeze(0).cpu().numpy()

    def encode_batch(self, smiles_list):
        """여러 분자 한번에 인코딩 → np.array[N, 22]"""
        return np.array([self.encode(s) for s in smiles_list])

    # ================================================================
    # v6 확장 기능
    # ================================================================

    def encode_full(self, smiles):
        """전체 10-head 출력 → dict

        Returns:
            dict: {
                'odor': np.array[22],
                'top': np.array[22],
                'mid': np.array[22],
                'base': np.array[22],
                'longevity': float,
                'sillage': float,
                'descriptors': np.array[138],
                'receptors': np.array[400],
                'hedonic': float,
                'super_res': np.array[200],
            }
        """
        with torch.no_grad():
            graph_batch, bert, phys = self._prepare_inputs(smiles)
            pred = self.odor_models[0](bert, graph_batch, phys, return_aux=True)

            return {k: v.squeeze(0).cpu().numpy()
                    if v.dim() > 1 else v.item()
                    for k, v in pred.items()}

    def predict_mixture(self, smiles_list, ratios, mood=None, season=None, time_t=None):
        """혼합물 향 예측 (MixtureNet 사용)

        Args:
            smiles_list: list of SMILES strings
            ratios: list of concentration percentages
            mood: int (0-15) or None
            season: int (0-3) or None
            time_t: float (hours) or None

        Returns:
            dict: {'mixture': np.array[22], 'synergy': float, ...}
        """
        with torch.no_grad():
            # 1. 각 원료의 향 벡터
            odor_vecs = self.encode_batch(smiles_list)
            n_items = len(smiles_list)

            # B2: Use BackboneMixtureNet with 128d backbone features if available
            if self._backbone_mixture is not None:
                return self._predict_mixture_backbone(smiles_list, ratios)

            # S5 fix: use legacy TrainableMixtureNet if loaded
            if hasattr(self, '_legacy_mixture') and self._legacy_mixture is not None:
                return self._predict_mixture_legacy(odor_vecs, ratios)

            # 2. MixtureNet 입력 준비
            conc = np.array(ratios, dtype=np.float32).reshape(-1, 1)
            time_arr = np.full((n_items, 1), time_t or 0, dtype=np.float32)

            # ingredient = [odor(22) + conc(1) + time(1) + reserved(7)] = 31d
            ingredients = np.zeros((1, n_items, 31), dtype=np.float32)
            ingredients[0, :, :22] = odor_vecs
            ingredients[0, :, 22:23] = conc
            ingredients[0, :, 23:24] = time_arr

            ing_tensor = torch.tensor(ingredients, dtype=torch.float32).to(self.device)
            mask = torch.zeros(1, n_items, dtype=torch.bool, device=self.device)

            mood_idx = torch.tensor([mood or 0], device=self.device) if mood is not None else None
            season_idx = torch.tensor([season or 0], device=self.device) if season is not None else None

            preds = self.mixture_model(ing_tensor, mask=mask,
                                        mood_idx=mood_idx, season_idx=season_idx)

            return {
                'mixture': preds['mixture'].squeeze(0).cpu().numpy(),
                'synergy': preds['synergy'].item(),
                'hedonic': preds['hedonic'].item(),
            }

    def _predict_mixture_legacy(self, odor_vecs, ratios):
        """Legacy TrainableMixtureNet prediction (S5 fallback)
        
        TrainableMixtureNet expects:
            odor_vecs: [batch, n_molecules, N_DIM]
            concentrations: [batch, n_molecules, 1]
        """
        n_items = len(ratios)
        n_dim = odor_vecs.shape[1]  # 20 or 22
        
        # Pad to max 5 molecules (as trained)
        padded_vecs = np.zeros((1, 5, n_dim), dtype=np.float32)
        padded_concs = np.zeros((1, 5, 1), dtype=np.float32)
        n_fill = min(n_items, 5)
        padded_vecs[0, :n_fill] = odor_vecs[:n_fill]
        padded_concs[0, :n_fill, 0] = np.array(ratios[:n_fill], dtype=np.float32)
        
        vecs_t = torch.tensor(padded_vecs, dtype=torch.float32).to(self.device)
        concs_t = torch.tensor(padded_concs, dtype=torch.float32).to(self.device)
        
        mixture_pred = self._legacy_mixture(vecs_t, concs_t)
        mixture_np = mixture_pred.squeeze(0).cpu().numpy()
        
        # Estimate synergy from difference vs weighted average
        weights = np.array(ratios[:n_fill]) / (sum(ratios[:n_fill]) + 1e-8)
        linear_mix = (odor_vecs[:n_fill] * weights.reshape(-1, 1)).sum(axis=0)
        synergy = float(np.mean(np.abs(mixture_np[:len(linear_mix)] - linear_mix)))
        
        return {
            'mixture': mixture_np,
            'synergy': synergy,
            'hedonic': 0.5,  # Not available from legacy model
        }

    def _extract_backbone_features(self, smiles_list):
        """B2: SMILES → 128d backbone features from OdorPredictorV6
        
        Uses _prepare_inputs() for correct graph/bert/phys preparation,
        then manually runs the v6 forward path up to backbone+skip (128d)
        without going through the task heads.
        """
        model = self.odor_models[0]  # Use first model
        features = []
        
        for smiles in smiles_list:
            # Reuse existing _prepare_inputs (handles cache, device, fallback)
            graph_batch, bert, phys = self._prepare_inputs(smiles)
            
            # Run through the 3 encoding paths + fusion + backbone
            feat_a = model.path_a(bert)           # [1, 384]
            feat_b = model.path_b(graph_batch)    # [1, 128]
            feat_c = model.path_c(phys)           # [1, 32]
            fused = model.fusion(feat_a, feat_b, feat_c)  # [1, 512]
            backbone_out = model.backbone(fused) + model.skip(fused)  # [1, 128]
            features.append(backbone_out.squeeze(0))
        
        return torch.stack(features)  # [N, 128]
    
    def _predict_mixture_backbone(self, smiles_list, ratios):
        """B2: BackboneMixtureNet prediction using 128d backbone features"""
        bb_features = self._extract_backbone_features(smiles_list)  # [N, 128]
        n_items = len(smiles_list)
        
        # Prepare inputs for BackboneMixtureNet
        feature_vecs = bb_features.unsqueeze(0)  # [1, N, 128]
        concs = torch.tensor(ratios, dtype=torch.float32).reshape(1, n_items, 1).to(self.device)
        
        mixture_pred = self._backbone_mixture(feature_vecs, concs)
        mixture_np = mixture_pred.squeeze(0).cpu().numpy()
        
        # Also compute 22d odor vectors for comparison
        odor_vecs = self.encode_batch(smiles_list)  # [N, 22]
        weights = np.array(ratios) / (sum(ratios) + 1e-8)
        linear_mix = (odor_vecs * weights.reshape(-1, 1)).sum(axis=0)
        synergy = float(np.mean(np.abs(mixture_np[:len(linear_mix)] - linear_mix)))
        
        return {
            'mixture': mixture_np,
            'synergy': synergy,
            'hedonic': 0.5,
            'backbone_mode': True,  # Flag indicating backbone features were used
        }

    def check_safety(self, smiles, concentration=5.0, category='Fine Fragrance'):
        """분자 안전성 확인 (SafetyNet 사용)

        Returns:
            dict: {
                'ifra_violation': float (0-1 probability),
                'allergen': list of (name, prob) for EU 26,
                'hazard': {'skin_irritation': float, ...},
                'max_concentration': float (%),
            }
        """
        from models.safety_net import SAFETY_CATEGORIES, EU_26_ALLERGENS

        with torch.no_grad():
            # Backbone features (from OdorPredictor)
            graph_batch, bert, phys = self._prepare_inputs(smiles)
            pred_dict = self.odor_models[0](bert, graph_batch, phys,
                                             return_aux=True, return_backbone=True)

            # SafetyNet expects 512d backbone — use real backbone(256d) + fused(768d→256d)
            backbone_feat = pred_dict.get('backbone_256', None)  # [1, 256]
            fused_feat = pred_dict.get('fused_768', None)        # [1, 768]

            if backbone_feat is not None and fused_feat is not None:
                # Concat backbone(256) + first 256 of fused(768) = 512d
                backbone = torch.cat([backbone_feat, fused_feat[:, :256]], dim=-1)
            elif backbone_feat is not None:
                # Pad backbone(256) to 512d
                backbone = torch.zeros(1, 512, device=self.device)
                backbone[0, :256] = backbone_feat.squeeze(0)
            else:
                # Fallback: pad odor vector (legacy behavior)
                odor_vec = pred_dict['odor'].squeeze(0)
                backbone = torch.zeros(1, 512, device=self.device)
                backbone[0, :odor_vec.shape[0]] = odor_vec

            conc = torch.tensor([[concentration]], device=self.device)
            cat_onehot = torch.zeros(1, 15, device=self.device)
            if category in SAFETY_CATEGORIES:
                cat_onehot[0, SAFETY_CATEGORIES.index(category)] = 1

            preds = self.safety_model(backbone, conc, cat_onehot)

            allergen_probs = preds['allergen'].squeeze(0).cpu().numpy()
            allergen_list = [
                (EU_26_ALLERGENS[i], float(allergen_probs[i]))
                for i in range(len(EU_26_ALLERGENS))
                if allergen_probs[i] > 0.3
            ]

            hazard_names = ['skin_irritation', 'sensitization', 'phototoxicity', 'environmental']
            hazard_probs = preds['hazard'].squeeze(0).cpu().numpy()

            return {
                'ifra_violation': preds['ifra_violation'].item(),
                'allergen': allergen_list,
                'hazard': {n: float(hazard_probs[i]) for i, n in enumerate(hazard_names)},
                'max_concentration': preds['max_concentration'].item(),
            }

    def generate_recipe(self, mood=None, season=None, top_k=8, n=1):
        """레시피 생성 (RecipeVAE 사용)

        Args:
            mood: int (0-15) or None
            season: int (0-3) or None
            top_k: 상위 K개 원료 선택
            n: 생성 레시피 수

        Returns:
            list of recipes, each = [{'ingredient': str, 'ratio': float}, ...]
        """
        return self.recipe_model.generate_random(
            n=n, mood_idx=mood, season_idx=season,
            device=self.device, top_k=top_k
        )


# ================================================================
# 기존 odor_engine.py와 연결
# ================================================================

def create_engine(weights_dir='weights/v6', device='cuda', use_v6=True, **kwargs):
    """엔진 팩토리 — v6 또는 기존 엔진 생성

    사용법:
        engine = create_engine()
        vec = engine.encode('CCO')  # 둘 다 동일 인터페이스
    """
    if use_v6:
        return OdorEngineV6(weights_dir=weights_dir, device=device, **kwargs)
    else:
        # Fallback to original OdorGNN
        from odor_engine import OdorGNN
        return OdorGNN(device=device)


if __name__ == '__main__':
    print("=== OdorEngine V6 Integration Test ===")

    engine = OdorEngineV6(weights_dir='weights/v6', device='cpu', use_ensemble=False)

    # 1. encode (기존 인터페이스)
    vec = engine.encode('CCO')
    print(f"\n  encode('CCO'): shape={vec.shape}")
    print(f"    Top dims: ", end='')
    top_idx = np.argsort(vec)[::-1][:5]
    for i in top_idx:
        print(f"{ODOR_DIMENSIONS[i]}={vec[i]:.3f} ", end='')
    print()

    # 2. encode_full (v6 확장)
    full = engine.encode_full('c1ccccc1O')
    print(f"\n  encode_full('phenol'):")
    for k, v in full.items():
        if isinstance(v, np.ndarray):
            print(f"    {k}: shape={v.shape}")
        else:
            print(f"    {k}: {v:.4f}")

    # 3. predict_mixture
    mix = engine.predict_mixture(
        ['CCO', 'c1ccccc1O', 'CC(=O)OC'],
        [30.0, 50.0, 20.0],
        mood=0, season=1,
    )
    print(f"\n  predict_mixture(): shape={mix['mixture'].shape}, synergy={mix['synergy']:.3f}")

    # 4. check_safety
    safety = engine.check_safety('c1ccccc1O', concentration=5.0)
    print(f"\n  check_safety('phenol'): ifra={safety['ifra_violation']:.3f}")

    # 5. generate_recipe
    recipes = engine.generate_recipe(mood=0, season=1, n=1)
    print(f"\n  generate_recipe(): {len(recipes[0])} ingredients")

    print("\n  ✅ ALL INTEGRATION TESTS PASSED!")
