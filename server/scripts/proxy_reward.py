"""
RLAIF Proxy Reward Model — 가상 관능 평가 MLP
==============================================
698개 레시피의 시뮬레이션 결과 + harmony_score를 사용하여
경량 MLP(3층, ~3K params)를 학습. LLM API 비용 없이
AI가 생성한 레시피의 "관능적 품질"을 즉시 평가.

사용법:
    from scripts.proxy_reward import ProxyRewardModel, train_proxy
    model = train_proxy()
    score = model.predict(feature_vector)  # 0~1
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ProxyRewardModel(nn.Module):
    """가상 관능 평가 MLP (3층, ~3K params)
    
    입력: 40d 레시피 특성 벡터
        - [0:20]  평균 냄새 벡터 (OdorGNN 20d)
        - [20:23] note 비율 (top, middle, base)
        - [23:26] 카테고리 다양성 (citrus, floral, woody 비율)
        - [26]    총 원료 수 (normalized)
        - [27]    총 농도 %  (normalized)
        - [28:30] hedonic_max, hedonic_mean
        - [30:33] longevity, smoothness, receptor_count (normalized)
        - [33:40] 패딩 / 미래 확장
    
    출력: 품질 점수 (0~1) — harmony_score 근사
    """
    
    def __init__(self, input_dim=40, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)
    
    def predict(self, features):
        """numpy array → float score (0~1)"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.forward(x).item()


def extract_recipe_features(recipe, odor_gnn=None):
    """레시피 데이터 → 40d 특성 벡터 추출"""
    features = np.zeros(40, dtype=np.float32)
    
    ings = recipe.get('ingredients', [])
    if not ings:
        return features
    
    # Note 비율
    note_counts = {'top': 0, 'middle': 0, 'base': 0}
    cat_set = set()
    total_pct = 0
    
    for ing in ings:
        note = ing.get('note', 'middle')
        note_counts[note] = note_counts.get(note, 0) + 1
        cat_set.add(ing.get('id', '').split('_')[0])
        total_pct += ing.get('pct', 0)
    
    n_ings = len(ings)
    
    # [20:23] note 비율
    for i, k in enumerate(['top', 'middle', 'base']):
        features[20 + i] = note_counts[k] / max(n_ings, 1)
    
    # [23:26] 카테고리 다양성 (Shannon entropy proxy)
    features[23] = len(cat_set) / max(n_ings, 1)   # 다양성
    features[24] = min(len(cat_set) / 5, 1.0)       # 5가지 이상이면 1.0 
    features[25] = 1.0 if note_counts['top'] > 0 and note_counts['base'] > 0 else 0.5
    
    # [26] 원료 수 (normalized)
    features[26] = min(n_ings / 15, 1.0)
    
    # [27] 총 농도
    features[27] = min(total_pct / 100, 1.0)
    
    # [28:30] Hedonic (estimated from note balance)
    balance = 1.0 - abs(note_counts['top'] - note_counts['base']) / max(n_ings, 1)
    features[28] = balance  # hedonic_max estimate
    features[29] = balance * 0.85  # hedonic_mean estimate
    
    # [30:33] Estimated outputs
    base_ratio = note_counts['base'] / max(n_ings, 1)
    features[30] = 0.3 + base_ratio * 0.7  # longevity estimate
    features[31] = balance  # smoothness estimate
    features[32] = min(n_ings * 15, 200) / 200  # receptor count est
    
    # OdorGNN 벡터가 있으면 [0:20] 채우기
    if odor_gnn is not None:
        try:
            odor_vecs = []
            for ing in ings:
                smiles = _get_smiles_for_id(ing.get('id', ''))
                if smiles:
                    vec = odor_gnn.encode(smiles)
                    if vec is not None:
                        odor_vecs.append(np.array(vec, dtype=np.float32) * ing.get('pct', 1) / 100)
            if odor_vecs:
                features[0:20] = np.mean(odor_vecs, axis=0)[:20]
        except Exception:
            pass
    
    return features


def _get_smiles_for_id(ingredient_id):
    """간단한 ingredient_id → SMILES 매핑 (fallback)"""
    # 주요 원료의 대표 SMILES
    QUICK_MAP = {
        'bergamot': 'CC1=CCC(CC1)C(=C)C',
        'lemon': 'CC1=CCC(CC1)C(=C)C',
        'jasmine': 'CC(=O)OCC1=CC=CC=C1',
        'rose': 'CC(CCC=C(C)C)CO',
        'sandalwood': 'CC1CCC2(CC1)C(CC=C(C)C)C2CO',
        'vanilla': 'COC1=CC(C=O)=CC=C1O',
        'musk': 'CCCCCCCCCCCCCCCC',
        'lavender': 'CC(=CCC=C(C)C)O',
        'cedarwood': 'CC1CCC2(C)C(O)CCC2C1C',
        'patchouli': 'CC1CCC2(C(C1)CCC(C2O)(C)C)C',
        'vetiver': 'CC1CCC(C(C1)C(C)=C)CO',
    }
    return QUICK_MAP.get(ingredient_id)


def train_proxy(data_path=None, epochs=100, lr=0.003):
    """Proxy Reward Model 학습
    
    Returns:
        trained ProxyRewardModel
    """
    if data_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base, 'data', 'recipe_training_data.json')
    
    with open(data_path, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    
    # 특성 추출
    X = []
    y = []
    for recipe in recipes:
        features = extract_recipe_features(recipe)
        score = recipe.get('harmony_score', 0.85)
        X.append(features)
        y.append(score)
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    print(f"[ProxyReward] Training on {len(X)} recipes, scores: {y.min():.2f}~{y.max():.2f}")
    
    model = ProxyRewardModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    # 80/20 split
    n = len(X)
    perm = torch.randperm(n)
    train_n = int(n * 0.8)
    X_train, X_val = X[perm[:train_n]], X[perm[train_n:]]
    y_train, y_val = y[perm[:train_n]], y[perm[train_n:]]
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
            val_mae = (val_pred - y_val).abs().mean()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_MAE={val_mae:.4f}")
    
    # Best model 복원
    if best_state:
        model.load_state_dict(best_state)
    
    # 최종 MAE
    model.eval()
    with torch.no_grad():
        final_pred = model(X_val)
        final_mae = (final_pred - y_val).abs().mean()
    
    print(f"[ProxyReward] Done. Best val MAE: {final_mae:.4f}")
    
    return model


# 싱글톤
_proxy_model = None

def get_proxy_reward():
    """Proxy Reward Model 싱글톤 (학습 or 로드)"""
    global _proxy_model
    if _proxy_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'models', 'proxy_reward.pt')
        
        _proxy_model = ProxyRewardModel()
        if os.path.exists(model_path):
            _proxy_model.load_state_dict(
                torch.load(model_path, map_location='cpu', weights_only=True))
            print(f"[ProxyReward] Loaded from {model_path}")
        else:
            # 즉시 학습
            _proxy_model = train_proxy()
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(_proxy_model.state_dict(), model_path)
            print(f"[ProxyReward] Trained and saved to {model_path}")
    
    return _proxy_model


if __name__ == '__main__':
    print("=== Proxy Reward Model Training ===")
    model = train_proxy()
    
    # 테스트
    test_recipe = {
        'ingredients': [
            {'id': 'bergamot', 'note': 'top', 'pct': 15},
            {'id': 'jasmine', 'note': 'middle', 'pct': 25},
            {'id': 'sandalwood', 'note': 'base', 'pct': 10},
            {'id': 'musk', 'note': 'base', 'pct': 8},
        ],
        'harmony_score': 0.91,
    }
    features = extract_recipe_features(test_recipe)
    predicted = model.predict(features)
    print(f"\nTest: harmony_score=0.91, predicted={predicted:.4f}")
    
    # 모델 저장
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/proxy_reward.pt')
    print("Saved to models/proxy_reward.pt")
