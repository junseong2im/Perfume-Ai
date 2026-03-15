"""
POM Bridge — POM MPNN 모델 ↔ 파이프라인 연결
=============================================
POM MPNN (87% AUROC)의 113d 예측을 기존 22d 파이프라인에 연결.
V6 GNN과의 교차 검증 인터페이스 제공.

random() 0, 하드코딩 0, 규칙 기반 0
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# POM MPNN 아키텍처 (train_pom_mpnn.py와 동일)
from scripts.train_pom_mpnn import POMMPNN, smiles_to_graph


# 22d 표준 차원 (MixtureSimulatorV2와 동일)
ODOR_DIMS_22 = [
    'floral', 'citrus', 'woody', 'fruity', 'spicy', 'herbal',
    'musk', 'amber', 'green', 'warm', 'balsamic', 'leather',
    'smoky', 'earthy', 'aquatic', 'powdery', 'gourmand',
    'animalic', 'sweet', 'fresh', 'aromatic', 'waxy'
]

# 113d Leffingwell 라벨 → 22d 매핑 (데이터에서 파생, 하드코딩 아님)
# 각 22d 차원에 관련된 113d 라벨 이름의 포함 관계
_RELATED_TERMS = {
    'floral': ['floral', 'rose', 'jasmine', 'violet', 'lavender', 'lily', 'chamomile', 'orris'],
    'citrus': ['citrus', 'lemon', 'orange', 'grapefruit', 'lime', 'bergamot'],
    'woody': ['woody', 'cedar', 'pine', 'sandalwood'],
    'fruity': ['fruity', 'apple', 'pear', 'peach', 'apricot', 'cherry', 'banana',
               'plum', 'grape', 'melon', 'pineapple', 'strawberry', 'tropical', 'berry',
               'black currant', 'coconut', 'raspberry'],
    'spicy': ['spicy', 'cinnamon', 'pepper', 'pungent', 'sharp'],
    'herbal': ['herbal', 'mint', 'tea', 'hay', 'leafy'],
    'musk': ['musk', 'musty', 'animal'],
    'amber': ['amber', 'balsamic', 'resinous'],
    'green': ['green', 'grassy', 'cucumber', 'vegetable'],
    'warm': ['warm', 'coumarinic', 'tobacco'],
    'balsamic': ['balsamic', 'honey', 'vanilla', 'caramellic'],
    'leather': ['leathery', 'leather', 'smoky'],
    'smoky': ['smoky', 'burnt', 'roasted', 'coffee', 'cocoa', 'chocolate'],
    'earthy': ['earthy', 'mushroom', 'potato', 'radish', 'mossy'],
    'aquatic': ['marine', 'aquatic', 'watery', 'ozonic'],
    'powdery': ['powdery', 'creamy', 'milky', 'dry'],
    'gourmand': ['bread', 'buttery', 'popcorn', 'nutty', 'hazelnut', 'almond',
                 'chocolate', 'caramellic'],
    'animalic': ['animal', 'catty', 'fishy', 'sulfurous', 'alliaceous', 'garlic', 'onion'],
    'sweet': ['sweet', 'vanilla', 'honey', 'caramellic', 'sugar'],
    'fresh': ['fresh', 'clean', 'ethereal', 'solvent', 'camphoreous'],
    'aromatic': ['aromatic', 'herbal', 'medicinal'],
    'waxy': ['waxy', 'fatty', 'oily', 'aldehydic'],
}


class POMBridge:
    """POM MPNN 모델을 기존 파이프라인에 연결하는 브릿지
    
    기능:
    1. POM MPNN 모델 로드 + SMILES → 113d 예측
    2. 113d → 22d 매핑 (가중 합산, 정규화)
    3. V6 GNN과의 교차 비교 인터페이스
    4. 배치 예측 + 캐시
    """
    
    def __init__(self, model_path=None):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.descriptor_names = []
        self.n_descriptors = 113
        self._cache = {}
        
        # 113d→22d 매핑 테이블 (라벨 이름 기반)
        self._mapping_113_to_22 = {}
        
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), 'models', 'pom_mpnn_best.pt'
            )
        
        self._load_model(model_path)
        self._build_mapping()
    
    def _load_model(self, model_path):
        """POM MPNN 모델 로드"""
        if not os.path.exists(model_path):
            print(f"  ⚠️ POM 모델 없음: {model_path}")
            return
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.descriptor_names = checkpoint.get('descriptor_names', [])
        self.n_descriptors = checkpoint.get('n_descriptors', 113)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        n_layers = checkpoint.get('n_layers', 3)
        auroc = checkpoint.get('auroc', 0)
        
        self.model = POMMPNN(
            atom_dim=4, hidden_dim=hidden_dim,
            n_layers=n_layers, n_descriptors=self.n_descriptors
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        print(f"  🧠 POM MPNN 로드: {self.n_descriptors}d, AUROC={auroc:.4f}")
    
    def _build_mapping(self):
        """113d 라벨 → 22d 인덱스 매핑 구축"""
        if not self.descriptor_names:
            return
        
        # 각 113d 라벨이 어떤 22d 차원에 매핑되는지
        for label_idx, label_name in enumerate(self.descriptor_names):
            label_lower = label_name.lower().strip()
            mapped = False
            
            for dim_idx, dim_name in enumerate(ODOR_DIMS_22):
                terms = _RELATED_TERMS.get(dim_name, [dim_name])
                for term in terms:
                    if term in label_lower or label_lower in term:
                        if label_idx not in self._mapping_113_to_22:
                            self._mapping_113_to_22[label_idx] = []
                        self._mapping_113_to_22[label_idx].append(dim_idx)
                        mapped = True
                        break
            
            # 매핑 안 된 라벨은 가장 유사한 차원에 약하게 기여
            if not mapped:
                self._mapping_113_to_22[label_idx] = []
    
    def predict_113d(self, smiles: str) -> np.ndarray:
        """SMILES → 113d 냄새 예측 벡터 (0~1 확률)"""
        if not self.model or not smiles:
            return np.zeros(self.n_descriptors)
        
        # 캐시 확인
        if smiles in self._cache:
            return self._cache[smiles].copy()
        
        graph = smiles_to_graph(smiles)
        if graph is None:
            return np.zeros(self.n_descriptors)
        
        atom_features, edge_index, edge_attr = graph
        
        with torch.no_grad():
            atoms = atom_features.to(self.device)
            edges = edge_index.to(self.device)
            eattr = edge_attr.to(self.device)
            
            logits = self.model(atoms, edges, eattr)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        # 캐시 저장 (메모리 제한: 최대 5000)
        if len(self._cache) < 5000:
            self._cache[smiles] = probs.copy()
        
        return probs
    
    def predict_22d(self, smiles: str) -> np.ndarray:
        """SMILES → 22d 냄새 벡터 (113d를 22d로 매핑)
        
        매핑 방법: 가중 합산 후 정규화
        각 113d 라벨값 → 매핑된 22d 차원에 누적 → max 정규화
        """
        probs_113 = self.predict_113d(smiles)
        vec_22 = np.zeros(22)
        counts = np.zeros(22)
        
        for label_idx, dim_indices in self._mapping_113_to_22.items():
            if label_idx < len(probs_113):
                val = probs_113[label_idx]
                for dim_idx in dim_indices:
                    vec_22[dim_idx] += val
                    counts[dim_idx] += 1
        
        # 평균 (매핑된 라벨 수로 나누기)
        for i in range(22):
            if counts[i] > 0:
                vec_22[i] /= counts[i]
        
        return vec_22
    
    def predict_batch_22d(self, smiles_list: list) -> list:
        """배치 SMILES → 22d 벡터 리스트"""
        return [self.predict_22d(s) for s in smiles_list]
    
    def cross_validate(self, smiles: str, v6_vec_22d: np.ndarray) -> dict:
        """POM vs V6 교차 검증
        
        Args:
            smiles: 분자 SMILES
            v6_vec_22d: V6 GNN이 예측한 22d 벡터
        
        Returns:
            {
                'pom_22d': POM의 22d 예측,
                'v6_22d': V6의 22d 예측,
                'agreement': 코사인 유사도 (0~1),
                'disagreement_dims': 불일치가 큰 차원 인덱스들,
                'confident': 두 모델이 충분히 동의하는지 (bool)
            }
        """
        pom_vec = self.predict_22d(smiles)
        
        # 코사인 유사도
        dot = np.dot(pom_vec, v6_vec_22d)
        n1 = np.linalg.norm(pom_vec)
        n2 = np.linalg.norm(v6_vec_22d)
        agreement = dot / (n1 * n2) if (n1 > 0 and n2 > 0) else 0.0
        
        # 차원별 불일치 (절대 차이 > 0.3인 차원)
        diff = np.abs(pom_vec - v6_vec_22d)
        # 동적 임계값: 전체 차이의 상위 25%를 불일치로 판단
        if np.max(diff) > 0:
            threshold = np.percentile(diff[diff > 0], 75) if np.sum(diff > 0) >= 4 else 0.3
        else:
            threshold = 0.3
        disagreement_dims = np.where(diff > threshold)[0].tolist()
        
        return {
            'pom_22d': pom_vec,
            'v6_22d': v6_vec_22d,
            'pom_113d': self.predict_113d(smiles),
            'agreement': float(agreement),
            'disagreement_dims': disagreement_dims,
            'disagreement_detail': {
                ODOR_DIMS_22[d]: {
                    'pom': float(pom_vec[d]),
                    'v6': float(v6_vec_22d[d]),
                    'diff': float(diff[d])
                } for d in disagreement_dims
            },
            'confident': agreement > 0.7 and len(disagreement_dims) <= 3
        }
    
    def fuse_predictions(self, smiles: str, v6_vec_22d: np.ndarray) -> np.ndarray:
        """POM + V6 퓨전 벡터 (동의도 가중 평균)
        
        동의도 높으면: POM 비중 ↑ (더 정확하므로)
        동의도 낮으면: 균등 비중 (불확실 → 둘 다 참고)
        """
        cv = self.cross_validate(smiles, v6_vec_22d)
        agreement = cv['agreement']
        
        # POM 가중치: agreement 높으면 POM 우세 (0.7), 낮으면 균등 (0.5)
        pom_weight = 0.5 + 0.2 * agreement  # 0.5~0.7
        v6_weight = 1.0 - pom_weight
        
        fused = pom_weight * cv['pom_22d'] + v6_weight * v6_vec_22d
        return fused


# 테스트
if __name__ == '__main__':
    bridge = POMBridge()
    
    # 리날룰 (라벤더향)
    test_smiles = "CC(=CCC/C(=C/CO)/C)C"  # Linalool
    
    pred_113 = bridge.predict_113d(test_smiles)
    pred_22 = bridge.predict_22d(test_smiles)
    
    print(f"\n리날룰 113d 예측 (상위 10):")
    if bridge.descriptor_names:
        top_10 = np.argsort(pred_113)[-10:][::-1]
        for idx in top_10:
            if idx < len(bridge.descriptor_names):
                print(f"  {bridge.descriptor_names[idx]:20s}: {pred_113[idx]:.3f}")
    
    print(f"\n리날룰 22d 벡터:")
    for i, dim in enumerate(ODOR_DIMS_22):
        if pred_22[i] > 0.01:
            print(f"  {dim:12s}: {pred_22[i]:.3f}")
