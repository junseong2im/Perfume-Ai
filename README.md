# Perfume AI — Graph Neural Network 기반 AI 조향 시스템

SMILES 분자 구조에서 138차원 향기 벡터를 예측하고, 이를 기반으로 자동 레시피 생성, IFRA 규제 준수 검증, 원가 추정까지 수행하는 엔드투엔드 AI 조향 시스템.

---

## 시스템 구성

```
Perfume-Ai/
├── server/                     # 백엔드 (Python)
│   ├── pom_engine.py           # 핵심 엔진 — 10-model 앙상블 + PairAttentionNet
│   ├── v6_bridge.py            # SafetyNet IFRA 위반 검사
│   ├── sommelier.py            # 조향 API 서버
│   ├── models/                 # GNN 모델 정의
│   │   ├── openpom_ensemble/   # 10-model OpenPOM (AUROC 0.789)
│   │   └── pair_attention/     # PairAttentionNet (AUROC 0.929)
│   ├── data/                   # 학습 데이터 + 검증 데이터
│   │   ├── curated_GS_LF_merged_4983.csv  # 138 향기 라벨
│   │   └── pom_upgrade/        # 보정 모델 (BP, 원가, ODT)
│   └── scripts/                # 데이터 처리 + 검증 스크립트
├── data/
│   └── ingredients.json        # 7,310 향료 성분 데이터베이스
├── js/                         # 프론트엔드
│   └── ingredient-db.js        # 성분 DB 자동 로딩
├── css/                        # 스타일
└── index.html                  # 메인 UI
```

---

## 핵심 모듈

### 1. 분자 향기 예측 (138d Odor Prediction)

SMILES 입력으로부터 138차원 향기 디스크립터 벡터를 예측.

- **모델**: OpenPOM 10-model 앙상블 (MPNN on DGL graphs)
- **AUROC**: 0.789 (scaffold split), 0.929 (PairAttentionNet)
- **검증**: GoodScents 4,622 분자 blind test — Top-5 **91%**

```python
from pom_engine import POMEngine

engine = POMEngine()
engine.load()

# 단일 분자 예측
pred = engine.predict_138d("COc1cc(C=O)ccc1O")  # Vanillin
# → ['vanilla', 'sweet', 'creamy']

# 혼합물 어코드 예측
recipe = [("CC1=CCC(CC1)C(=C)C", 30), ("COc1cc(C=O)ccc1O", 20)]
accord = engine.predict_accord(recipe)
```

### 2. 노트 분류 (Note Classification)

분자량 기반 끓는점 추정으로 Top/Middle/Base 자동 분류.

- **모델**: Ridge 회귀 (9 features, NIST 보정)
- **정확도**: 25/30 (83%) NIST 기준
- **BP MAE**: 16.5 C

### 3. IFRA 규제 준수

IFRA 49th Amendment 기반 사용 제한 자동 검사.

- **커버리지**: 277/453 IFRA 물질 매핑 (CAS → SMILES)
- **금지 물질**: 74종 (max_pct = 0 강제)
- **제한 물질**: 188종 (농도 상한 적용)
- **위반**: 0건

### 4. 원가 추정 (Cost Estimation)

분자 구조로부터 도매가(USD/kg) 추정.

- **모델**: 20-bag Ridge 앙상블 (16 features)
- **학습**: 112 분자 검증 도매가 (Vigon/Penta/Bedoukian 2024)
- **정확도**: Band(저/중/고) 76%, 2배 이내 82%

### 5. 성분 데이터베이스

7,310개 향료 성분의 통합 데이터베이스.

- **출처**: Pyrfume (GoodScents, Leffingwell, FlavorNet, AromaDB, FragranceDB) + IFRA
- **필드**: SMILES, 끓는점, 노트, 가격, IFRA 규제, CAS, 138d 임베딩
- **커버리지**: 99% (SMILES, BP, Note, Price)

---

## 성능 검증 결과

| 모듈 | 테스트 방법 | 점수 | 등급 |
|------|-----------|------|------|
| 향기 예측 (138d) | GoodScents blind 500 Top-5 | 91% | A+ |
| 138d 커버리지 | 200 랜덤 분자 | 100% | A+ |
| 데이터베이스 | 7,310 entries, 99% | 99% | A+ |
| PairAttention | molecule split AUROC | 0.929 | A+ |
| IFRA 규제 | 277/453, 0 위반 | 277/453 | A |
| 노트 분류 | 30 NIST 분자 | 25/30 | A |
| 원가 모델 | 112mol 5-fold CV band | 76% | A- |
| 앙상블 AUROC | scaffold split | 0.789 | A- |

---

## 기술 스택

- **GNN**: OpenPOM (DeepChem + DGL + PyTorch)
- **화학 정보**: RDKit
- **데이터**: Pyrfume, IFRA 49th Amendment, NIST
- **프론트엔드**: Vanilla JS + CSS
- **서버**: Python (Flask)

## 환경 설정

```bash
# Python 3.12 가상환경
python -m venv openpom_env
openpom_env\Scripts\activate  # Windows

# 의존성
pip install torch dgl deepchem rdkit-pypi openpom numpy

# 서버 실행
cd server
python sommelier.py
```

---

## 라이선스

All Rights Reserved. 무단 사용, 복제, 배포를 금지합니다.
자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.
