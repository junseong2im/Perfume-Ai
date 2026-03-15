# Perfume AI -- Graph Neural Network 기반 엔드투엔드 AI 조향 시스템

SMILES 분자 구조에서 138차원 향기 벡터를 예측하고, 이를 기반으로 자동 레시피 생성, IFRA 규제 준수 검증, 원가 추정까지 수행하는 AI 조향 시스템의 설계, 구현, 그리고 검증에 관한 기술 보고서.

**검증된 성능**: GoodScents 500분자 blind test Top-5 **91%**, PairAttention AUROC **0.929**, 10-model Ensemble AUROC **0.789** (Scaffold Split)

---

## 목차

1. [연구 배경 및 동기](#1-연구-배경-및-동기)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [Phase 1 -- 공식 OpenPOM 재현](#3-phase-1----공식-openpom-재현)
4. [Phase 2 -- POM 임베딩 추출](#4-phase-2----pom-임베딩-추출)
5. [Phase 3 -- 메가데이터 통합 학습](#5-phase-3----메가데이터-통합-학습)
6. [Phase 4 -- POM Engine v2](#6-phase-4----pom-engine-v2)
7. [Phase 5 -- 실전 조향 시스템 구축](#7-phase-5----실전-조향-시스템-구축)
8. [데이터 엔지니어링](#8-데이터-엔지니어링)
9. [시행착오 기록](#9-시행착오-기록)
10. [최종 성능 검증](#10-최종-성능-검증)
11. [환경 설정](#11-환경-설정)
12. [라이선스](#12-라이선스)

---

## 1. 연구 배경 및 동기

### 1.1 문제 정의

전통적인 조향(Perfumery)은 숙련된 조향사가 수천 개의 원료를 기억하고, 각 원료의 향기 특성과 상호작용을 경험적으로 학습하여 수행하는 장인의 영역이다. 이 과정에는 통상 10년 이상의 수련이 필요하며, 결과물의 품질은 조향사 개인의 역량에 크게 의존한다.

본 연구는 이 과정을 Graph Neural Network(GNN)으로 자동화할 수 있는지를 검증하고, 실용적인 엔드투엔드 시스템으로 구현하는 것을 목표로 한다.

### 1.2 핵심 가설

> 분자의 SMILES 구조를 Message Passing Neural Network(MPNN)으로 인코딩하면, 인간이 인지하는 향기 디스크립터(향의 특성을 기술하는 형용 단어, 예: "floral", "woody", "citrus")를 높은 정확도로 예측할 수 있다.

이 가설은 Lee et al. (2023)의 "Principal Odor Map" 논문에서 영감을 받았으며, 공식 OpenPOM 라이브러리를 기반으로 재현 및 확장하였다.

### 1.3 설계 원칙

1. **Direct Data-Driven Everything**: 하드코딩된 규칙이나 휴리스틱 제거. 모든 파라미터는 과학 문헌과 대규모 후각 데이터셋에서 직접 도출.
2. **Dual Perfumer & Judge Architecture**: Main Perfumer(POM MPNN)가 레시피를 제안하고, Assistant(v6 GNN)가 교차 검증하며, Judge Model이 8개 메트릭으로 최종 평가.
3. **Explainable Olfaction**: 블랙박스 벡터 출력 대신, 심리물리학 법칙(Stevens, Raoult, Cain)에 기반한 구조화된 평가 보고서를 생성.

---

## 2. 시스템 아키텍처

```
[사용자 입력] "Lavender with Honey"
     |
[1. Target Encoder] 자연어 -> 138d 타겟 벡터
     |
[2. POM Engine v2]
     ├── 10-model MPNN 앙상블 (AUROC 0.789, Scaffold Split)
     ├── PairAttentionNet (AUROC 0.929, Molecule Split)
     ├── 256d POM Embedding DB (5,330 molecules)
     └── 138d Odor Descriptor Predictor
     |
[3. Recipe Optimizer]
     ├── L-BFGS-B 경사하강법 (비선형 혼합 제약)
     ├── Stevens' Power Law 농도 가중
     └── Max-pooling 시너지 모델
     |
[4. Safety Filter]
     ├── IFRA 49th Amendment (277/453 물질)
     ├── SafetyNet 모델 (v6_bridge.py)
     └── 금지 물질 74종 차단
     |
[5. Cost Estimator]
     ├── 20-bag Ridge 앙상블 (112 molecules)
     └── 16 분자 기술자 (Molecular Descriptors)
     |
[최종 출력] 레시피 (성분명, SMILES, 비율, 원가, IFRA 준수)
```

### 2.1 프로젝트 구조

```
Perfume-Ai/
├── server/
│   ├── pom_engine.py            # 핵심 엔진 -- 앙상블 로딩, 예측, 캐시
│   ├── v6_bridge.py             # SafetyNet IFRA 검사, 듀얼 모델 브릿지
│   ├── sommelier.py             # API 서버
│   ├── models/
│   │   ├── openpom_ensemble/    # 10-model x 5 checkpoints (AUROC 0.789)
│   │   ├── pair_attention/      # PairAttentionNet (AUROC 0.929)
│   │   └── molecular_engine.py  # MPNN 아키텍처 정의
│   ├── data/
│   │   ├── curated_GS_LF_merged_4983.csv  # 138 향기 라벨
│   │   ├── pom_data/            # Pyrfume 데이터셋
│   │   └── pom_upgrade/         # BP/비용/ODT 보정 모델
│   ├── weights/                 # v6 사전학습 모델 (2x 606MB)
│   └── scripts/                 # 데이터 처리, 검증, 학습 스크립트
├── data/
│   └── ingredients.json         # 7,310 향료 성분 DB
├── js/                          # 프론트엔드
├── css/
└── index.html
```

---

## 3. Phase 1 -- 공식 OpenPOM 재현

### 3.1 목표

Lee et al. (2023) "A Principal Odor Map to Unify Diverse Tasks in Human Olfaction" 논문의 MPNN 앙상블을 공식 GoodScents-Leffingwell(GS-LF) 데이터셋에서 재현하여 검증 가능한 AUROC 벤치마크를 확보.

### 3.2 모델 구성

| Parameter | Value |
|-----------|-------|
| Atom Features | 134d (RDKit) |
| Bond Features | 6d |
| Message Passing | 5 stages (D-MPNN) |
| Readout | Set2Set (2 layers, 3 steps) |
| FFN Hidden | [392, 392] |
| Embedding | 256d |
| Dropout | 0.12 |
| Optimizer | Adam (LR: 1e-4) |
| Labels | 138 odor descriptors |
| Dataset | GS-LF Merged (5,057 molecules) |

### 3.3 6개의 Critical Fix -- AUROC Gap의 원인

공식 논문의 AUROC를 재현하기 위해 발견하고 해결한 6가지 핵심 구현 문제:

**Fix 1: Best Weights Restoration**
D-MPNN은 Epoch 15-25 사이에 peak를 찍고 이후 과적합된다. `model.restore()`를 호출하여 최고 성능 체크포인트를 복원해야 한다.

```python
model.fit(train_ds, nb_epoch=50)
model.restore()  # 이것 없이는 AUROC가 0.05-0.10 하락
```

**Fix 2: True Soft Voting Ensemble**
10개 모델의 확률을 평균한 후 단일 AUROC를 계산. 개별 모델 AUROC의 평균이 아님.

```python
ensemble_preds = np.mean([model_i.predict(test) for model_i in models], axis=0)
auroc = roc_auc_score(y_test, ensemble_preds, average='macro')
```

**Fix 3: Data Split Consistency**
Scaffold Split을 단일 시드로 1회만 수행하고, 모든 앙상블 모델이 동일한 test set을 공유.

**Fix 4: SMILES 정규화 (Data Leakage 방지)**
`CC(=O)O`와 `O=C(O)C`는 동일한 분자(아세트산)지만, 정규화 없이는 train과 test에 동시 출현하여 성능이 과대 추정된다.

```python
from rdkit import Chem
canonical = Chem.MolToSmiles(Chem.MolFromSmiles(raw_smiles))
```

**Fix 5: 매 Epoch 검증**
Overfitting이 빠르므로 매 Epoch마다 validation을 수행하여 정확한 peak를 포착.

**Fix 6: CosineAnnealingLR**
DeepChem의 optimizer를 직접 접근하여 학습률 스케줄러 적용.

```python
for epoch in range(epochs):
    loss = model.fit(train_ds, nb_epoch=1)
    if epoch == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            model._pytorch_optimizer, T_max=epochs)
    scheduler.step()
```

### 3.4 라벨 추출: Regex Word Boundary

단순 부분 문자열 매칭(`"pine" in desc`)은 "pineapple"도 매칭하여 레이블 오염을 일으킨다. Regex word boundary를 사용하여 +602개 추가 라벨을 안전하게 추출.

```python
regex = re.compile(rf'\b{re.escape(task)}\b', re.IGNORECASE)
if regex.search(descriptor_text):
    label[task_idx] = 1.0
```

### 3.5 재현 결과

| Metric | Value |
|--------|-------|
| Test AUROC (Scaffold Split) | **0.7890** |
| Active Tasks | 132+ |
| Training Time (RTX 4060) | 16.9분 (10 models) |
| Per-model Time | ~1.7분 |

---

## 4. Phase 2 -- POM 임베딩 추출

### 4.1 256d Latent Space

10-model 앙상블의 FFN 은닉층에서 256차원 잠재 벡터를 추출. 이 벡터는 분자의 "후각 지문(Olfactory Fingerprint)"으로 기능한다.

### 4.2 전이학습 벤치마크

| Model | Test AUROC | Notes |
|-------|-----------|-------|
| MPNN Ensemble (Soft Voting) | **0.7890** | Baseline |
| POM Embedding + XGBoost | 0.7511 | 강력한 전이학습 |
| POM Embedding + Random Forest | 0.7218 | |
| POM Embedding + MLP | 0.6665 | 과적합 위험 |

### 4.3 임베딩의 실용적 활용

- **유사 원료 검색**: Cosine Similarity로 5,330개 분자 중 가장 유사한 후각 프로필을 가진 원료 탐색
- **혼합물 벡터**: `MixVec = sum(Vec_i * Ratio_i)` -- Weber-Fechner 법칙에 따른 로그 스케일링 적용
- **역설계**: `min ||MixVec - V_target||^2` 최적화로 타겟 향기를 재구성

---

## 5. Phase 3 -- 메가데이터 통합 학습

### 5.1 5개 데이터셋 통합

| Dataset | Size | Key Characteristic |
|---------|------|--------------------|
| Leffingwell | 3,522 | Expert curated, 고품질 |
| GoodScents | 4,565 | CAS 인덱스, 포괄적 |
| FlavorDB | 25,207 | 대용량, 저밀도 라벨 |
| AromaDB | 648 | 특수 방향족 |
| FlavorNet | 373 | 타겟 플레이버 |
| **Total** | **28,101** | 유니크 분자 |

### 5.2 Masked BCE -- 핵심 기술적 기여

이질적인 데이터셋을 통합할 때 누락된 라벨을 `0`(negative)으로 처리하면 학습이 실패한다. DeepChem의 weight matrix `w`를 활용한 Masked Binary Cross-Entropy를 도입.

```python
masks = np.zeros_like(labels)
masks[i, j] = 1.0  # molecule i에 task j 어노테이션이 있는 경우만
dataset = dc.data.NumpyDataset(X=X, y=labels, w=masks)
```

### 5.3 Ensemble Diversity Law -- 핵심 발견

사전학습 체크포인트에서 앙상블 10개를 파인튜닝하면, 모델 간 다양성이 붕괴되어 soft voting의 효과가 감소한다.

| 전략 | AUROC |
|------|-------|
| Scratch-trained 앙상블 | **0.789** |
| Pre-trained -> Fine-tuned 앙상블 | 0.74 |
| Pre-trained 단일 모델 | 0.78 |

결론: **앙상블은 다양한 seed에서 scratch 학습한 모델 조합이 최적**. 동일 체크포인트 파인튜닝은 오히려 성능 저하.

### 5.4 Two-Stage 전략

1. **Stage 1 -- Structural Pre-training**: 28,101 분자 전체에서 Masked BCE로 "화학적 직관" 습득
2. **Stage 2 -- Perceptual Fine-tuning**: 5,061 고품질 GS-LF 분자에서 5x 낮은 LR(5e-5)로 미세 조정

---

## 6. Phase 4 -- POM Engine v2

### 6.1 Persistent Engine Architecture

매 호출마다 모델을 리로드하는 서브프로세스 방식에서, 싱글턴 엔진으로 전환하여 10개 모델(~500MB VRAM)을 상주시킴.

```python
class POMEngine:
    def load(self):
        # 10 ensemble models + PairAttentionNet + 256d embeddings
        # 최초 1회 로딩, LRU 캐시로 반복 예측 최적화

    def predict_138d(self, smiles):
        # 10-model soft voting -> 138d probability vector

    def predict_accord(self, recipe):
        # Stevens' Power Law + Max-pooling synergy
```

### 6.2 PairAttentionNet -- Phase 4의 핵심 돌파구

단일 분자 예측에서 **AUROC 0.929** (Molecule-level split)를 달성한 Pair Attention 메커니즘. 108개 라벨에 대해 분자 쌍의 관계를 학습.

### 6.3 혼합물 어코드 예측의 진화

**v1 -- Naive Averaging (50% 정확도)**
```python
mix_vec = np.mean([predict(smi) * ratio for smi, ratio in recipe])
```

**v2 -- Stevens' Power Law + Max-pooling (92% 정확도)**

Stevens의 심리물리학 법칙에 따르면, 냄새 강도는 농도의 0.5-0.7 제곱에 비례한다:

```
perceived_intensity = concentration ^ 0.6
```

Max-pooling은 혼합물에서 각 어코드 차원의 최대값을 취하여 시너지 효과를 모델링:

```python
def predict_accord(recipe):
    vectors = []
    for smi, ratio in recipe:
        pred = predict_138d(smi)
        weight = (ratio / 100.0) ** 0.6  # Stevens' Power Law
        vectors.append(pred * weight)
    # Max-pooling: 각 차원에서 최강 성분이 지배
    accord = np.max(vectors, axis=0)
    return accord
```

---

## 7. Phase 5 -- 실전 조향 시스템 구축

### 7.1 성분 데이터베이스 확장

| 단계 | 성분 수 | 방법 |
|------|---------|------|
| Wave 1 | 82 | 핵심 천연물 + 대표 합성물 수동 입력 |
| Wave 2 | 296 | Aldehyde, Terpene, Musk 계열 확장 |
| Wave 3 | 477 | Pyrfume 데이터 자동 매핑 |
| Wave 4 | 7,310 | GoodScents + Leffingwell + FlavorDB + AromaDB + FlavorNet + FragranceDB 전수 통합 |

### 7.2 IFRA 규제 통합

IFRA(International Fragrance Association) 49th Amendment 기반 안전 규제 시스템을 구축:

- **CAS --> SMILES 매핑**: 453개 IFRA 규제 물질 중 **277개 (61%) 매핑 성공**
- **금지 물질**: 74종 (GA 제약 조건으로 max_pct = 0 강제)
- **제한 물질**: 188종 (카테고리별 농도 상한 적용)
- **에센셜 오일 가상 조성**: 4종 (Lavender, Tea Tree 등 주요 화학 구성물 대리 분석)
- **검증**: 0건의 위반 -- 금지 물질 완전 차단 확인

### 7.3 끓는점 (Boiling Point) 추정 및 노트 분류

분자의 끓는점은 휘발성과 직결되며, 향수의 Top/Middle/Base 노트 분류의 물리적 근거가 된다.

- **모델**: Ridge 회귀 (9개 분자 기술자)
- **보정**: NIST Chemistry WebBook 30개 분자로 calibration
- **임계값**: Grid search로 최적 threshold 탐색 (Top < 201C, Middle < 269C)
- **정확도**: 25/30 (83%)

### 7.4 원가 추정 모델

향료 원료의 도매가(USD/kg)를 분자 구조로부터 추정:

- **학습 데이터**: 112개 분자의 검증된 도매가 (Vigon International, Penta Manufacturing, Bedoukian Research 2024년 기준)
- **모델**: 20-bag Bagged Ridge 앙상블, 16개 특성
- **특성**: MW, LogP, HBD, HBA, Aromatic Rings, Rotatable Bonds, TPSA, FractionCSP3, Heavy Atoms + interaction terms
- **검증**: 5-fold CV, Band(저/중/고) 정확도 76%, 2배 이내 82%

가격 분포:

| Price Band | Count | Examples |
|-----------|-------|---------|
| $1-5/kg (commodity) | 26 | Ethanol, Acetone, Hexanal |
| $5-15/kg (bulk) | 48 | Benzaldehyde, Octanal, Cyclohexanone |
| $15-40/kg (specialty) | 29 | Geraniol, Linalool, Cinnamaldehyde |
| $40-100/kg (premium) | 5 | gamma-Decalactone, Carvone, Borneol |
| $100+/kg (rare) | 5 | Muscone, Raspberry Ketone |

---

## 8. 데이터 엔지니어링

### 8.1 CAS-CID-SMILES 매핑 체인

후각 데이터셋들은 서로 다른 화학물질 식별자를 사용한다. 이를 통합하는 매핑 체인:

```
CAS Number -> PubChem CID -> Isomeric SMILES -> Canonical SMILES
```

**매핑 손실 경고**: CAS에서 CID로의 매핑 과정에서 하나의 CAS가 다수의 입체이성질체(CID)로 해석될 수 있다. 공식 `molecules.csv`에 존재하는 IsomericSMILES를 우선시.

**API 엔리치먼트**:
```
GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/IsomericSMILES/JSON
```
ThreadPoolExecutor (max 3 workers)로 rate limit 준수, 100건마다 JSON checkpoint 저장.

### 8.2 라벨 통합 전략

- **SMILES 정규화**: 모든 분자를 Canonical Isomeric SMILES로 변환하여 중복 감지 및 data leakage 방지
- **라벨 셋 통합**: 변형 명칭 통합 (예: "Rose" vs "Floral: Rose" -> 공식 138 descriptor list)
- **빈도 가중 평균**: 동일 SMILES에 대한 상충 평가는 연구 표본 크기 기반 가중 평균
- **무기물 필터링**: 향기 없는 용매(H2O)와 featurization 실패 분자(He, Zn, 단순 염) 제거

---

## 9. 시행착오 기록 -- 조향 AI 모델 개발의 여정

조향 AI를 처음 구상한 시점부터 현재의 10-model 앙상블 + PairAttentionNet 시스템에 이르기까지, 수차례의 실패와 방향 전환이 있었다. 이 섹션은 그 과정을 시간순으로 기록한다.

### 9.1 자체 MPNN 구현의 한계 (v1-v5)

**배경**: 프로젝트 초기, GNN 기반 향기 예측을 위해 PyTorch로 자체 Message Passing Neural Network를 구현했다. 분자를 그래프로 변환하고, 노드/엣지 특성을 GRU로 업데이트하는 기본적인 D-MPNN 아키텍처였다.

**결과**: Leffingwell 3,522개 분자 데이터셋에서 AUROC **0.62 ~ 0.68** 수준에 정체. 논문에서 보고한 0.83-0.89에 크게 미달했다.

**원인 분석**:
- Atom feature 차원이 낮았다 (자체 구현 32d vs 공식 134d). RDKit의 풍부한 화학 기술자를 활용하지 못함
- Set2Set readout 대신 단순 mean pooling을 사용하여 그래프 수준 표현력이 부족
- Bond feature를 아예 사용하지 않아 결합 유형(단일/이중/방향족) 정보가 누락
- 학습률 스케줄링 없이 고정 LR로 학습하여 수렴이 불완전

**교훈**: "논문을 읽고 직접 구현"하는 것과 "논문의 성능을 재현"하는 것은 전혀 다른 문제다. 세부 하이퍼파라미터, 특성 설계, 평가 프로토콜의 차이가 AUROC 0.15 이상의 격차를 만든다.

**전환 결정**: v5까지 자체 개선을 시도했으나 0.70 벽을 넘지 못해, 공식 OpenPOM 라이브러리로 전면 전환하기로 결정.

---

### 9.2 OpenPOM 재현의 6가지 벽

**배경**: OpenPOM(Lee et al. 2023)을 설치하고 공식 코드를 실행했으나, 처음에는 AUROC **0.71** 수준에 그쳤다. 논문의 0.83-0.89와 여전히 큰 차이가 있었다.

**원인 1 -- Best Weights 미복원**: D-MPNN은 Epoch 15-25에서 peak를 찍고 이후 급격히 과적합된다. `model.restore()`를 호출하지 않으면 과적합된 최종 가중치로 평가하게 되어 AUROC가 0.05-0.10 하락한다. 이 한 줄을 추가하자 AUROC가 0.71에서 **0.76**으로 상승.

**원인 2 -- 잘못된 앙상블 평균**: 10개 모델의 AUROC를 개별 계산 후 평균내는 것과, 10개 모델의 확률을 먼저 평균 낸 후 단일 AUROC를 계산하는 것은 결과가 다르다. 후자(True Soft Voting)가 정확한 방법이며, 이것만으로 +0.02 향상.

**원인 3 -- Data Leakage**: `CC(=O)O`와 `O=C(O)C`는 동일한 아세트산이지만, SMILES 정규화 없이는 train/test 양쪽에 출현하여 성능이 과대 추정된다. Canonical SMILES 변환을 적용한 후 정직한 AUROC를 측정할 수 있게 되었다.

**원인 4 -- 라벨 오염 ("pine" vs "pineapple")**: 부분 문자열 매칭으로 라벨을 추출하면 "pine"이 "pineapple"에도 매칭되어 라벨이 오염된다. Regex word boundary(`\b`)를 적용하여 +602개의 정확한 라벨을 추가 확보하면서 오염은 제거.

**원인 5 & 6 -- 검증 빈도와 학습률**: 매 Epoch 검증 + CosineAnnealingLR 적용.

**최종 결과**: 6가지 수정을 모두 적용한 후 **AUROC 0.7890** (Scaffold Split) 달성. 자체 MPNN의 0.65에서 0.14 이상 향상되었고, 이 수치는 이후 모든 개선의 기준선(baseline)이 되었다.

---

### 9.3 메가데이터 통합의 역설 -- 더 많은 데이터가 더 나쁜 성능을 만든 사례

**가설**: "데이터가 많을수록 모델이 좋아진다." GoodScents+Leffingwell 5,061개에 FlavorDB 25,207개, AromaDB 648개, FlavorNet 373개를 합쳐 28,101개 분자로 학습하면 AUROC가 크게 향상될 것으로 기대했다.

**첫 시도 -- 단순 병합**: 모든 데이터셋의 라벨을 138 디스크립터에 매핑하고, 누락 라벨을 0(negative)으로 채워 학습. 결과: AUROC **0.58**. 기존 0.79에서 0.21이나 하락.

**원인**: FlavorDB의 25,207개 분자 대부분은 1-2개의 라벨만 가지고 있었다. 나머지 136개 라벨은 "0"(없음)으로 처리되었는데, 이는 모델에게 "이 분자는 136개 향을 갖지 않는다"는 잘못된 정보를 학습시킨 것이다. 라벨 밀도가 0.033에서 0.0126으로 떨어져 gradient signal이 noise에 묻혔다.

**해결 -- Masked BCE**: 누락 라벨을 0이 아닌 "모름(Unknown)"으로 처리. DeepChem의 weight matrix `w`를 활용하여 어노테이션이 있는 라벨만 loss에 반영. 결과: AUROC 0.58에서 **0.73**으로 회복. 하지만 여전히 baseline 0.79에 미달.

**근본 원인**: FlavorDB 같은 대용량-저품질 데이터는 scaffold 다양성은 높이지만, 라벨 noise가 이득을 상쇄한다. 데이터의 "양"보다 "질"이 중요하다는 것을 실증적으로 확인.

---

### 9.4 앙상블 다양성 법칙의 발견 -- 사전학습이 앙상블을 죽이다

**가설**: 28K 분자로 사전학습한 체크포인트에서 10개 모델을 파인튜닝하면, 각 모델이 더 좋은 초기값에서 출발하므로 앙상블 성능이 향상될 것이다.

**결과**:

| 전략 | AUROC |
|------|-------|
| Scratch 10-model 앙상블 | **0.789** |
| Pre-trained 단일 모델 | 0.78 |
| Pre-trained 10-model 앙상블 | **0.74** |

사전학습된 앙상블이 **scratch 앙상블보다 0.05 낮았다.** 더 놀라운 것은 사전학습된 단일 모델(0.78)이 같은 체크포인트의 앙상블(0.74)보다 높다는 점이다.

**원인**: 동일한 사전학습 체크포인트에서 파인튜닝하면, 10개 모델이 loss landscape의 비슷한 지점에 수렴한다. 모델 간 "다양성(diversity)"이 붕괴되어 soft voting의 핵심 원리인 "다양한 관점의 합의"가 작동하지 않는다.

**교훈 (Ensemble Diversity Law)**: 앙상블의 성능은 개별 모델의 정확도뿐 아니라 모델 간의 예측 다양성에 의존한다. 동일 체크포인트 파인튜닝은 다양성을 제거하여 앙상블 이득을 파괴한다. 최적 전략은 서로 다른 random seed에서 독립적으로 scratch 학습하는 것이다.

---

### 9.5 Contrastive Learning 시도와 좌절

**가설**: 유사한 향기를 가진 분자 쌍은 embedding 공간에서 가까이, 다른 향기의 분자 쌍은 멀리 위치하도록 contrastive loss를 추가하면 표현 품질이 향상될 것이다.

**구현**: 기존 ASL(Asymmetric Loss) 학습에 contrastive learning을 결합. Positive pair(같은 향기 라벨 공유)와 Negative pair를 구성하여 InfoNCE loss를 추가.

**결과**: AUROC가 baseline 0.789에서 **0.783**으로 오히려 하락. Contrastive loss가 분류 loss와 경쟁하며 gradient 방향이 충돌한 것으로 분석.

**교훈**: 이미 높은 성능의 모델에 추가 학습 기법을 적용할 때는, 기존 학습 목표와의 간섭을 반드시 확인해야 한다. "좋은 기법의 병합"이 항상 "더 좋은 결과"를 만들지는 않는다.

---

### 9.6 혼합물 향기 예측 -- Naive Averaging의 실패

**문제**: AI 조향의 핵심은 "여러 분자를 섞었을 때 어떤 향이 나는가"를 예측하는 것이다. 가장 직관적인 접근은 각 분자의 138d 벡터를 비율에 따라 가중 평균하는 것이었다.

**결과**: 12개 reference 혼합물(Lavender, Citrus, Rose 등)에 대해 어코드 예측 정확도 **50%**. 동전 던지기 수준.

**원인 분석**: 실제 혼합물의 향기에서는 비선형 현상이 지배적이다:
- **마스킹**: 강한 향이 약한 향을 완전히 덮어버림 (벡터 평균으로 표현 불가)
- **시너지**: 두 분자가 만나 완전히 새로운 향을 생성 (선형 결합에서 나타나지 않음)
- **임계값**: 인간의 후각은 일정 농도 이하에서는 감지 불가 (연속적 평균과 다름)

**해결 -- Stevens' Power Law + Max-pooling**:

Stevens의 심리물리학 법칙에 따르면 감각 강도는 자극 강도의 거듭제곱에 비례한다. 후각의 경우 지수는 약 0.5-0.7이다:

```
perceived = concentration ^ 0.6
```

이를 적용하고, 벡터 평균 대신 각 차원의 최대값을 취하는 Max-pooling으로 전환:

```python
weight = (ratio / 100.0) ** 0.6  # Stevens
accord = np.max(weighted_vectors, axis=0)  # Max-pooling
```

**결과**: 정확도 50%에서 **92%**로 도약. 이 개선은 모델 자체를 바꾸지 않고, 예측값의 후처리 방식만 심리물리학 법칙에 맞추어 변경한 것이다.

---

### 9.7 GoodScents Blind Test -- 0%에서 91%까지

**배경**: 모델의 진짜 성능을 측정하기 위해 GoodScents 데이터를 blind test용으로 사용하기로 결정. 학습에 사용한 분자를 제외하고 500개 분자를 무작위 추출하여 Top-5 정확도를 측정.

**1차 시도 -- 0% 정확도**: GoodScents `behavior.csv`의 `Stimulus` 컬럼이 PubChem CID가 아닌 CAS 번호 형식(예: "100-52-7")이었다. 파서가 이를 숫자 CID로 해석하려 했으므로 하나도 매칭되지 않았다. 또한 `Descriptors` 컬럼이 semicolon으로 구분되어 있었는데(예: "almond;cherry;marzipan"), 이를 단일 문자열로 처리하여 라벨 매칭이 완전 실패.

**2차 시도 -- 매핑 체인 구축**:
```
CAS(behavior.csv) -> PubChem CID(molecules.csv) -> IsomericSMILES -> Canonical SMILES
```
4,622개 entries 중 3,896개의 SMILES를 성공적으로 해석. Semicolon 파서를 구현하여 디스크립터를 올바르게 추출.

**결과**:

| Rank | 정확도 |
|------|--------|
| Top-1 | 65% |
| Top-3 | 86% |
| Top-5 | **91%** |

**의미**: 자체 선정한 15개 분자의 검증(89%)보다 무작위 500개 blind test(91%)가 오히려 높았다. 이는 모델이 특정 분자에 과적합된 것이 아니라, 일반적인 화학 구조-향기 관계를 학습했음을 의미한다.

---

### 9.8 Cost 모델 -- 소규모 데이터의 과적합 함정

**1차 시도**: 향료 원료 가격을 예측하기 위해 19개 분자의 도매가로 다항 회귀 모델을 학습. R^2 = 0.89로 매우 높은 성능이라고 판단.

**현실**: Leave-One-Out Cross-Validation을 수행하자 R^2가 **0.45**로 폭락. 19개 데이터로 다항 특성(2차항, 교호항)까지 포함하면 설명 변수가 데이터 수보다 많아져 완벽한 과적합이 발생했던 것이다.

**2차 시도**: 60개 분자로 확장 + Ridge 회귀로 정규화. R^2 = 0.823, LOOCV R^2 = 0.674. 개선되었지만 여전히 과적합 징후.

**3차 시도 (현재)**: 검증된 도매가 112개 + 20-bag Bagged Ridge 앙상블 + 5-fold CV. Band(저/중/고) 정확도 76%, 2배 이내 82%. R^2는 여전히 낮지만(가격 범위가 $1.5~$200으로 100배 차이), 실용적인 가격 범주 분류 성능은 확보.

**교훈**: 소규모 데이터에서 R^2는 믿을 수 없다. 반드시 교차검증으로 일반화 성능을 확인해야 하며, 극단적 가격 범위에서는 R^2 대신 band accuracy나 within-2x 같은 실용적 메트릭이 더 유의미하다.

---

### 9.9 PairAttentionNet -- 돌파구

**배경**: Scaffold Split AUROC 0.789는 앙상블 크기를 늘려도, 학습률을 조정해도, 데이터를 추가해도 더 이상 올라가지 않았다. GNN 아키텍처 자체의 한계에 도달한 것으로 판단.

**접근**: 분자들 사이의 "관계"를 직접 학습하는 Pair Attention 메커니즘을 도입. 기존 GNN이 각 분자를 독립적으로 인코딩하는 반면, PairAttentionNet은 분자 쌍의 유사성/차이를 attention weight로 학습한다.

**결과**: Molecule-level split에서 AUROC **0.929**. 기존 0.789 대비 0.14 향상. 108개 라벨에 대해 높은 정확도를 보여, 실제 inference에서 PairAttentionNet을 주 모델로 사용하게 되었다.

**단, 주의점**: 이 성능은 molecule-level split(분자 단위의 무작위 분할)에서 측정된 것이다. Scaffold split(화학 골격 기반 분할)에서는 더 낮을 수 있다. 두 지표를 혼동하지 않는 것이 중요하다.

---

### 9.10 성분 DB 확장 -- 82개에서 7,310개까지

**파도 1 (82개)**: 수동으로 선정한 핵심 천연물(Lavender, Rose, Bergamot)과 대표 합성물(Vanillin, Coumarin). 레시피 생성은 가능했지만 선택지가 너무 적어 대부분의 향을 표현할 수 없었다.

**파도 2 (296개)**: Aldehyde, Terpene, Musk 계열을 체계적으로 추가. 향의 다양성은 개선되었으나 여전히 전문 조향사의 팔레트(2,000-3,000종)에 크게 미달.

**파도 3 (477개)**: Pyrfume 데이터베이스에서 자동 매핑을 시도. SMILES가 있는 분자만 추출. 이 시점에서 발견한 문제: 많은 천연 원료(에센셜 오일)는 단일 분자가 아닌 복합 혼합물이라 SMILES로 표현이 불가능. 해결책으로 주요 화학 구성물(예: Lavender Oil -> Linalool + Linalyl Acetate)로 대리 분석하는 "가상 조성" 기법을 도입.

**파도 4 (7,310개)**: GoodScents + Leffingwell + FlavorDB + AromaDB + FlavorNet + FragranceDB를 전수 통합. CAS-CID-SMILES 매핑 체인과 SMILES 정규화를 적용하여 중복 제거. 99%의 분자에 대해 BP, Note, Price, 138d embedding을 계산하여 상용 수준의 성분 DB 구축.

**발견**: 성분 수가 500개를 넘어가면, 레시피 최적화 엔진이 검색 공간에서 길을 잃기 시작했다. 후보 성분의 사전 필터링(cosine similarity > 0.3인 것만 선별)을 도입하여 해결.

---

### 9.11 Fragrantica 검증 실패 -- 산업계와 학술계의 간극

**시도**: Fragrantica.com에서 478개 실제 향수의 성분/노트 정보를 수집하여, AI 시스템의 예측과 비교하는 backtest를 시도.

**결과**: 성분 이름 매칭률이 극히 낮았다. 향수 산업에서는 "Hedione", "Iso E Super", "Ambroxan" 같은 상품명을 사용하는데, 화학 데이터베이스에는 "methyl dihydrojasmonate", "1-(2,3,8,8-tetramethyl-1,2,3,4,5,6,7,8-octahydronaphthalen-2-yl)ethan-1-one" 같은 IUPAC명이 저장되어 있다. 두 체계 사이의 체계적인 매핑 사전이 존재하지 않는다.

**현재 상태**: 미해결. 향수 업계 고유의 상품명-to-CAS 사전 구축이 필요하며, 이는 순수한 공학적 문제라기보다 도메인 지식의 영역이다.

---

## 10. 최종 성능 검증

### 10.1 검증 방법론

모든 점수는 cherry-pick 없는 **blind holdout** 방식으로 측정:

- **Accord**: GoodScents 4,622개 entries에서 CAS->SMILES 해석 가능한 3,896개 중 500개 무작위 샘플링
- **Cost**: 112개 분자 5-fold Cross-Validation (nested bagging)
- **138d**: 7,310개 성분 중 200개 무작위 샘플링
- **Note**: NIST WebBook 30개 분자 calibration
- **IFRA**: Direct count (사실 기반)

### 10.2 최종 점수표

| Module | Test Method | Sample Size | Score | Grade |
|--------|-----------|-------------|-------|-------|
| Accord (138d) | GoodScents blind Top-5 | 500 | **91%** | **A+** |
| Accord (138d) | GoodScents blind Top-3 | 500 | 86% | |
| Accord (138d) | GoodScents blind Top-1 | 500 | 65% | |
| 138d Coverage | Random sampling | 200 | **100%** (99% non-zero) | **A+** |
| Database | Direct count | 7,310 | **99%** complete | **A+** |
| PairAttention | Molecule-level split | Full | **AUROC 0.929** | **A+** |
| IFRA | CAS matching | 453 | **277 matched**, 0 violations | **A** |
| Note/BP | NIST calibration | 30 | **25/30** (83%) | **A** |
| Cost | 5-fold CV band | 112 | **76%** band, 82% within-2x | **A-** |
| AUROC | Scaffold split | Full | **0.789** | **A-** |

### 10.3 평가 기준

| Grade | 기준 |
|-------|------|
| A+ | 90% 이상 또는 AUROC >= 0.90 |
| A | 80% 이상 또는 AUROC >= 0.80 |
| A- | 70% 이상 또는 AUROC >= 0.70 |
| B+ | 60% 이상 또는 AUROC >= 0.60 |

---

## 11. 환경 설정

### 11.1 요구 사항

- Python 3.12.7 (3.13 미지원)
- PyTorch 2.4.1 + CUDA 12.1
- DGL 2.4.0 (cu121)
- GPU: NVIDIA RTX 4060+ (VRAM 8GB+)

### 11.2 설치

```bash
# Python 3.12 가상환경
py -3.12 -m venv openpom_env
openpom_env\Scripts\activate

# PyTorch (CUDA 12.1)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# DGL (CUDA 12.1)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

# Dependencies
pip install deepchem rdkit-pypi openpom numpy scipy scikit-learn
pip install tensorflow-cpu tensorboard markdown

# DGL Graphbolt 패치
python scripts/_patch_dgl.py
```

### 11.3 실행

```bash
# 서버
python server/sommelier.py

# 검증
python server/scripts/honest_validation.py

# 성능 테스트
python server/scripts/full_verification.py
```

---

## 12. 라이선스

Copyright (c) 2024-2026 junseong2im. All Rights Reserved.

이 소프트웨어, 관련 문서, 데이터, 모델 가중치를 포함한 모든 자산은 저작권자의 독점 재산입니다. 무단 사용, 복제, 배포를 금지합니다. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

---

## 참고 문헌

1. Lee, B. K., et al. (2023). "A principal odor map unifies diverse tasks in human olfaction." *Science*, 381(6661).
2. OpenPOM: Official implementation of the Principal Odor Map. https://github.com/BioML-UGD/OpenPOM
3. Pyrfume: A Python library for olfactory data. https://pyrfume.org/
4. IFRA 49th Amendment. International Fragrance Association.
5. Stevens, S. S. (1957). "On the psychophysical law." *Psychological Review*, 64(3).

---

## 13. 데이터 출처

본 프로젝트에서 사용한 모든 데이터, 모델, 외부 자료의 출처를 명시한다.

### 13.1 Pyrfume 후각 데이터셋 (18개)

Pyrfume(https://pyrfume.org/) 라이브러리를 통해 수집한 후각 연구 데이터셋:

| Dataset | 출처 논문/기관 | 데이터 내용 | 사용 목적 |
|---------|-------------|-----------|----------|
| **GoodScents** | The Good Scents Company | 4,622 molecules, CAS, semicolon-separated descriptors | 핵심 학습 데이터, blind test |
| **Leffingwell** | Leffingwell & Associates | 3,522 molecules, expert-curated binary labels | 핵심 학습 데이터 (GS-LF merged) |
| **FlavorDB** | FlavorDB (Moon et al.) | 25,207 molecules, food-grade odorants | 메가데이터 통합 (Phase 3) |
| **AromaDB** | AromaDB | 648 molecules, aromatic compounds | 메가데이터 통합 |
| **FlavorNet** | Arn & Acree, Cornell Univ. | 373 molecules, GC-olfactometry | 메가데이터 통합 |
| **FragranceDB** | FragranceDB | 향료 성분 DB | 성분 라이브러리 확장 |
| **FooDB** | FooDB (The Metabolomics Innovation Centre) | 식품 화합물 데이터 | 분자 다양성 확장 |
| **Dravnieks 1985** | Dravnieks, A. (1985) | 144 molecules, 146 descriptors, 507 panelists | 인간 후각 기준 데이터 |
| **Keller 2012** | Keller & Vosshall, Rockefeller Univ. | 476 molecules, intensity/pleasantness | 심리물리학적 후각 데이터 |
| **Keller 2016** | Keller et al. (2016) | 강도/향미 평가 확장 데이터 | 후각 예측 검증 |
| **Snitz 2013** | Snitz et al. (2013) | 분자 유사성 평가 데이터 | 혼합물 유사도 모델링 |
| **Bushdid 2014** | Bushdid et al. (2014), *Science* | 혼합물 변별 실험 데이터 | 혼합물 예측 평가 |
| **Abraham 2012** | Abraham et al. (2012) | 221 molecules, ODT + solvation descriptors | ODT 보정 모델 학습 |
| **Haddad 2008** | Haddad et al. (2008) | 분자 기술자-후각 관계 | 특성 엔지니어링 참고 |
| **Arctander 1960** | Arctander, S. (1960) | 클래식 향료 참고 문헌 | 성분 분류 참고 |
| **Arshamian 2022** | Arshamian et al. (2022) | 범문화적 후각 인지 연구 | 디스크립터 보편성 검증 |
| **IFRA 2019** | International Fragrance Assoc. (2019) | 453 규제 물질, CAS, 카테고리별 한도 | IFRA 안전 규제 시스템 |
| **FreeSolv** | Mobley et al. | 수화 자유에너지 데이터 | 물리화학 특성 참고 |

### 13.2 외부 컬렉션 데이터

| Dataset | 출처 | 데이터 내용 | 사용 목적 |
|---------|------|-----------|----------|
| **Osmo POM (Lee 2023)** | Lee et al. / Osmo Inc. | curated_GS_LF_merged_4983.csv (138 labels) | 공식 GS-LF 학습 데이터 |
| **Osmo Taxonomy** | Osmo Inc. | 22d scent taxonomy mapping | 138d -> 22d 산업 표준 변환 |
| **Fragrantica** | Fragrantica.com | 478+ 실제 향수 성분/노트 정보 | Backtest 검증 |
| **DREAM Mixture** | DREAM Challenge 2015 | 혼합물 향기 예측 도전 과제 데이터 | 혼합물 모델 학습 |
| **Zenodo** | Zenodo Open Repository | 보충 후각 데이터셋 | 데이터 확장 |

### 13.3 화학 데이터베이스 및 API

| 출처 | URL | 사용 목적 |
|------|-----|----------|
| **PubChem** | https://pubchem.ncbi.nlm.nih.gov/ | CAS -> CID -> SMILES 매핑, 분자 정보 |
| **PubChem PUG REST API** | https://pubchem.ncbi.nlm.nih.gov/rest/pug/ | 분자 식별자 해석, 물성 조회 |
| **NIST Chemistry WebBook** | https://webbook.nist.gov/ | 끓는점 calibration (30 molecules) |
| **RDKit** | https://www.rdkit.org/ | 134d atom features, 분자 기술자 계산 |

### 13.4 모델 및 프레임워크

| 모델/프레임워크 | 출처 | 버전 | 사용 목적 |
|-------------|------|------|----------|
| **OpenPOM** | https://github.com/BioML-UGD/OpenPOM | 1.0.0 | MPNN 앙상블 아키텍처 |
| **DeepChem** | https://deepchem.io/ | 2.5.0 | 모델 학습/추론 프레임워크 |
| **DGL (Deep Graph Library)** | https://www.dgl.ai/ | 2.4.0 (cu121) | 그래프 신경망 백엔드 |
| **PyTorch** | https://pytorch.org/ | 2.4.1 (cu121) | 딥러닝 프레임워크 |
| **ChemBERTa** | Chithrananda et al. (2020) | LoRA v22 | 분자 임베딩 (보조) |

### 13.5 가격 데이터 출처

원가 추정 모델의 학습에 사용한 도매가 데이터 (112 molecules):

| 출처 | 데이터 범위 | 비고 |
|------|-----------|------|
| **Vigon International** | 향료 원료 도매가 | 2024 catalogue, 100kg+ lots |
| **Penta Manufacturing** | 합성 방향제 가격 | Fine chemicals supplier |
| **Bedoukian Research** | Terpene/specialty 가격 | Aroma chemicals, terpenoids |
| **PerfumersWorld** | 향료 소매/교육용 가격 | 가격 범위 참고 |
| **Creating Perfume Guide** | 산업 평균 가격 | 가격 band 분류 기준 |

### 13.6 규제 데이터

| 출처 | 내용 | 적용 |
|------|------|------|
| **IFRA 49th Amendment** (2019) | 453 물질, 11 카테고리, 금지/제한 한도 | SafetyNet 규제 필터 |
| **REACH (EU)** | 화학물질 등록/평가 규정 | 규제 참조 (직접 미사용) |

### 13.7 참고 논문

| No. | 논문 | 인용 |
|-----|------|------|
| 1 | Lee, B. K., et al. (2023) | "A principal odor map unifies diverse tasks in human olfaction." *Science*, 381(6661) |
| 2 | Dravnieks, A. (1985) | "Atlas of Odor Character Profiles." ASTM Data Series 61 |
| 3 | Keller, A. & Vosshall, L. B. (2016) | "Olfactory perception of chemically diverse molecules." *BMC Neuroscience*, 17(1) |
| 4 | Bushdid, C. et al. (2014) | "Humans can discriminate more than 1 trillion olfactory stimuli." *Science*, 343(6177) |
| 5 | Snitz, K. et al. (2013) | "Predicting odor perceptual similarity from odor structure." *PLoS Computational Biology* |
| 6 | Abraham, M. H. et al. (2012) | "An algorithm for 353 odor detection thresholds in humans." *Chemical Senses*, 37(3) |
| 7 | Stevens, S. S. (1957) | "On the psychophysical law." *Psychological Review*, 64(3) |
| 8 | Haddad, R. et al. (2008) | "A metric for odorant comparison." *Nature Methods*, 5(5) |
| 9 | Arshamian, A. et al. (2022) | "The perception of odor pleasantness is shared across cultures." *Current Biology* |
| 10 | Chithrananda, S. et al. (2020) | "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction." *arXiv* |
| 11 | Moon, S. et al. (2020) | "FlavorDB2: An Updated Database of Flavor Molecules." *Frontiers in Nutrition* |

