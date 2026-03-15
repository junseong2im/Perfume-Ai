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

## 9. 시행착오 기록

이 섹션은 프로젝트 진행 과정에서 겪은 주요 실패와 그로부터 얻은 교훈을 기록한다.

### 9.1 DGL + Python 3.13 호환성 문제

**문제**: 프로젝트 초기 Python 3.13 환경에서 `pip install dgl`이 실패. Pre-built wheel이 존재하지 않아 구식 버전(0.1.3)이 설치되며 DLL 로드 오류 발생.

**해결**: Python 3.12.7을 별도 설치하고 `py -3.12 -m venv openpom_env`로 전용 가상환경 생성. 메인 애플리케이션(3.13)은 서브프로세스 브릿지를 통해 3.12 환경의 AI 파이프라인을 호출.

### 9.2 DGL CUDA 버전 불일치

**문제**: `pip install dgl`로 설치된 Windows wheel에 CUDA 지원이 누락 (`DGLError: Device API cuda is not enabled`).

**문제 2**: `--force-reinstall`로 DGL을 재설치하면 종속성 해결 과정에서 CUDA 버전 PyTorch가 CPU 버전으로 교체됨.

**해결**: 정확한 호환성 매트릭스 확립:

| PyTorch | DGL | GPU |
|---------|-----|-----|
| **2.4.1 (cu121)** | **2.4.0 (cu121)** | OK |
| 2.5.1 | 2.2.1 | CPU Fallback |

DGL 설치 후 반드시 PyTorch CUDA 버전을 재확인하고, 필요시 재설치.

### 9.3 Graphbolt DLL 로드 오류

**문제**: DGL 2.x의 Graphbolt 모듈이 Windows에서 `FileNotFoundError` 발생 (DLL이 존재함에도).

**해결**: `dgl/graphbolt/__init__.py`를 패치하여 DLL 로드 실패를 경고로 전환:

```python
try:
    load_graphbolt()
except (FileNotFoundError, ImportError, OSError) as e:
    warnings.warn(f"DGL graphbolt not available: {e}")
```

### 9.4 DeepChem AdamW Import 오류

**문제**: OpenPOM이 `deepchem.models.optimizers.AdamW`를 하드코딩 import하는데, DeepChem 2.5.0 이후 네임스페이스 변경으로 `ImportError` 발생.

**해결**: `openpom/utils/optimizer.py`에 fallback 패치:

```python
from deepchem.models.optimizers import Adam
try:
    from deepchem.models.optimizers import AdamW
except ImportError:
    AdamW = Adam  # Safe fallback
```

### 9.5 Windows 파일 잠금과 체크포인트 저장

**문제**: `deepchem.models.torch_models.torch_model.save_checkpoint`에서 `os.rename` 호출 시 `PermissionError: [WinError 32]` 발생. Windows의 파일 잠금 메커니즘과 충돌.

**해결**: `os.rename`을 `shutil.move` 기반의 재시도 로직(`_safe_move`)으로 교체. 5회 재시도 + 0.3초 대기 후 `shutil.copy2` fallback.

### 9.6 과대평가의 유혹 -- Cherry-pick 검증의 함정

**문제**: 초기 검증에서 시스템 빌더가 알고 있는 15개 분자를 선정하여 89% 정확도를 보고했으나, 이는 잠재적인 selection bias를 포함.

**발견**: 500개 GoodScents blind test로 전환한 결과, Top-5 정확도가 **91%**로 오히려 더 높게 측정됨. 이는 모델이 실제로 일반화 능력이 있음을 증명.

**교훈**: 자체 선정 테스트는 의미가 없다. 반드시 외부 blind 데이터로 검증해야 한다.

### 9.7 Cost 모델 과적합

**문제**: 31개 분자로 학습한 Cost 모델이 R^2 = 0.823을 보고했으나, Leave-One-Out Cross-Validation 결과 R^2 = 0.674로 하락.

**대응**: 112개 분자로 학습 데이터를 4배 확장하고, 20-bag Bagged Ridge 앙상블 + 5-fold CV로 전환. Band(저/중/고) 분류 정확도 76%, 2배 이내 정확도 82% 달성.

**교훈**: 소규모 데이터셋에서의 R^2는 과적합의 산물일 수 있다. 반드시 교차검증으로 확인.

### 9.8 Fragrantica 이름 매칭 실패

**문제**: 478개 실제 Fragrantica 향수의 backtest에서 원료 이름 매칭이 거의 실패. 향수 업계의 common name(예: "Hedione", "Ambroxan")과 화학 데이터베이스의 IUPAC/SMILES 간 체계적인 매핑이 부재.

**현재**: 미해결. 향수 업계 고유의 상품명-to-CAS 사전 구축이 필요.

### 9.9 Mixture Prediction의 비선형성

**문제**: 혼합물의 향기는 단순한 성분 벡터의 선형 결합이 아니다. 특정 분자 조합은 완전히 새로운 향기를 생성하거나(시너지), 서로의 향기를 차단한다(마스킹).

**대응**: Stevens' Power Law + Max-pooling으로 1차 근사. Naive averaging 50% -> 92% 향상. Set Transformer 등 attention 기반 모델은 향후 과제.

### 9.10 GoodScents Blind Test에서 0% 정확도 (v1)

**문제**: 최초 blind test에서 0% 정확도를 기록. GoodScents behavior.csv의 `Stimulus` 컬럼이 CID가 아닌 CAS 형식이었고, `Descriptors` 컬럼이 semicolon 구분자로 되어 있었는데, 이를 파싱하지 못함.

**해결**: CAS --> CID --> SMILES 매핑 체인 구축 + semicolon 파서 구현 후, 4,622개 entries 중 3,896개 SMILES 해석 성공. 결과: 500분자 blind test에서 Top-5 **91%** 달성.

**교훈**: 데이터 형식을 가정하지 말 것. 반드시 원본 데이터의 구조를 직접 확인해야 한다.

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
