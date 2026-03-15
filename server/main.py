# main.py — AI Perfumery Production Backend
# =============================================
import json, asyncio, time, logging, os
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from models import (MolecularEngine, GeometricGNN, CompetitiveBinding,
                     ActiveLearner, SensorEngine, NeuralNet, VAEGenerator)
import database as db
import recipe_engine
import molecular_harmony as mh
import biophysics_simulator as biophys
import odor_engine
import sommelier

# ===== Structured Logging =====
log = logging.getLogger("perfumery")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ===== 전역 상태 =====
DATA_DIR = Path(__file__).parent.parent / 'data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 엔진 인스턴스
molecular = MolecularEngine(device)
gnn3d = GeometricGNN(device)
binding = CompetitiveBinding(device)
active_learner = ActiveLearner(device)
sensor = SensorEngine(device)
neural_net = None
vae = None

# 향료 DB (JS 규칙 엔진 대체)
ingredient_db = []
harmony_rules = {}
primary_odors = None
level = 1


def load_json(name):
    p = DATA_DIR / name
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


# ===== Model Singletons (preloaded once) =====
_cached_v5_model = None
_cached_v5_device = None


def _get_v5_model():
    """Get cached v5 GATConv model (singleton — loaded once)"""
    global _cached_v5_model, _cached_v5_device
    if _cached_v5_model is not None:
        return _cached_v5_model, _cached_v5_device
    
    from models.odor_gat_v5 import OdorGATv5
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v5_path = Path(__file__).parent / 'weights' / 'odor_gnn_v5.pt'
    if not v5_path.exists():
        return None, dev
    
    cp = torch.load(v5_path, map_location=dev, weights_only=False)
    model = OdorGATv5(bert_dim=384).to(dev)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    _cached_v5_model = model
    _cached_v5_device = dev
    log.info("v5 GATConv model loaded and cached")
    return model, dev


# ===== Lifespan (replaces deprecated on_event) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production startup/shutdown lifecycle"""
    global ingredient_db, harmony_rules, primary_odors
    log.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # DB 로딩
    try:
        stats = db.get_db_stats()
        ingredient_db = db.get_all_ingredients()
        log.info(f"DB connected: {stats['molecules']:,} molecules, {stats['ingredients']} ingredients")
    except Exception as e:
        log.warning(f"DB connection failed, JSON fallback: {e}")
        ingredient_db = load_json('ingredients.json')
    
    harmony_rules = load_json('accords.json')
    primary_odors = load_json('primary-odors.json')
    molecular.load(str(DATA_DIR))
    gnn3d.load(str(DATA_DIR))
    
    # Preload v5 model (singleton cache)
    _get_v5_model()
    
    log.info("Server ready on http://localhost:8001")
    yield  # App runs here
    log.info("Server shutting down")


app = FastAPI(title="AI Perfumery Engine", lifespan=lifespan, version="2.0")

# CORS — production whitelist
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ===== Global Error Handler =====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ===== Pydantic 모델 =====
class PerfumeRequest(BaseModel):
    mood: str
    season: str
    preferences: list[str] = []
    intensity: int = 50

class RecipeRequest(BaseModel):
    mood: str = 'romantic'
    season: str = 'spring'
    preferences: list[str] = []
    intensity: int = 50
    complexity: Optional[int] = None
    batch_ml: int = 100

class LatentRequest(BaseModel):
    x: float
    y: float

class MoleculeRequest(BaseModel):
    id: Optional[str] = None
    smiles: Optional[str] = None

class MixtureRequest(BaseModel):
    moleculeIds: list[str]

class TextRequest(BaseModel):
    text: str


# ===== 간단한 규칙 기반 조향 =====
def formulate(mood, season, prefs):
    """향료 DB 기반 결정론적 조향 (random 제거)"""
    if not ingredient_db:
        return []
    scored = []
    for ing in ingredient_db:
        score = 0
        cat = ing.get('category', '')
        if cat in prefs: score += 3
        moods = ing.get('moods', [])
        if mood in moods: score += 2
        seasons = ing.get('seasons', [])
        if season in seasons: score += 1
        # 결정론적 타이브레이커: 원료 ID 해시 → 소수점 이하 값
        tiebreaker = (hash(ing.get('id', '')) % 1000) / 2000.0  # 0~0.5 결정론적
        score += tiebreaker
        scored.append((ing, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = scored[:10]  # 결정론적으로 상위 10개 선택
    total = sum(s for _, s in selected) or 1
    return [{'id': ing['id'], 'percentage': round(s / total * 100, 1)} for ing, s in selected]


# ===== 엔드포인트 =====
@app.get("/api/health")
def health():
    try:
        stats = db.get_db_stats()
    except Exception:
        stats = {}
    return {
        "status": "ok",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "level": level,
        "ingredients": len(ingredient_db),
        "molecules_json": len(molecular.molecules),
        "db": stats
    }


@app.get("/api/data/ingredients")
def get_ingredients():
    return ingredient_db


# ===== DB 쿼리 엔드포인트 =====
@app.get("/api/db/molecules")
def db_get_molecules(limit: int = 100, offset: int = 0):
    """DB에서 분자 목록 (페이지네이션)"""
    mols = db.get_all_molecules(limit=limit)
    return {"molecules": mols, "total": db.get_db_stats()['molecules']}


@app.get("/api/db/molecules/search")
def db_search_molecules(q: str, limit: int = 20):
    """분자 검색 (이름/SMILES)"""
    return db.search_molecules(q, limit)


@app.get("/api/db/molecules/by-odor/{odor}")
def db_molecules_by_odor(odor: str, limit: int = 50):
    """특정 냄새 descriptor로 분자 필터"""
    return db.get_molecules_by_odor(odor, limit)


@app.get("/api/db/descriptors")
def db_get_descriptors():
    """전체 냄새 descriptor + 분자 수"""
    return db.get_all_descriptors()


@app.get("/api/db/stats")
def db_stats():
    """DB 통계"""
    return db.get_db_stats()


# ===== 고급 레시피 생성 엔드포인트 =====
@app.post("/api/recipe/generate")
def generate_recipe(req: RecipeRequest):
    """DB 489개 원료 기반 향수 레시피 생성"""
    result = recipe_engine.generate_recipe(
        mood=req.mood,
        season=req.season,
        preferences=req.preferences,
        intensity=req.intensity,
        complexity=req.complexity,
        batch_ml=req.batch_ml
    )
    return result


@app.post("/api/recipe/variations")
def generate_variations(req: RecipeRequest):
    """기본 레시피 + 3가지 변형 생성"""
    base = recipe_engine.generate_recipe(
        mood=req.mood,
        season=req.season,
        preferences=req.preferences,
        intensity=req.intensity,
        complexity=req.complexity,
        batch_ml=req.batch_ml
    )
    variations = recipe_engine.generate_variations(base, count=3)
    return {'base': base, 'variations': variations}


@app.get("/api/recipe/moods")
def get_available_moods():
    """사용 가능한 무드 목록"""
    return {
        'moods': list(recipe_engine.MOOD_CATEGORIES.keys()),
        'seasons': ['spring', 'summer', 'autumn', 'winter'],
        'styles': list(recipe_engine.NOTE_RATIOS.keys()),
    }


@app.post("/api/recipe/train")
def train_recipe_ai(epochs: int = 50):
    """AI 레시피 모델 수동 학습"""
    recipe_engine.train_ai(epochs=epochs)
    return {
        'status': 'trained',
        'device': str(recipe_engine._engine.device),
        'epochs': recipe_engine._engine.train_epochs,
        'loss': round(recipe_engine._engine.train_loss, 4),
    }


@app.get("/api/recipe/clones")
def get_clone_list():
    """사용 가능한 클론 포뮬러 목록"""
    return {'clones': recipe_engine.list_clones()}


@app.post("/api/recipe/clone")
def generate_clone_recipe(clone_id: str, batch_ml: int = 100):
    """유명 향수 클론 레시피 생성 (제작 레시피 포맷)"""
    result = recipe_engine.clone_recipe(clone_id=clone_id, batch_ml=batch_ml)
    if 'error' in result:
        return JSONResponse(status_code=404, content=result)
    return result


@app.get("/api/recipe/status")
def recipe_ai_status():
    """AI 모델 상태"""
    return {
        'trained': recipe_engine._engine.trained,
        'device': str(recipe_engine._engine.device),
        'epochs': recipe_engine._engine.train_epochs,
        'loss': round(recipe_engine._engine.train_loss, 4),
        'ingredients_loaded': len(recipe_engine._engine.ingredient_ids),
    }


# ===== 바이오피직스 시뮬레이션 =====
_sim_training_status = {'running': False, 'progress': 0, 'total': 0, 'results': None}


class SimulateRequest(BaseModel):
    smiles_list: list[str]
    concentrations: list[float] = []


@app.post("/api/simulate/recipe")
def simulate_recipe(req: SimulateRequest):
    """레시피 바이오피직스 시뮬레이션 (VirtualNose + Hedonic + Thermo)"""
    conc = req.concentrations if req.concentrations else None
    result = biophys.simulate_recipe(req.smiles_list, conc)
    return biophys._make_serializable(result)


@app.post("/api/simulate/train")
def train_simulation(n: int = 300):
    """유명 향수 데이터로 시뮬레이션 학습 실행"""
    import threading
    if _sim_training_status['running']:
        return {'status': 'already_running', 'progress': _sim_training_status['progress']}

    def _run():
        _sim_training_status['running'] = True
        _sim_training_status['total'] = n
        try:
            sys_path = os.path.join(os.path.dirname(__file__), 'scripts')
            if sys_path not in sys.path:
                sys.path.insert(0, sys_path)
            from train_biophysics import run_training
            results = run_training(n_iterations=n, verbose=True)
            _sim_training_status['results'] = {
                'total': len(results),
                'mean_similarity': round(sum(r['similarity']['overall'] for r in results) / max(len(results), 1), 4),
                'mean_confidence': round(sum(r['confidence']['confidence'] for r in results) / max(len(results), 1), 4),
            }
        except Exception as e:
            _sim_training_status['results'] = {'error': str(e)}
        finally:
            _sim_training_status['running'] = False

    threading.Thread(target=_run, daemon=True).start()
    return {'status': 'started', 'total': n}


@app.get("/api/simulate/status")
def simulate_status():
    """시뮬레이션 학습 상태"""
    return {
        'running': _sim_training_status['running'],
        'progress': _sim_training_status['progress'],
        'total': _sim_training_status['total'],
        'results': _sim_training_status['results'],
        'biophys': biophys.get_status(),
    }


# ===== 분자 궁합 분석 =====
@app.post("/api/harmony/check")
def check_harmony(ingredient_ids: list[str]):
    """원료 목록의 분자 수준 궁합 분석"""
    return mh.check_harmony(ingredient_ids)


@app.post("/api/harmony/train")
def train_harmony(epochs: int = 40):
    """분자 궁합 AI 모델 학습"""
    mh.train_harmony(epochs=epochs)
    return mh.get_status()


@app.get("/api/harmony/status")
def harmony_status():
    """분자 궁합 엔진 상태"""
    return mh.get_status()


# ===== 바이오피직스 시뮬레이터 =====
@app.post("/api/biophysics/simulate")
def biophysics_simulate(smiles_list: list[str], concentrations: list[float] = None):
    """SMILES 리스트의 바이오피직스 시뮬레이션 (가상 코 + 쾌락 + 열역학)"""
    return biophys.simulate_recipe(smiles_list, concentrations)


@app.post("/api/biophysics/evolve")
def biophysics_evolve(generations: int = 100, population: int = 20):
    """RL 자가 대결 진화 실행"""
    return biophys.evolve(generations=generations, population=population)


@app.get("/api/biophysics/status")
def biophysics_status():
    """바이오피직스 시뮬레이터 상태"""
    return biophys.get_status()



# ===== 3-Engine Pipeline =====

class OdorRequest(BaseModel):
    smiles: Optional[str] = None
    smiles_list: Optional[list[str]] = None
    concentrations: Optional[list[float]] = None

class TimelineRequest(BaseModel):
    smiles_list: list[str]
    concentrations: Optional[list[float]] = None
    duration_hours: float = 8

class OptimizeRequest(BaseModel):
    target: str
    generations: int = 50
    population: int = 20
    budget: Optional[int] = None
    longevity_min: Optional[float] = None


@app.post("/api/engine/predict-odor")
def engine_predict_odor(req: OdorRequest):
    """Engine 2: 분자 → 20d 냄새 벡터 예측"""
    if req.smiles:
        return odor_engine.predict_single(req.smiles)
    elif req.smiles_list:
        return odor_engine.predict_mixture(req.smiles_list, req.concentrations)
    else:
        raise HTTPException(400, "smiles or smiles_list required")


@app.get("/api/engine/attention-map")
def engine_attention_map(smiles: str):
    """GATConv attention heatmap: 분자의 어떤 원자가 냄새 예측에 중요한지 시각화"""
    try:
        from scripts.attention_heatmap import generate_attention_heatmap
        
        v5_model, dev = _get_v5_model()  # Cached singleton
        if v5_model is None:
            raise HTTPException(404, "v5 GATConv model not available")
        
        result = generate_attention_heatmap(smiles, v5_model, device=str(dev))
        
        if 'error' in result:
            raise HTTPException(400, result['error'])
        
        pred = odor_engine.predict_single(smiles)
        result['prediction'] = pred
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Attention map failed for {smiles}: {e}", exc_info=True)
        raise HTTPException(500, f"Attention map failed: {type(e).__name__}")


@app.get("/api/engine/discover")
def engine_discover(query: str, top_k: int = 10):
    """향기 검색: 텍스트 → 20d → CosSim → PubChem 미지 분자 Top-K
    예: query='sweet 60% woody 40%' 또는 'citrus fresh'"""
    try:
        from scripts.discovery_engine import search_by_text
        result = search_by_text(query, top_k=top_k)
        if 'error' in result:
            raise HTTPException(400, result['error'])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Discovery search failed: {e}")


@app.get("/api/engine/odor-space")
def engine_odor_space():
    """UMAP 2D odor space data (precomputed)"""
    from pathlib import Path
    umap_json = Path(__file__).parent / 'data' / 'odor_space_umap.json'
    if not umap_json.exists():
        raise HTTPException(404, "UMAP data not generated yet")
    with open(umap_json, 'r') as f:
        return json.load(f)


@app.post("/api/engine/simulate-timeline")
def engine_simulate_timeline(req: TimelineRequest):
    """3-Engine 파이프라인: 물리엔진→후각엔진→소믈리에"""
    concentrations = req.concentrations or [3.0] * len(req.smiles_list)

    # Engine 1: 물리 시뮬레이션 (시간별 분자 농도)
    thermo = biophys._thermo
    physics_result = thermo.simulate_evaporation(req.smiles_list, concentrations, req.duration_hours)

    # Engine 2: AI 냄새 예측 (시간별 냄새 벡터)
    odor_timeline = odor_engine.predict_timeline(
        req.smiles_list, concentrations, physics_result['timeline']
    )

    # Engine 3: 소믈리에 (시적 표현 생성)
    story = sommelier.generate_story('AI Perfume', odor_timeline)
    descriptions = sommelier.describe_evolution(odor_timeline)

    return {
        'physics': {
            'longevity_hours': physics_result['longevity_hours'],
            'transitions': physics_result['transitions'],
            'transition_smoothness': physics_result['transition_smoothness'],
        },
        'odor_timeline': odor_timeline,
        'story': story,
        'descriptions': descriptions,
        'initial_molecules': physics_result['initial_molecules'],
    }


class OptimizeRequest(BaseModel):
    target: str                                  # "fresh floral with citrus"
    target_vector: Optional[list[float]] = None  # 직접 20d 벡터 지정 (선택)
    n_ingredients: int = 5                       # 선택할 재료 수
    n_steps: int = 80                            # 최적화 스텝
    sparsity: float = 0.01                       # 재료 절약 계수
    temperature: float = 0.5                     # softmax 온도


@app.post("/api/engine/optimize")
def engine_optimize(req: OptimizeRequest):
    """Gradient Descent 레시피 최적화
    
    DeepDream 방식: recipe ratios를 trainable parameter로 놓고
    OdorGNN + MixtureTransformer를 통해 역전파하여
    목표 향과의 cosine similarity 최대화
    """
    predictor = odor_engine.get_predictor()
    dev = predictor.device
    
    # 1. 목표 벡터 생성
    if req.target_vector:
        target = np.array(req.target_vector, dtype=np.float32)
    else:
        target = odor_engine.target_from_text(req.target)
    target_t = torch.tensor(target, dtype=torch.float32).to(dev)
    
    # 2. DB에서 후보 재료 선택 (목표 벡터와 가장 유사한 분자들)
    import database as db_module
    all_mols = db_module.get_all_molecules(limit=500)
    
    # 각 분자의 odor vector 계산 + 유사도 정렬
    candidates = []
    for mol in all_mols:
        smiles = mol.get('smiles', '')
        if not smiles:
            continue
        try:
            vec = predictor.gnn.encode(smiles)
            sim = float(np.dot(vec, target) / (np.linalg.norm(vec) * np.linalg.norm(target) + 1e-8))
            candidates.append({
                'smiles': smiles,
                'name': mol.get('name', smiles[:30]),
                'vec': vec,
                'similarity': sim,
            })
        except:
            continue
    
    # 상위 N개 선택
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    selected = candidates[:req.n_ingredients]
    
    if len(selected) < 2:
        raise HTTPException(400, "Not enough candidate ingredients")
    
    N = len(selected)
    
    # 3. 각 분자의 odor vector를 텐서로 변환
    odor_vecs = torch.tensor(
        np.array([c['vec'] for c in selected]), dtype=torch.float32
    ).to(dev)  # [N, 20]
    
    # 4. recipe logits를 학습 가능 파라미터로 설정
    recipe_logits = torch.randn(N, device=dev, requires_grad=True)
    optimizer_gd = torch.optim.Adam([recipe_logits], lr=0.1)
    
    # 5. Gradient Descent 루프
    optimization_curve = []
    best_sim = -1
    best_recipe = None
    
    for step in range(req.n_steps):
        optimizer_gd.zero_grad()
        
        # softmax로 ratios 계산 (합=1)
        ratios = torch.softmax(recipe_logits / req.temperature, dim=0)  # [N]
        
        # 가중 평균으로 혼합 벡터 계산
        mixture_vec = (odor_vecs * ratios.unsqueeze(1)).sum(dim=0)  # [20]
        
        # Cosine similarity loss
        cos_sim = torch.nn.functional.cosine_similarity(
            mixture_vec.unsqueeze(0), target_t.unsqueeze(0)
        )
        
        # Sparsity penalty (작은 비율 패널티)
        sparsity_loss = req.sparsity * (ratios > 0.01).float().sum()
        
        loss = -cos_sim + sparsity_loss
        loss.backward()
        optimizer_gd.step()
        
        sim_val = cos_sim.item()
        optimization_curve.append(round(sim_val, 4))
        
        if sim_val > best_sim:
            best_sim = sim_val
            best_recipe = ratios.detach().cpu().numpy().copy()
    
    # 6. 결과 정리
    recipe = {}
    for i, c in enumerate(selected):
        pct = round(float(best_recipe[i]) * 100, 1)
        if pct > 0.5:  # 0.5% 이하는 무시
            recipe[c['name']] = {
                'percentage': pct,
                'smiles': c['smiles'],
                'individual_similarity': round(c['similarity'], 3),
            }
    
    # 최종 예측값
    final_vec = np.zeros(20)
    for i, c in enumerate(selected):
        final_vec += c['vec'] * best_recipe[i]
    
    # 소믈리에 설명
    description = sommelier.quick_describe(final_vec)
    
    return {
        'target': req.target,
        'target_vector': target.tolist(),
        'recipe': recipe,
        'predicted_odor': final_vec.tolist(),
        'similarity': round(best_sim, 4),
        'optimization_curve': optimization_curve,
        'n_steps': req.n_steps,
        'description': description,
    }


@app.get("/api/data/molecules")
def get_molecules():
    return molecular.get_all()


@app.get("/api/data/molecules3d")
def get_molecules_3d():
    return gnn3d.get_all_3d()


@app.get("/api/data/primary-odors")
def get_primary_odors():
    return primary_odors


@app.post("/api/train")
async def train_all():
    """전체 모델 학습 (SSE 스트리밍)"""
    global neural_net, vae, level

    async def stream():
        global neural_net, vae, level

        def send(stage, msg):
            return f"data: {json.dumps({'stage': stage, 'message': msg})}\n\n"

        yield send('start', f'PyTorch {device.upper()} 학습 시작...')

        # L2: NeuralNet
        yield send('neural', '신경망 학습 (Level 2)...')
        neural_net = NeuralNet(ingredient_db, device)
        neural_net.build_model()
        def nn_progress(e, t, l): pass
        neural_net.train(lambda m,s,p: formulate(m,s,p), 30, nn_progress)
        level = 2
        yield send('neural', '신경망 완료 ✓')

        # L3: VAE
        yield send('vae', 'β-VAE 학습 (Level 3)...')
        vae = VAEGenerator(ingredient_db, device)
        vae.build_model()
        vae.train(lambda m,s,p: formulate(m,s,p), 40)
        level = 3
        yield send('vae', f'VAE 완료 ✓ (KL={vae.last_kl:.4f})')

        # L4: Molecular
        yield send('mol', '분자 GNN 학습 (Level 4)...')
        molecular.build_model()
        molecular.train(60)
        level = 4
        yield send('mol', '분자 GNN 완료 ✓')

        # L5-P1: MPNN
        yield send('3dgnn', 'MPNN 학습 (Pillar 1)...')
        gnn3d.build_model()
        gnn3d.train(molecular, 40)
        yield send('3dgnn', 'MPNN 완료 ✓')

        # L5-P2: Attention
        yield send('binding', 'Multi-Head Attention 학습 (Pillar 2)...')
        binding.build_model()
        binding.train(molecular, 30)
        yield send('binding', 'Attention 완료 ✓')

        # L5-P3: Conv1D
        yield send('sensor', 'Conv1D SR 학습 (Pillar 3)...')
        sensor.build_model()
        sensor.train(molecular, 40)
        yield send('sensor', 'Conv1D SR 완료 ✓')

        # L5-P4: GRU
        yield send('active', 'Embedding+GRU 학습 (Pillar 4)...')
        active_learner.train(molecular, 50)
        yield send('active', 'GRU CLM 완료 ✓')

        level = 5
        yield send('complete', f'Level 5 완전체 — {device.upper()} 전 모듈 가동!')

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/perfume/create")
def create_perfume(req: PerfumeRequest):
    formula = formulate(req.mood, req.season, req.preferences)
    if neural_net and neural_net.trained:
        nn_formula = neural_net.predict(req.mood, req.season, req.preferences)
        if nn_formula:
            formula = _ensemble(formula, nn_formula)

    binding_result = None
    if binding.trained:
        mols = [molecular.get_by_id(m) for m in
                [mol['id'] for mol in molecular.get_all()
                 if any(ing['id'] in (mol.get('source_ingredients') or []) for ing in formula)]
                if molecular.get_by_id(m)]
        if len(mols) >= 2:
            binding_result = binding.simulate_binding(mols[:5], molecular)

    # 결정론적 이름: mood + season 해시 기반
    name_hash = hash(f"{req.mood}_{req.season}_{'_'.join(req.preferences)}") % 9000 + 1000
    return {
        'name': f"AI Perfume #{name_hash}",
        'formula': formula,
        'bindingResult': binding_result,
        'level': level
    }


@app.post("/api/perfume/generate")
def generate_novel():
    if not vae or not vae.trained:
        raise HTTPException(400, "VAE not trained")
    formula = vae.generate_random()
    # 결정론적 이름: formula 해시 기반
    name_hash = hash(str(formula)) % 9999
    return {'name': f'AI Generated №{name_hash}', 'formula': formula, 'level': 3}


@app.post("/api/perfume/latent")
def generate_from_latent(req: LatentRequest):
    if not vae or not vae.trained:
        raise HTTPException(400, "VAE not trained")
    formula = vae.generate_from_latent([req.x, req.y])
    return {'name': f'Latent [{req.x:.1f},{req.y:.1f}]', 'formula': formula, 'level': 3}


@app.post("/api/molecule/predict")
def predict_smell(req: MoleculeRequest):
    if not molecular.trained:
        raise HTTPException(400, "Model not trained")
    return molecular.predict_odor(req.smiles or '')


@app.post("/api/molecule/explore")
def explore_molecule(req: MoleculeRequest):
    mol = molecular.get_by_id(req.id)
    if not mol:
        raise HTTPException(404, "Molecule not found")
    preds = molecular.predict_odor(mol.get('smiles', '')) if molecular.trained else []
    return {**mol, 'predictions': preds}


@app.post("/api/molecule/variants")
def get_variants(req: MoleculeRequest):
    return molecular.generate_variants(req.id)


@app.post("/api/binding/simulate")
def simulate_mixture(req: MixtureRequest):
    mols = [molecular.get_by_id(mid) for mid in req.moleculeIds]
    mols = [m for m in mols if m]
    if len(mols) < 2:
        raise HTTPException(400, "Need at least 2 molecules")
    return binding.simulate_binding(mols, molecular)


@app.post("/api/sensor/simulate")
def simulate_sensor(req: MoleculeRequest):
    mol = molecular.get_by_id(req.id)
    if not mol:
        raise HTTPException(404, "Molecule not found")
    return sensor.get_full_profile(mol)


@app.post("/api/text/predict")
def predict_from_text(req: TextRequest):
    return active_learner.predict_from_text(req.text)


@app.get("/api/chirality/pairs")
def get_chiral_pairs():
    return gnn3d.get_chiral_pairs()


def _ensemble(rule_f, nn_f, w=0.6):
    m = {}
    for i in rule_f: m[i['id']] = {'id': i['id'], 'percentage': i['percentage'] * (1-w)}
    for i in nn_f:
        if i['id'] in m: m[i['id']]['percentage'] += i['percentage'] * w
        else: m[i['id']] = {'id': i['id'], 'percentage': i['percentage'] * w}
    merged = sorted([v for v in m.values() if v['percentage'] > 1], key=lambda x: x['percentage'], reverse=True)[:12]
    total = sum(f['percentage'] for f in merged) or 1
    for f in merged: f['percentage'] = round(f['percentage'] / total * 100, 1)
    adj = 100 - sum(f['percentage'] for f in merged)
    if merged: merged[0]['percentage'] = round(merged[0]['percentage'] + adj, 1)
    return merged


# ===== Human Panel Test =====
PANEL_RESULTS_FILE = Path(__file__).parent / 'data' / 'panel_results.json'

class PanelEvaluation(BaseModel):
    smiles: str
    molecule_name: str
    predicted_top: list
    actual_labels: list
    rating: int  # 1-5 scale
    comment: str = ""

@app.get("/api/panel/random")
def panel_random(count: int = 1):
    """패널 테스트용 랜덤 분자 출제 (DB 기반)"""
    import random, csv
    
    gs_path = Path(__file__).parent / 'data' / 'curated_GS_LF_merged_4983.csv'
    if not gs_path.exists():
        raise HTTPException(404, "GoodScents dataset not found")
    
    with open(gs_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        odor_cols = header[1:]
        rows = list(reader)
    
    samples = random.sample(rows, min(count, len(rows)))
    results = []
    
    for row in samples:
        smiles = row[0]
        active_labels = [col for col, val in zip(odor_cols, row[1:]) if val == '1']
        
        # Get prediction
        try:
            pred = odor_engine.predict_single(smiles)
        except Exception:
            continue
        
        # Get molecule name from DB
        mol_name = smiles
        try:
            mol_info = db.search_molecules(smiles, limit=1)
            if mol_info:
                mol_name = mol_info[0].get('name', smiles)
        except Exception:
            pass
        
        results.append({
            'smiles': smiles,
            'molecule_name': mol_name,
            'actual_labels': active_labels[:8],  # Top 8 labels
            'predicted': pred,
        })
    
    return {'molecules': results, 'total_available': len(rows)}


@app.post("/api/panel/evaluate")
def panel_evaluate(ev: PanelEvaluation):
    """패널 평가 결과 저장"""
    # Load existing results
    results = []
    if PANEL_RESULTS_FILE.exists():
        with open(PANEL_RESULTS_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
    
    results.append({
        'smiles': ev.smiles,
        'molecule_name': ev.molecule_name,
        'predicted_top': ev.predicted_top,
        'actual_labels': ev.actual_labels,
        'rating': ev.rating,
        'comment': ev.comment,
        'timestamp': time.time(),
    })
    
    with open(PANEL_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return {'status': 'saved', 'total_evaluations': len(results)}


@app.get("/api/panel/stats")
def panel_stats():
    """패널 테스트 통계"""
    if not PANEL_RESULTS_FILE.exists():
        return {'total': 0, 'avg_rating': 0, 'distribution': {}}
    
    with open(PANEL_RESULTS_FILE, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if not results:
        return {'total': 0, 'avg_rating': 0, 'distribution': {}}
    
    ratings = [r['rating'] for r in results]
    dist = {str(i): ratings.count(i) for i in range(1, 6)}
    
    return {
        'total': len(results),
        'avg_rating': round(sum(ratings) / len(ratings), 2),
        'distribution': dist,
        'recent': results[-5:],
    }


# ===== Interactive Demo ====================================
@app.get("/demo")
def serve_demo():
    """Interactive AI Perfumery demo page"""
    from fastapi.responses import FileResponse
    demo_path = Path(__file__).parent / 'static' / 'demo.html'
    if not demo_path.exists():
        raise HTTPException(404, "Demo page not found")
    return FileResponse(demo_path, media_type='text/html')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)
