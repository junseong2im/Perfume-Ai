-- 향 AI v3 — PostgreSQL Schema
-- ==============================

-- 분자 기본 정보
CREATE TABLE IF NOT EXISTS molecules (
    smiles       TEXT PRIMARY KEY,
    canonical    TEXT UNIQUE,
    name         TEXT,
    cas          TEXT,
    mw           REAL,
    logp         REAL,
    vp           REAL,          -- 증기압 (mmHg)
    bp           REAL,          -- 끓는점 (°C)
    tpsa         REAL,
    hbd          INTEGER,
    hba          INTEGER,
    rotatable    INTEGER,
    rings        INTEGER,
    aromatic_rings INTEGER,
    heavy_atoms  INTEGER,
    fsp3         REAL,
    has_chiral   BOOLEAN DEFAULT FALSE,
    n_chiral     INTEGER DEFAULT 0,
    sources      TEXT[],        -- 데이터 출처 배열
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 22d 향 벡터 라벨
CREATE TABLE IF NOT EXISTS odor_labels (
    id           SERIAL PRIMARY KEY,
    smiles       TEXT REFERENCES molecules(smiles) ON DELETE CASCADE,
    source       TEXT NOT NULL,   -- 'local', 'pyrfume', 'dream', etc.
    sweet        REAL, sour       REAL, woody      REAL, floral     REAL,
    citrus       REAL, spicy      REAL, musk       REAL, fresh      REAL,
    green        REAL, warm       REAL, fruity     REAL, smoky      REAL,
    powdery      REAL, aquatic    REAL, herbal     REAL, amber      REAL,
    leather      REAL, earthy     REAL, ozonic     REAL, metallic   REAL,
    fatty        REAL, waxy       REAL,
    confidence   REAL DEFAULT 1.0,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(smiles, source)
);

-- 노트 위치 (Top/Mid/Base) + 지속력 + 확산력
CREATE TABLE IF NOT EXISTS note_positions (
    id           SERIAL PRIMARY KEY,
    smiles       TEXT REFERENCES molecules(smiles) ON DELETE CASCADE,
    top_ratio    REAL,          -- 0-1
    mid_ratio    REAL,
    base_ratio   REAL,
    longevity    REAL,          -- 0-1 (지속력)
    sillage      REAL,          -- 0-1 (확산력)
    source       TEXT,
    UNIQUE(smiles, source)
);

-- 수용체 바인딩 (M2OR 데이터)
CREATE TABLE IF NOT EXISTS receptor_binds (
    id           SERIAL PRIMARY KEY,
    smiles       TEXT REFERENCES molecules(smiles) ON DELETE CASCADE,
    receptor_id  TEXT NOT NULL,
    binding_score REAL,
    ec50         REAL,
    source       TEXT DEFAULT 'm2or',
    UNIQUE(smiles, receptor_id, source)
);

-- 쾌적도 점수
CREATE TABLE IF NOT EXISTS hedonic_scores (
    id           SERIAL PRIMARY KEY,
    smiles       TEXT REFERENCES molecules(smiles) ON DELETE CASCADE,
    pleasantness REAL,          -- -1 ~ +1
    source       TEXT,
    UNIQUE(smiles, source)
);

-- 레시피
CREATE TABLE IF NOT EXISTS recipes (
    id           SERIAL PRIMARY KEY,
    name         TEXT,
    style        TEXT,          -- 'floral', 'woody', etc.
    mood         TEXT,          -- 'romantic', 'energetic', etc.
    season       TEXT,          -- 'spring', 'summer', etc.
    source       TEXT,          -- 'fragrantica', 'local', etc.
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 레시피 구성 원료
CREATE TABLE IF NOT EXISTS recipe_items (
    id           SERIAL PRIMARY KEY,
    recipe_id    INTEGER REFERENCES recipes(id) ON DELETE CASCADE,
    smiles       TEXT,          -- nullable (일부 원료 SMILES 없음)
    ingredient   TEXT NOT NULL,
    ratio        REAL NOT NULL, -- 비율 (%)
    note_type    TEXT           -- 'top', 'mid', 'base', null
);

-- 안전 규제 (IFRA)
CREATE TABLE IF NOT EXISTS safety_limits (
    id           SERIAL PRIMARY KEY,
    smiles       TEXT,
    cas          TEXT,
    ifra_cat     TEXT,          -- IFRA 카테고리
    max_pct      REAL,          -- 최대 사용 농도 (%)
    allergen_26  BOOLEAN DEFAULT FALSE,  -- EU 26종 알레르겐
    skin_irr     BOOLEAN DEFAULT FALSE,
    photo_tox    BOOLEAN DEFAULT FALSE,
    env_tox      BOOLEAN DEFAULT FALSE,
    source       TEXT DEFAULT 'ifra',
    UNIQUE(smiles, ifra_cat)
);

-- 디스크립터 라벨 (138d 다중 라벨)
CREATE TABLE IF NOT EXISTS descriptor_labels (
    id           SERIAL PRIMARY KEY,
    smiles       TEXT REFERENCES molecules(smiles) ON DELETE CASCADE,
    descriptors  TEXT[],        -- 활성화된 디스크립터 목록
    source       TEXT,
    UNIQUE(smiles, source)
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_odor_labels_smiles ON odor_labels(smiles);
CREATE INDEX IF NOT EXISTS idx_recipe_items_recipe ON recipe_items(recipe_id);
CREATE INDEX IF NOT EXISTS idx_receptor_binds_smiles ON receptor_binds(smiles);
CREATE INDEX IF NOT EXISTS idx_hedonic_smiles ON hedonic_scores(smiles);
CREATE INDEX IF NOT EXISTS idx_safety_smiles ON safety_limits(smiles);
CREATE INDEX IF NOT EXISTS idx_molecules_cas ON molecules(cas);
CREATE INDEX IF NOT EXISTS idx_molecules_name ON molecules(name);

-- 뷰: 학습 데이터 export
CREATE OR REPLACE VIEW training_export AS
SELECT
    m.smiles, m.name, m.mw, m.logp, m.tpsa, m.hbd, m.hba,
    m.rotatable, m.rings, m.aromatic_rings, m.fsp3,
    m.has_chiral, m.n_chiral,
    ol.sweet, ol.sour, ol.woody, ol.floral, ol.citrus,
    ol.spicy, ol.musk, ol.fresh, ol.green, ol.warm,
    ol.fruity, ol.smoky, ol.powdery, ol.aquatic, ol.herbal,
    ol.amber, ol.leather, ol.earthy, ol.ozonic, ol.metallic,
    ol.fatty, ol.waxy, ol.confidence,
    np.top_ratio, np.mid_ratio, np.base_ratio,
    np.longevity, np.sillage,
    hs.pleasantness
FROM molecules m
LEFT JOIN odor_labels ol ON m.smiles = ol.smiles
LEFT JOIN note_positions np ON m.smiles = np.smiles
LEFT JOIN hedonic_scores hs ON m.smiles = hs.smiles;
