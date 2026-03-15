-- ==============================================
-- AI Perfumer Database Schema
-- ==============================================

-- 1. 냄새 descriptor 마스터 테이블
CREATE TABLE IF NOT EXISTS odor_descriptors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50),  -- 'fruity', 'floral', 'woody' 등 상위 분류
    name_ko VARCHAR(100)
);

-- 2. 분자 테이블 (Pyrfume + 기존 데이터)
CREATE TABLE IF NOT EXISTS molecules (
    id SERIAL PRIMARY KEY,
    cid BIGINT UNIQUE,                  -- PubChem CID
    name VARCHAR(500) NOT NULL,
    name_ko VARCHAR(500),
    iupac_name VARCHAR(1000),
    smiles TEXT,                         -- IsomericSMILES
    molecular_weight FLOAT,
    logp FLOAT,
    hbd INT DEFAULT 0,                  -- H-bond donors
    hba INT DEFAULT 0,                  -- H-bond acceptors
    rotatable_bonds INT DEFAULT 0,
    rings INT DEFAULT 0,
    aromatic_rings INT DEFAULT 0,
    functional_groups TEXT[],            -- PostgreSQL array
    odor_strength INT DEFAULT 5,
    source VARCHAR(50) DEFAULT 'pyrfume', -- 'pyrfume', 'custom', 'goodscents'
    created_at TIMESTAMP DEFAULT NOW()
);

-- 3. 분자-냄새 매핑 (다대다)
CREATE TABLE IF NOT EXISTS molecule_odors (
    id SERIAL PRIMARY KEY,
    molecule_id INT REFERENCES molecules(id) ON DELETE CASCADE,
    descriptor_id INT REFERENCES odor_descriptors(id) ON DELETE CASCADE,
    strength FLOAT DEFAULT 1.0,         -- 0~1 (강도, 기본 1=있음)
    source VARCHAR(50) DEFAULT 'pyrfume',
    UNIQUE(molecule_id, descriptor_id)
);

-- 4. 향료 원료 테이블 (기존 ingredients.json 이전)
CREATE TABLE IF NOT EXISTS ingredients (
    id VARCHAR(100) PRIMARY KEY,
    name_ko VARCHAR(200),
    name_en VARCHAR(200),
    category VARCHAR(50),               -- citrus, floral, woody...
    note_type VARCHAR(20),              -- top, middle, base
    volatility INT DEFAULT 5,
    intensity INT DEFAULT 5,
    longevity INT DEFAULT 5,
    typical_pct FLOAT,
    max_pct FLOAT,
    descriptors TEXT[],
    moods TEXT[],
    seasons TEXT[],
    source VARCHAR(50) DEFAULT 'custom'
);

-- 5. 향료-분자 매핑
CREATE TABLE IF NOT EXISTS ingredient_molecules (
    id SERIAL PRIMARY KEY,
    ingredient_id VARCHAR(100) REFERENCES ingredients(id) ON DELETE CASCADE,
    molecule_id INT REFERENCES molecules(id) ON DELETE CASCADE,
    percentage FLOAT,                   -- 해당 향료 내 분자 비율
    UNIQUE(ingredient_id, molecule_id)
);

-- 6. 어코드(조화 규칙) 테이블
CREATE TABLE IF NOT EXISTS accords (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    name_ko VARCHAR(200),
    type VARCHAR(50),                   -- 'classic', 'modern', 'niche'
    ingredients TEXT[],                 -- 구성 향료 ID 배열
    ratios FLOAT[],                    -- 배합 비율
    description TEXT
);

-- 인덱스
CREATE INDEX IF NOT EXISTS idx_molecules_smiles ON molecules(smiles);
CREATE INDEX IF NOT EXISTS idx_molecules_cid ON molecules(cid);
CREATE INDEX IF NOT EXISTS idx_molecules_name ON molecules(name);
CREATE INDEX IF NOT EXISTS idx_molecule_odors_mol ON molecule_odors(molecule_id);
CREATE INDEX IF NOT EXISTS idx_molecule_odors_desc ON molecule_odors(descriptor_id);
CREATE INDEX IF NOT EXISTS idx_ingredients_category ON ingredients(category);

-- 113개 Leffingwell 냄새 descriptor 초기 데이터
INSERT INTO odor_descriptors (name, category, name_ko) VALUES
    ('alcoholic', 'chemical', '알코올'),
    ('aldehydic', 'chemical', '알데히드'),
    ('alliaceous', 'sulfurous', '마늘/양파'),
    ('almond', 'nutty', '아몬드'),
    ('animal', 'animalic', '동물적'),
    ('anisic', 'spicy', '아니스'),
    ('apple', 'fruity', '사과'),
    ('apricot', 'fruity', '살구'),
    ('aromatic', 'herbal', '방향성'),
    ('balsamic', 'resinous', '발사믹'),
    ('banana', 'fruity', '바나나'),
    ('beefy', 'meaty', '쇠고기'),
    ('berry', 'fruity', '베리'),
    ('black currant', 'fruity', '블랙커런트'),
    ('brandy', 'alcoholic', '브랜디'),
    ('bread', 'baked', '빵'),
    ('brothy', 'meaty', '국물'),
    ('burnt', 'smoky', '탄'),
    ('buttery', 'dairy', '버터'),
    ('cabbage', 'sulfurous', '양배추'),
    ('camphoreous', 'fresh', '장뇌'),
    ('caramellic', 'sweet', '카라멜'),
    ('catty', 'sulfurous', '고양이'),
    ('chamomile', 'floral', '캐모마일'),
    ('cheesy', 'dairy', '치즈'),
    ('cherry', 'fruity', '체리'),
    ('chicken', 'meaty', '닭'),
    ('chocolate', 'sweet', '초콜릿'),
    ('cinnamon', 'spicy', '시나몬'),
    ('citrus', 'citrus', '시트러스'),
    ('cocoa', 'sweet', '코코아'),
    ('coconut', 'tropical', '코코넛'),
    ('coffee', 'roasted', '커피'),
    ('cognac', 'alcoholic', '코냑'),
    ('coumarinic', 'sweet', '쿠마린'),
    ('creamy', 'dairy', '크리미'),
    ('cucumber', 'green', '오이'),
    ('dairy', 'dairy', '유제품'),
    ('dry', 'woody', '드라이'),
    ('earthy', 'earthy', '흙냄새'),
    ('ethereal', 'chemical', '에테르'),
    ('fatty', 'waxy', '지방질'),
    ('fermented', 'fermented', '발효'),
    ('fishy', 'marine', '비린내'),
    ('floral', 'floral', '꽃'),
    ('fresh', 'fresh', '신선한'),
    ('fruity', 'fruity', '과일'),
    ('garlic', 'sulfurous', '마늘'),
    ('gasoline', 'chemical', '가솔린'),
    ('grape', 'fruity', '포도'),
    ('grapefruit', 'citrus', '자몽'),
    ('grassy', 'green', '풀'),
    ('green', 'green', '그린'),
    ('hay', 'green', '건초'),
    ('hazelnut', 'nutty', '헤이즐넛'),
    ('herbal', 'herbal', '허브'),
    ('honey', 'sweet', '꿀'),
    ('horseradish', 'spicy', '고추냉이'),
    ('jasmine', 'floral', '재스민'),
    ('ketonic', 'chemical', '케톤'),
    ('leafy', 'green', '잎'),
    ('leathery', 'animalic', '가죽'),
    ('lemon', 'citrus', '레몬'),
    ('malty', 'baked', '맥아'),
    ('meaty', 'meaty', '고기'),
    ('medicinal', 'chemical', '약'),
    ('melon', 'fruity', '멜론'),
    ('metallic', 'chemical', '금속'),
    ('milky', 'dairy', '우유'),
    ('mint', 'fresh', '민트'),
    ('mushroom', 'earthy', '버섯'),
    ('musk', 'musk', '머스크'),
    ('musty', 'earthy', '곰팡이'),
    ('nutty', 'nutty', '견과류'),
    ('odorless', 'neutral', '무취'),
    ('oily', 'waxy', '기름'),
    ('onion', 'sulfurous', '양파'),
    ('orange', 'citrus', '오렌지'),
    ('orris', 'floral', '오리스'),
    ('peach', 'fruity', '복숭아'),
    ('pear', 'fruity', '배'),
    ('phenolic', 'chemical', '페놀'),
    ('pine', 'woody', '소나무'),
    ('pineapple', 'tropical', '파인애플'),
    ('plum', 'fruity', '자두'),
    ('popcorn', 'baked', '팝콘'),
    ('potato', 'vegetable', '감자'),
    ('pungent', 'spicy', '매운'),
    ('radish', 'vegetable', '무'),
    ('ripe', 'fruity', '익은'),
    ('roasted', 'roasted', '로스팅'),
    ('rose', 'floral', '장미'),
    ('rum', 'alcoholic', '럼'),
    ('savory', 'meaty', '짭짤한'),
    ('sharp', 'chemical', '날카로운'),
    ('smoky', 'smoky', '스모키'),
    ('solvent', 'chemical', '솔벤트'),
    ('sour', 'acidic', '신맛'),
    ('spicy', 'spicy', '스파이시'),
    ('strawberry', 'fruity', '딸기'),
    ('sulfurous', 'sulfurous', '유황'),
    ('sweet', 'sweet', '달콤한'),
    ('tea', 'herbal', '차'),
    ('tobacco', 'smoky', '담배'),
    ('tomato', 'vegetable', '토마토'),
    ('tropical', 'tropical', '트로피컬'),
    ('vanilla', 'sweet', '바닐라'),
    ('vegetable', 'vegetable', '채소'),
    ('violet', 'floral', '바이올렛'),
    ('warm', 'warm', '따뜻한'),
    ('waxy', 'waxy', '왁스'),
    ('winey', 'alcoholic', '와인'),
    ('woody', 'woody', '우디')
ON CONFLICT (name) DO NOTHING;
