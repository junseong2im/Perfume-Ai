"""
향 AI v3 — 데이터 수집 스크립트 (수정판)
========================================
7개 공개 소스 + 기존 로컬 데이터를 통합합니다.

사용법:
    python scripts/collect_data.py --all          # 전체 수집
    python scripts/collect_data.py --pyrfume      # Pyrfume만
    python scripts/collect_data.py --dream        # DREAM만
    python scripts/collect_data.py --m2or         # M2OR만
    python scripts/collect_data.py --pubchem      # PubChem 물리속성
    python scripts/collect_data.py --local        # 로컬 데이터 정리
    python scripts/collect_data.py --unify        # 통합 + 중복 제거
    python scripts/collect_data.py --rdkit        # RDKit 물리속성 계산
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict

import requests
import pandas as pd
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("[WARNING] RDKit not found — SMILES canonicalization disabled")

# ================================================================
# 경로 설정
# ================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for d in [RAW_DIR, PROCESSED_DIR,
          RAW_DIR / "pyrfume", RAW_DIR / "dream", RAW_DIR / "m2or",
          RAW_DIR / "pubchem", RAW_DIR / "fragrantica", RAW_DIR / "ifra",
          RAW_DIR / "flavordb"]:
    d.mkdir(parents=True, exist_ok=True)


def canonical_smiles(smi):
    """SMILES → 표준형 (RDKit)"""
    if not HAS_RDKIT or not smi:
        return smi
    try:
        mol = Chem.MolFromSmiles(str(smi).strip())
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return str(smi).strip()


def download_file(url, dest, desc=None):
    """URL → 파일 다운로드 (progress 표시)"""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        size_kb = os.path.getsize(dest) / 1024
        print(f"  [SKIP] {desc or dest} ({size_kb:.0f} KB)")
        return True
    print(f"  [DL] {desc or url}")
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        size_kb = os.path.getsize(dest) / 1024
        print(f"       → {size_kb:.0f} KB ✓")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def parse_csv_smiles(path, source_name):
    """CSV 파일에서 SMILES 추출 (유연한 컬럼 탐색)"""
    molecules = {}
    try:
        df = pd.read_csv(path, dtype=str, on_bad_lines='skip')
        smiles_col = None
        name_col = None

        for col in df.columns:
            cl = col.lower().strip()
            if cl in ('smiles', 'isomericsmiles', 'canonicalsmiles', 'isosmiles'):
                smiles_col = col
            if cl in ('name', 'iupacname', 'preferredname', 'moleculename'):
                name_col = col

        if smiles_col is None:
            # Try any column with 'smiles' in it
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break

        if smiles_col:
            for _, row in df.iterrows():
                smi = canonical_smiles(row.get(smiles_col, ''))
                if smi and len(smi) > 1:
                    molecules[smi] = {
                        'smiles': smi,
                        'source': source_name,
                        'name': str(row.get(name_col, '')) if name_col else '',
                    }
        else:
            print(f"    No SMILES column found. Columns: {list(df.columns)[:10]}")

    except Exception as e:
        print(f"    Parse error: {e}")

    return molecules


# ================================================================
# Source 1: Pyrfume — pyrfume/pyrfume-data 레포
# ================================================================
def collect_pyrfume():
    """Pyrfume: 다수의 후각 데이터셋 (pyrfume-data 레포)"""
    print("\n=== [1/7] Pyrfume ===")
    out_dir = RAW_DIR / "pyrfume"

    # pyrfume-data 레포의 개별 데이터셋 molecules.csv
    datasets = {
        "leffingwell": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/leffingwell/molecules.csv",
        "goodscents": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/goodscents/molecules.csv",
        "arctander": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/arctander_1969/molecules.csv",
        "keller_2016": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/keller_2016/molecules.csv",
        "bushdid_2014": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/bushdid_2014/molecules.csv",
        "mainland_2015": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/mainland_2015/molecules.csv",
        "snitz_2013": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/snitz_2013/molecules.csv",
        "dravnieks_1985": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/dravnieks_1985/molecules.csv",
        "sigma_2014": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/sigma_2014/molecules.csv",
    }

    # 지각 데이터 (behavior.csv)
    behavior_urls = {
        "leffingwell_behavior": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/leffingwell/behavior.csv",
        "goodscents_behavior": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/goodscents/behavior.csv",
        "keller_2016_behavior": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/keller_2016/behavior.csv",
        "dravnieks_1985_behavior": "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/dravnieks_1985/behavior.csv",
    }

    # Download molecules
    all_molecules = {}
    for ds_name, url in datasets.items():
        dest = out_dir / f"{ds_name}_molecules.csv"
        if download_file(url, dest, f"Pyrfume/{ds_name}"):
            mols = parse_csv_smiles(dest, f"pyrfume_{ds_name}")
            all_molecules.update(mols)
            print(f"    → {len(mols)} molecules from {ds_name}")

    # Download behavior data
    for bh_name, url in behavior_urls.items():
        dest = out_dir / f"{bh_name}.csv"
        download_file(url, dest, f"Pyrfume/{bh_name}")

    print(f"\n  Pyrfume TOTAL: {len(all_molecules)} unique molecules")
    with open(PROCESSED_DIR / "pyrfume_molecules.json", 'w', encoding='utf-8') as f:
        json.dump(list(all_molecules.values()), f, indent=2, ensure_ascii=False)
    return len(all_molecules)


# ================================================================
# Source 2: FlavorDB
# ================================================================
def collect_flavordb():
    """FlavorDB: 식품 향미 분자 (API/크롤링)"""
    print("\n=== [2/7] FlavorDB ===")
    out_dir = RAW_DIR / "flavordb"

    # FlavorDB has a REST API
    molecules = {}
    base_url = "https://cosylab.iiitd.edu.in/flavordb/entities_json"
    max_pages = 50  # ~25K molecules across ~500 entities

    # Try fetching entity list first
    try:
        print(f"  Fetching FlavorDB entities...")
        r = requests.get("https://cosylab.iiitd.edu.in/flavordb/entities_json?category=all&page=0",
                         timeout=30)
        if r.status_code == 200:
            data = r.json()
            print(f"  Got response with {len(data)} items")
            # Save raw
            with open(out_dir / "entities_page0.json", 'w') as f:
                json.dump(data, f, indent=2)
        else:
            print(f"  FlavorDB API returned {r.status_code}")
    except Exception as e:
        print(f"  FlavorDB API error: {e}")
        print(f"  [NOTE] FlavorDB may have rate limits or be temporarily down.")

    print(f"  FlavorDB: {len(molecules)} molecules collected")
    if molecules:
        with open(PROCESSED_DIR / "flavordb_molecules.json", 'w') as f:
            json.dump(list(molecules.values()), f, indent=2)
    return len(molecules)


# ================================================================
# Source 3: DREAM Olfaction Challenge
# ================================================================
def collect_dream():
    """DREAM: 476 molecules × 21 attributes × 49 subjects"""
    print("\n=== [3/7] DREAM Olfaction ===")
    out_dir = RAW_DIR / "dream"

    # From dream-olfaction/olfaction-prediction repo
    urls = {
        "TrainSet.txt": "https://raw.githubusercontent.com/dream-olfaction/olfaction-prediction/master/data/TrainSet.txt",
        "LBs1.txt": "https://raw.githubusercontent.com/dream-olfaction/olfaction-prediction/master/data/LBs1.txt",
        "LBs2.txt": "https://raw.githubusercontent.com/dream-olfaction/olfaction-prediction/master/data/LBs2.txt",
        "molecular_descriptors_data.txt": "https://raw.githubusercontent.com/dream-olfaction/olfaction-prediction/master/data/molecular_descriptors_data.txt",
        "CID_names.txt": "https://raw.githubusercontent.com/dream-olfaction/olfaction-prediction/master/data/CID_names.txt",
    }

    # Also from Keller 2016 via pyrfume-data (same underlying data)
    urls["keller_molecules.csv"] = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/keller_2016/molecules.csv"
    urls["keller_stimuli.csv"] = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/keller_2016/stimuli.csv"
    urls["keller_behavior.csv"] = "https://raw.githubusercontent.com/pyrfume/pyrfume-data/main/keller_2016/behavior.csv"

    for name, url in urls.items():
        download_file(url, out_dir / name, f"DREAM/{name}")

    # Parse molecules from keller_molecules.csv (has SMILES)
    molecules = {}
    keller_path = out_dir / "keller_molecules.csv"
    if keller_path.exists():
        molecules = parse_csv_smiles(keller_path, "dream_keller")
        print(f"  DREAM/Keller: {len(molecules)} molecules with SMILES")

    # Parse perceptual data from TrainSet.txt
    train_path = out_dir / "TrainSet.txt"
    perceptual_data = []
    if train_path.exists():
        try:
            df = pd.read_csv(train_path, sep='\t', dtype=str, on_bad_lines='skip')
            print(f"  TrainSet: {len(df)} rows, {len(df.columns)} columns")
            print(f"    Columns: {list(df.columns)[:10]}...")
            perceptual_data = df.to_dict('records')
        except Exception as e:
            print(f"  TrainSet parse error: {e}")

    print(f"  DREAM TOTAL: {len(molecules)} molecules, {len(perceptual_data)} perceptual rows")
    with open(PROCESSED_DIR / "dream_molecules.json", 'w', encoding='utf-8') as f:
        json.dump(list(molecules.values()), f, indent=2, ensure_ascii=False)
    if perceptual_data:
        with open(PROCESSED_DIR / "dream_perceptual.json", 'w', encoding='utf-8') as f:
            json.dump(perceptual_data[:2000], f, indent=2)  # Limit size
    return len(molecules)


# ================================================================
# Source 4: M2OR — Molecule-to-Olfactory Receptor
# ================================================================
def collect_m2or():
    """M2OR: 51K+ molecule-receptor binding pairs"""
    print("\n=== [4/7] M2OR ===")
    out_dir = RAW_DIR / "m2or"

    # chemosim-lab/M2OR repo
    urls = {
        "M2OR_20230428.csv": "https://raw.githubusercontent.com/chemosim-lab/M2OR/main/data/M2OR_20230428.csv",
    }

    for name, url in urls.items():
        download_file(url, out_dir / name, f"M2OR/{name}")

    # Parse M2OR
    molecules = {}
    bindings = []
    m2or_path = out_dir / "M2OR_20230428.csv"
    if m2or_path.exists():
        try:
            df = pd.read_csv(m2or_path, dtype=str, on_bad_lines='skip')
            print(f"  M2OR: {len(df)} rows, columns: {list(df.columns)[:12]}")

            # Find SMILES column
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break

            if smiles_col:
                for _, row in df.iterrows():
                    smi = canonical_smiles(row.get(smiles_col, ''))
                    if smi and len(smi) > 1:
                        molecules[smi] = {'smiles': smi, 'source': 'm2or'}

            # Store binding data
            bindings = df.to_dict('records')
            print(f"  Unique molecules: {len(molecules)}")
            print(f"  Binding pairs: {len(bindings)}")

        except Exception as e:
            print(f"  M2OR parse error: {e}")

    with open(PROCESSED_DIR / "m2or_molecules.json", 'w', encoding='utf-8') as f:
        json.dump(list(molecules.values()), f, indent=2)
    with open(PROCESSED_DIR / "m2or_bindings.json", 'w', encoding='utf-8') as f:
        json.dump(bindings[:5000], f, indent=2)  # First 5K rows
    return len(molecules)


# ================================================================
# Source 5: IFRA 49th Amendment — 안전 규제 데이터
# ================================================================
def collect_ifra():
    """IFRA: 향료 안전 규제 (카테고리별 최대 농도)"""
    print("\n=== [5/7] IFRA ===")
    out_dir = RAW_DIR / "ifra"

    # IFRA 데이터는 PDF 기반이나, 기존 코드에서 추출된 데이터 활용
    # biophysics_simulator.py에서 IFRA_MAX_CONCENTRATION 딕셔너리 추출
    ifra_data = {
        'categories': {
            'Fine Fragrance': 20.0, 'Eau de Toilette': 15.0,
            'Deodorant': 2.0, 'Body Lotion': 3.0, 'Shampoo': 1.0,
            'Soap': 2.5, 'Detergent': 0.5, 'Air Freshener': 5.0,
            'Candle': 10.0, 'Lip Product': 0.1, 'Baby Product': 0.1,
        },
        'restricted_materials': [],
        'source': 'IFRA 49th Amendment (extracted from codebase)',
    }

    # Also try to get IFRA standards list from existing code
    ifra_path = BASE_DIR / "biophysics_simulator.py"
    if ifra_path.exists():
        try:
            with open(ifra_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Extract IFRA_MAX_CONCENTRATION dict
            import re
            match = re.search(r'IFRA_MAX_CONCENTRATION\s*=\s*\{([^}]+)\}', content)
            if match:
                print(f"  Found IFRA_MAX_CONCENTRATION in biophysics_simulator.py")
                # Parse key-value pairs
                pairs = re.findall(r"'([^']+)'\s*:\s*([\d.]+)", match.group(1))
                for key, val in pairs:
                    ifra_data['categories'][key] = float(val)
                print(f"  {len(pairs)} category limits extracted")
        except Exception as e:
            print(f"  IFRA extraction error: {e}")

    with open(PROCESSED_DIR / "ifra_limits.json", 'w', encoding='utf-8') as f:
        json.dump(ifra_data, f, indent=2, ensure_ascii=False)
    print(f"  IFRA: {len(ifra_data['categories'])} category limits")
    return len(ifra_data['categories'])


# ================================================================
# Source 6: PubChem — 물리속성 보강
# ================================================================
def collect_pubchem(smiles_list=None, max_queries=500):
    """PubChem API로 분자 물리속성 수집"""
    print("\n=== [6/7] PubChem 물리속성 ===")

    if smiles_list is None:
        smiles_list = set()
        # Collect from all processed molecules
        for jf in PROCESSED_DIR.glob("*_molecules.json"):
            try:
                with open(jf) as f:
                    mols = json.load(f)
                for m in mols:
                    s = m.get('smiles', '')
                    if s:
                        smiles_list.add(s)
            except:
                pass
        # Also from unified training data
        csv_path = DATA_DIR / "unified_training_data.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, dtype=str)
                for col in df.columns:
                    if 'smiles' in col.lower():
                        smiles_list.update(df[col].dropna().tolist())
                        break
            except:
                pass

    smiles_list = list(smiles_list)
    if not smiles_list:
        print("  No SMILES to query")
        return 0

    print(f"  Total SMILES: {len(smiles_list)}")

    # Load existing properties
    existing = {}
    for props_path in [DATA_DIR / "experimental_properties.json",
                       PROCESSED_DIR / "pubchem_properties.json"]:
        if props_path.exists():
            with open(props_path) as f:
                existing.update(json.load(f))
    print(f"  Already have properties for {len(existing)} molecules")

    missing = [s for s in smiles_list if s not in existing][:max_queries]
    print(f"  Will fetch: {len(missing)} (max {max_queries})")

    if not missing:
        with open(PROCESSED_DIR / "pubchem_properties.json", 'w') as f:
            json.dump(existing, f, indent=2)
        return len(existing)

    collected = 0
    errors = 0
    for i, smi in enumerate(missing):
        try:
            encoded = requests.utils.quote(smi, safe='')
            url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
                   f"{encoded}/property/"
                   f"MolecularWeight,XLogP,TPSA,HBondDonorCount,"
                   f"HBondAcceptorCount,RotatableBondCount/JSON")
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if 'PropertyTable' in data:
                    props = data['PropertyTable']['Properties'][0]
                    existing[smi] = {
                        'mw': props.get('MolecularWeight'),
                        'logp': props.get('XLogP'),
                        'tpsa': props.get('TPSA'),
                        'hbd': props.get('HBondDonorCount'),
                        'hba': props.get('HBondAcceptorCount'),
                        'rotatable': props.get('RotatableBondCount'),
                    }
                    collected += 1
            elif r.status_code == 404:
                pass  # SMILES not found
            else:
                errors += 1
            time.sleep(0.25)  # Rate limit
        except Exception:
            errors += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(missing)} ({collected} new, {errors} errors)")

    with open(PROCESSED_DIR / "pubchem_properties.json", 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"  PubChem: +{collected} new (total: {len(existing)})")
    return len(existing)


# ================================================================
# Source 7: 기존 로컬 데이터 정리
# ================================================================
def collect_local():
    """기존 로컬 데이터 파싱 + 정리"""
    print("\n=== [7/7] 로컬 데이터 ===")

    molecules = {}
    recipes = []

    # 1. unified_training_data.csv
    csv_path = DATA_DIR / "unified_training_data.csv"
    if csv_path.exists():
        mols = parse_csv_smiles(csv_path, "local_unified")
        molecules.update(mols)
        print(f"  unified_training_data: {len(mols)} molecules")

    # 2. curated_GS_LF_merged.csv
    gs_path = DATA_DIR / "curated_GS_LF_merged_4983.csv"
    if gs_path.exists():
        mols = parse_csv_smiles(gs_path, "gs_lf_curated")
        new = {k: v for k, v in mols.items() if k not in molecules}
        molecules.update(new)
        print(f"  curated_GS_LF: +{len(new)} molecules")

    # 3. chemprop_train.csv
    cp_path = DATA_DIR / "chemprop_train.csv"
    if cp_path.exists():
        mols = parse_csv_smiles(cp_path, "chemprop")
        new = {k: v for k, v in mols.items() if k not in molecules}
        molecules.update(new)
        print(f"  chemprop_train: +{len(new)} molecules")

    # 4. recipe_data.json
    recipe_path = DATA_DIR / "recipe_data.json"
    if recipe_path.exists():
        try:
            with open(recipe_path, 'r', encoding='utf-8') as f:
                rd = json.load(f)
            if isinstance(rd, list):
                recipes = rd
            elif isinstance(rd, dict):
                recipes = list(rd.values())
            print(f"  recipe_data: {len(recipes)} recipes")
        except Exception as e:
            print(f"  recipe_data error: {e}")

    # 5. fragrantica_raw.csv
    frag_path = DATA_DIR / "fragrantica_raw.csv"
    if frag_path.exists():
        try:
            df = pd.read_csv(frag_path, dtype=str)
            print(f"  fragrantica_raw: {len(df)} perfumes")
        except:
            pass

    # 6. compatibility_pairs.json
    compat_path = DATA_DIR / "compatibility_pairs.json"
    if compat_path.exists():
        try:
            with open(compat_path) as f:
                pairs = json.load(f)
            if isinstance(pairs, list):
                print(f"  compatibility_pairs: {len(pairs)} pairs")
        except:
            pass

    # 7. hedonic_weights.json
    hedonic_path = DATA_DIR / "hedonic_weights.json"
    if hedonic_path.exists():
        print(f"  hedonic_weights: exists ✓")

    print(f"\n  LOCAL TOTAL: {len(molecules)} unique molecules, {len(recipes)} recipes")
    with open(PROCESSED_DIR / "local_molecules.json", 'w', encoding='utf-8') as f:
        json.dump(list(molecules.values()), f, indent=2, ensure_ascii=False)
    return len(molecules)


# ================================================================
# 통합 + 중복 제거
# ================================================================
def unify_all():
    """모든 소스 통합 + SMILES 중복 제거"""
    print("\n=== UNIFICATION ===")

    all_molecules = {}
    source_counts = defaultdict(int)

    for json_file in sorted(PROCESSED_DIR.glob("*_molecules.json")):
        try:
            with open(json_file, encoding='utf-8') as f:
                mols = json.load(f)
            source = json_file.stem.replace("_molecules", "")
            for mol in mols:
                smi = canonical_smiles(mol.get('smiles', ''))
                if smi and len(smi) > 1:
                    if smi not in all_molecules:
                        all_molecules[smi] = {
                            'smiles': smi,
                            'sources': [],
                            'name': mol.get('name', ''),
                        }
                    if source not in all_molecules[smi]['sources']:
                        all_molecules[smi]['sources'].append(source)
                    source_counts[source] += 1
        except Exception as e:
            print(f"  [WARN] {json_file.name}: {e}")

    print(f"\n  Source statistics:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {source}: {count:,}")

    # Multi-source molecules (higher confidence)
    multi = sum(1 for m in all_molecules.values() if len(m['sources']) > 1)
    print(f"\n  TOTAL UNIQUE MOLECULES: {len(all_molecules):,}")
    print(f"  Multi-source (higher confidence): {multi:,}")

    # Save JSON
    unified = list(all_molecules.values())
    with open(PROCESSED_DIR / "unified_molecules.json", 'w', encoding='utf-8') as f:
        json.dump(unified, f, indent=2, ensure_ascii=False)

    # Save CSV
    rows = []
    for mol in unified:
        rows.append({
            'smiles': mol['smiles'],
            'name': mol.get('name', ''),
            'sources': '|'.join(mol['sources']),
            'n_sources': len(mol['sources']),
        })
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_DIR / "unified_molecules.csv", index=False)

    print(f"\n  Saved: unified_molecules.json + .csv")
    return len(unified)


# ================================================================
# RDKit 물리속성 벌크 계산
# ================================================================
def compute_rdkit_properties():
    """모든 통합 분자에 대해 RDKit 물리속성 계산"""
    print("\n=== RDKit 물리속성 계산 ===")
    if not HAS_RDKIT:
        print("  [SKIP] RDKit not available")
        return 0

    unified_path = PROCESSED_DIR / "unified_molecules.json"
    if not unified_path.exists():
        print("  [SKIP] Run --unify first")
        return 0

    with open(unified_path, encoding='utf-8') as f:
        molecules = json.load(f)

    properties = {}
    errors = 0
    for i, mol_data in enumerate(molecules):
        smi = mol_data['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            errors += 1
            continue
        try:
            props = {
                'mw': round(Descriptors.MolWt(mol), 2),
                'logp': round(Descriptors.MolLogP(mol), 3),
                'tpsa': round(Descriptors.TPSA(mol), 2),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'rotatable': Descriptors.NumRotatableBonds(mol),
                'rings': Descriptors.RingCount(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'fsp3': round(Descriptors.FractionCSP3(mol), 3),
                'has_chiral': int(len(Chem.FindMolChiralCenters(mol)) > 0),
                'n_chiral': len(Chem.FindMolChiralCenters(mol)),
            }
            properties[smi] = props
        except Exception:
            errors += 1

        if (i + 1) % 5000 == 0:
            print(f"  Progress: {i+1}/{len(molecules)}")

    print(f"  Computed properties for {len(properties):,} molecules ({errors} errors)")
    with open(PROCESSED_DIR / "rdkit_properties.json", 'w') as f:
        json.dump(properties, f, indent=2)
    return len(properties)


# ================================================================
# 메인
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="향 AI v3 데이터 수집")
    parser.add_argument('--all', action='store_true', help='전체 수집')
    parser.add_argument('--pyrfume', action='store_true')
    parser.add_argument('--flavordb', action='store_true')
    parser.add_argument('--dream', action='store_true')
    parser.add_argument('--m2or', action='store_true')
    parser.add_argument('--ifra', action='store_true')
    parser.add_argument('--pubchem', action='store_true')
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--unify', action='store_true')
    parser.add_argument('--rdkit', action='store_true')
    parser.add_argument('--max-pubchem', type=int, default=500, help='PubChem max queries')
    args = parser.parse_args()

    if args.all or not any([args.pyrfume, args.flavordb, args.dream, args.m2or,
                             args.ifra, args.pubchem, args.local, args.unify, args.rdkit]):
        args.pyrfume = args.dream = args.m2or = args.ifra = True
        args.local = args.pubchem = args.unify = args.rdkit = True

    results = {}
    t0 = time.time()

    if args.pyrfume:     results['pyrfume'] = collect_pyrfume()
    if args.flavordb:    results['flavordb'] = collect_flavordb()
    if args.dream:       results['dream'] = collect_dream()
    if args.m2or:        results['m2or'] = collect_m2or()
    if args.ifra:        results['ifra'] = collect_ifra()
    if args.local:       results['local'] = collect_local()
    if args.pubchem:     results['pubchem'] = collect_pubchem(max_queries=args.max_pubchem)
    if args.unify:       results['unified'] = unify_all()
    if args.rdkit:       results['rdkit'] = compute_rdkit_properties()

    elapsed = time.time() - t0
    print("\n" + "=" * 50)
    print(f"  수집 결과 ({elapsed:.0f}s):")
    for source, count in results.items():
        print(f"    {source}: {count:,}")
    print("=" * 50)


if __name__ == '__main__':
    main()
