"""
POM Upgrade Script -- PubChem SMILES + XGBoost ODT + Odor-Pair
==============================================================
1. PubChem PUG REST: recover missing SMILES (CAS/name -> SMILES)
2. XGBoost: predict ODT from 256d POM embeddings
3. Update pom_engine fragrance DB
"""
import os, sys, json, csv, time, math
import urllib.request
import urllib.parse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(BASE)  # server/
sys.path.insert(0, SERVER_DIR)
sys.path.insert(0, BASE)

# ============================================================
# Step 1: PubChem SMILES Recovery
# ============================================================

def get_smiles_pubchem(identifier, use_cas=False):
    """Query PubChem PUG REST for SMILES"""
    safe_id = urllib.parse.quote(str(identifier).strip())
    
    if use_cas:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_id}/property/IsomericSMILES/JSON"
    else:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{safe_id}/property/IsomericSMILES/JSON"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'POM-Engine/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data['PropertyTable']['Properties'][0]['IsomericSMILES']
    except Exception:
        return None


def recover_smiles(ingredients_path, output_path):
    """Recover SMILES for all ingredients missing them"""
    with open(ingredients_path, 'r', encoding='utf-8') as f:
        ingredients = json.load(f)
    
    # Also load known SMILES from pom_engine
    from pom_engine import _KNOWN_SMILES
    
    total = len(ingredients)
    has_smiles = 0
    recovered = 0
    failed_list = []
    smiles_map = {}  # id -> smiles
    
    for i, ing in enumerate(ingredients):
        ing_id = ing.get('id', '')
        cas = ing.get('cas_number', '')
        name_en = ing.get('name_en', ing_id)
        
        # Already have SMILES?
        existing = _KNOWN_SMILES.get(ing_id, '')
        if existing:
            smiles_map[ing_id] = existing
            has_smiles += 1
            continue
        
        # Try PubChem
        smiles = None
        
        # Try CAS first (most reliable)
        if cas and cas != '-':
            smiles = get_smiles_pubchem(cas, use_cas=True)
            time.sleep(0.25)
        
        # Try English name
        if not smiles and name_en:
            smiles = get_smiles_pubchem(name_en)
            time.sleep(0.25)
        
        # Try ingredient ID (underscore -> space)
        if not smiles:
            alt_name = ing_id.replace('_', ' ')
            smiles = get_smiles_pubchem(alt_name)
            time.sleep(0.25)
        
        if smiles:
            smiles_map[ing_id] = smiles
            recovered += 1
            has_smiles += 1
            if (recovered % 20 == 0):
                print(f"    [{recovered} recovered] {ing_id} -> {smiles[:40]}")
        else:
            failed_list.append({'id': ing_id, 'name': name_en, 'cas': cas})
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{total} ({has_smiles} have SMILES, {recovered} recovered)")
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'smiles_map': smiles_map,
            'failed': failed_list,
            'stats': {
                'total': total,
                'has_smiles': has_smiles,
                'recovered': recovered,
                'failed': len(failed_list),
                'coverage': round(has_smiles / total * 100, 1)
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n  [SMILES] Total: {total}, Has SMILES: {has_smiles} ({has_smiles/total*100:.1f}%)")
    print(f"  [SMILES] Recovered: {recovered}, Failed: {len(failed_list)}")
    print(f"  [SMILES] Saved to: {output_path}")
    
    return smiles_map, failed_list


# ============================================================
# Step 2: XGBoost ODT Prediction
# ============================================================

def train_odt_predictor(embedding_db_path, abraham_dir, output_path):
    """Train XGBoost to predict ODT from 256d POM embeddings"""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    import pickle
    
    # Load Abraham ODT data
    mol_path = os.path.join(abraham_dir, 'molecules.csv')
    beh_path = os.path.join(abraham_dir, 'behavior.csv')
    
    cid_to_smi = {}
    with open(mol_path, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid_to_smi[row['CID']] = row.get('IsomericSMILES', '')
    
    odt_data = []
    with open(beh_path, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid = row.get('Stimulus', '')
            odt_str = row.get('Log (1/ODT)', '')
            smi = cid_to_smi.get(cid, '')
            if smi and odt_str:
                try:
                    odt_data.append((smi, float(odt_str)))
                except:
                    pass
    
    print(f"  [ODT] Abraham data: {len(odt_data)} molecules with ODT")
    
    # Load embeddings DB
    emb_data = np.load(embedding_db_path, allow_pickle=True)
    emb_smiles = {str(s): i for i, s in enumerate(emb_data['smiles'])}
    embeddings = emb_data['embeddings']
    
    # Match Abraham ODT with embeddings
    X, y = [], []
    for smi, odt in odt_data:
        idx = emb_smiles.get(smi)
        if idx is not None:
            X.append(embeddings[idx])
            y.append(odt)
    
    X = np.array(X)
    y = np.array(y)
    print(f"  [ODT] Matched with embeddings: {len(X)}")
    
    if len(X) < 20:
        print("  [ODT] Not enough matched data, skipping")
        return None
    
    # Train GBR (more robust than XGBoost for small datasets)
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42
    )
    
    # Cross-validation
    scores = cross_val_score(model, X, y, cv=min(5, len(X)//5), scoring='r2')
    print(f"  [ODT] Cross-val R2: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Train on all data
    model.fit(X, y)
    
    # Predict for ALL molecules in embedding DB
    all_emb = emb_data['embeddings']
    all_smiles = emb_data['smiles']
    all_predictions = model.predict(all_emb)
    
    # Save model + predictions
    odt_map = {}
    for i, smi in enumerate(all_smiles):
        odt_map[str(smi)] = round(float(all_predictions[i]), 3)
    
    result = {
        'model_r2': round(float(scores.mean()), 4),
        'model_std': round(float(scores.std()), 4),
        'n_training': len(X),
        'n_predicted': len(all_smiles),
        'odt_map': odt_map,
        'stats': {
            'mean': round(float(np.mean(all_predictions)), 3),
            'std': round(float(np.std(all_predictions)), 3),
            'min': round(float(np.min(all_predictions)), 3),
            'max': round(float(np.max(all_predictions)), 3),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    # Also save pickle model
    model_pkl = output_path.replace('.json', '_model.pkl')
    with open(model_pkl, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  [ODT] Predicted ODT for {len(all_smiles)} molecules")
    print(f"  [ODT] R2={scores.mean():.4f}, range=[{np.min(all_predictions):.2f}, {np.max(all_predictions):.2f}]")
    print(f"  [ODT] Saved: {output_path}")
    
    return result


# ============================================================
# Step 3: Odor-Pair Dataset Download
# ============================================================

def download_odor_pair(output_dir):
    """Download Odor-Pair dataset from GitHub"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Try downloading the main CSV from the GitHub repo
    urls = [
        ("https://raw.githubusercontent.com/odor-pair/odor-pair/main/data/odor_pairs.csv",
         "odor_pairs.csv"),
        ("https://raw.githubusercontent.com/odor-pair/odor-pair/main/data/molecules.csv",
         "odor_pair_molecules.csv"),
    ]
    
    downloaded = 0
    for url, filename in urls:
        fpath = os.path.join(output_dir, filename)
        try:
            print(f"  [PAIR] Downloading {filename}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'POM-Engine/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                with open(fpath, 'wb') as f:
                    f.write(data)
                downloaded += 1
                print(f"  [PAIR] Saved: {fpath} ({len(data)} bytes)")
        except Exception as e:
            print(f"  [PAIR] Failed to download {filename}: {e}")
    
    return downloaded


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    t0 = time.time()
    
    print("=" * 60)
    print("  POM Engine v3 -- Data Quality Upgrade")
    print("=" * 60)
    
    data_dir = os.path.join(SERVER_DIR, 'data', 'pom_upgrade')
    os.makedirs(data_dir, exist_ok=True)
    
    # === Step 1: PubChem SMILES ===
    print("\n--- Step 1: PubChem SMILES Recovery ---")
    ing_path = os.path.join(SERVER_DIR, '..', 'data', 'ingredients.json')
    if not os.path.exists(ing_path):
        ing_path = r'C:\Users\user\Desktop\Game\data\ingredients.json'
    
    smiles_out = os.path.join(data_dir, 'pubchem_smiles.json')
    
    if os.path.exists(smiles_out):
        print(f"  [SKIP] Already exists: {smiles_out}")
        with open(smiles_out, 'r') as f:
            existing = json.load(f)
        print(f"  Stats: {existing.get('stats', {})}")
    else:
        smiles_map, failed = recover_smiles(ing_path, smiles_out)
    
    # === Step 2: XGBoost ODT ===
    print("\n--- Step 2: XGBoost ODT Prediction ---")
    emb_path = os.path.join(SERVER_DIR, 'models', 'openpom_ensemble', 'pom_embeddings.npz')
    abr_dir = os.path.join(SERVER_DIR, 'data', 'pom_data', 'pyrfume_all', 'abraham_2012')
    odt_out = os.path.join(data_dir, 'predicted_odt.json')
    
    if os.path.exists(odt_out):
        print(f"  [SKIP] Already exists: {odt_out}")
        with open(odt_out, 'r') as f:
            odt_data = json.load(f)
        print(f"  R2={odt_data.get('model_r2')}, N={odt_data.get('n_predicted')}")
    else:
        odt_result = train_odt_predictor(emb_path, abr_dir, odt_out)
    
    # === Step 3: Odor-Pair Dataset ===
    print("\n--- Step 3: Odor-Pair Dataset ---")
    pair_dir = os.path.join(data_dir, 'odor_pair')
    if os.path.exists(os.path.join(pair_dir, 'odor_pairs.csv')):
        print(f"  [SKIP] Already exists: {pair_dir}")
    else:
        n_downloaded = download_odor_pair(pair_dir)
        print(f"  Downloaded: {n_downloaded} files")
    
    elapsed = time.time() - t0
    print(f"\n[OK] Upgrade complete ({elapsed:.1f}s)")
