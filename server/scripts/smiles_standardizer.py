"""
SMILES Standardizer + Embedding Expansion
==========================================
Phase 1: Production-grade data pipeline
1. Extract unique SMILES from 166K Odor-Pair
2. Canonicalize ALL SMILES via RDKit
3. Identify missing molecules (not in 5,050 DB)
4. Generate 256d embeddings for missing via Phase 1 ensemble
5. Save expanded pom_embeddings_v2.npz
"""
import os, sys, json, time, csv
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(BASE)
sys.path.insert(0, SERVER_DIR)

# ============================================================
# Step 1: RDKit Canonicalization
# ============================================================

def canonicalize_smiles(smiles):
    """Canonicalize SMILES using RDKit"""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    
    if not smiles or not smiles.strip():
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip(), sanitize=True)
        if mol is None:
            return None
        
        # Cleanup (normalize/reionize)
        try:
            mol = rdMolStandardize.Cleanup(mol)
        except:
            pass
        
        # Parent fragment (remove salts)
        try:
            mol = rdMolStandardize.FragmentParent(mol)
        except:
            try:
                chooser = rdMolStandardize.LargestFragmentChooser()
                mol = chooser.choose(mol)
            except:
                pass
        
        # Uncharge
        try:
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
        except:
            pass
        
        # Canonical SMILES
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return can
    except Exception:
        return None


def get_inchikey(smiles):
    """Get InChIKey for a SMILES"""
    from rdkit import Chem
    try:
        from rdkit.Chem.inchi import MolToInchiKey
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol:
            return MolToInchiKey(mol)
    except:
        pass
    return None


# ============================================================
# Step 2: Extract & Diff
# ============================================================

def extract_unique_smiles_from_odor_pair(data_path):
    """Extract all unique SMILES from Odor-Pair JSON"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    all_smiles = set()
    for item in data:
        s1 = item.get('mol1', '')
        s2 = item.get('mol2', '')
        if s1 and s1.strip():
            all_smiles.add(s1.strip())
        if s2 and s2.strip():
            all_smiles.add(s2.strip())
    
    return all_smiles, len(data)


def load_existing_db(emb_path):
    """Load existing embedding DB and build canonical index"""
    data = np.load(emb_path, allow_pickle=True)
    smiles_list = [str(s) for s in data['smiles']]
    embeddings = data['embeddings']
    labels = data['labels'] if 'labels' in data else None
    
    # Build canonical index
    canonical_map = {}  # canonical -> original_idx
    for i, smi in enumerate(smiles_list):
        can = canonicalize_smiles(smi)
        if can:
            canonical_map[can] = i
        # Also store raw for fallback
        canonical_map[smi] = i
    
    return {
        'smiles': smiles_list,
        'embeddings': embeddings,
        'labels': labels,
        'canonical_map': canonical_map,
    }


# ============================================================
# Step 3: Batch Embedding Generation  
# ============================================================

def generate_embeddings_batch(smiles_list, model_dir, device='cpu', batch_size=128):
    """Generate 256d embeddings for new molecules using Phase 1 ensemble"""
    try:
        from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
        from openpom.models.mpnn_pom import MPNNPOMModel
        import deepchem as dc
    except ImportError as e:
        print(f"  [ERROR] OpenPOM not available: {e}")
        return None
    
    featurizer = GraphFeaturizer()
    n_tasks = 138
    
    # Load first model (for embedding extraction)
    exp_dir = os.path.join(model_dir, 'experiments_1')
    if not os.path.exists(os.path.join(exp_dir, 'checkpoint1.pt')):
        print(f"  [ERROR] No model checkpoint at {exp_dir}")
        return None
    
    model = MPNNPOMModel(
        n_tasks=n_tasks, batch_size=batch_size,
        class_imbalance_ratio=[1.0]*n_tasks,
        loss_aggr_type='sum', node_out_feats=100,
        edge_hidden_feats=75, edge_out_feats=100,
        num_step_message_passing=5, mpnn_residual=True,
        message_aggregator_type='sum', mode='classification',
        number_atom_features=GraphConvConstants.ATOM_FDIM,
        number_bond_features=GraphConvConstants.BOND_FDIM,
        n_classes=1, readout_type='set2set',
        num_step_set2set=3, num_layer_set2set=2,
        ffn_hidden_list=[392, 392], ffn_embeddings=256,
        ffn_activation='relu', ffn_dropout_p=0.12,
        ffn_dropout_at_input_no_act=False, weight_decay=1e-5,
        self_loop=False, optimizer_name='adam',
        model_dir=exp_dir, device_name=device,
    )
    model.restore(model_dir=exp_dir)
    
    import torch
    
    all_embeddings = []
    valid_smiles = []
    failed = 0
    
    for batch_start in range(0, len(smiles_list), batch_size):
        batch_smi = smiles_list[batch_start:batch_start + batch_size]
        batch_embs = []
        
        for smi in batch_smi:
            try:
                feat = featurizer.featurize([smi])
                ds = dc.data.NumpyDataset(X=feat,
                    y=np.zeros((1, n_tasks)),
                    w=np.ones((1, n_tasks)))
                
                # Extract embedding from FFN layer
                emb = model.predict_embedding(ds)
                if emb is not None and len(emb) > 0 and np.linalg.norm(emb[0]) > 0:
                    batch_embs.append(emb[0])
                    valid_smiles.append(smi)
                else:
                    failed += 1
            except Exception:
                failed += 1
        
        if batch_embs:
            all_embeddings.extend(batch_embs)
        
        if (batch_start + batch_size) % 500 == 0 or batch_start + batch_size >= len(smiles_list):
            print(f"    Progress: {min(batch_start + batch_size, len(smiles_list))}/{len(smiles_list)} "
                  f"(valid: {len(valid_smiles)}, failed: {failed})")
    
    if all_embeddings:
        return np.array(all_embeddings), valid_smiles
    return None, []


# ============================================================
# Main Pipeline
# ============================================================

if __name__ == '__main__':
    t0 = time.time()
    
    print("=" * 60)
    print("  Phase 1: SMILES Standardization + Embedding Expansion")
    print("=" * 60)
    
    # === Step 1: Extract unique SMILES from Odor-Pair ===
    print("\n--- Step 1: Extract unique SMILES from Odor-Pair ---")
    odor_pair_path = os.path.join(SERVER_DIR, 'data', 'pom_upgrade', 'odor_pair', 'full.json')
    
    pair_smiles, n_pairs = extract_unique_smiles_from_odor_pair(odor_pair_path)
    print(f"  Odor-Pair: {n_pairs} pairs -> {len(pair_smiles)} unique SMILES")
    
    # === Step 2: Canonicalize all SMILES ===
    print("\n--- Step 2: Canonicalize SMILES (RDKit) ---")
    canonical_set = {}  # canonical -> [raw_smiles_list]
    failed_parse = 0
    
    for smi in pair_smiles:
        can = canonicalize_smiles(smi)
        if can:
            if can not in canonical_set:
                canonical_set[can] = []
            canonical_set[can].append(smi)
        else:
            failed_parse += 1
    
    print(f"  Canonical: {len(canonical_set)} unique molecules")
    print(f"  Failed to parse: {failed_parse}")
    
    # Also canonicalize known SMILES from pom_engine
    from pom_engine import _KNOWN_SMILES
    for name, smi in _KNOWN_SMILES.items():
        can = canonicalize_smiles(smi)
        if can and can not in canonical_set:
            canonical_set[can] = [smi]
    
    # === Step 3: Load existing DB and find missing ===
    print("\n--- Step 3: Diff against existing embedding DB ---")
    emb_path = os.path.join(SERVER_DIR, 'models', 'openpom_ensemble', 'pom_embeddings.npz')
    existing = load_existing_db(emb_path)
    
    existing_canonical = set()
    for smi in existing['smiles']:
        can = canonicalize_smiles(smi)
        if can:
            existing_canonical.add(can)
        existing_canonical.add(smi)  # also raw
    
    missing = []
    for can_smi in canonical_set:
        if can_smi not in existing_canonical:
            missing.append(can_smi)
    
    print(f"  Existing DB: {len(existing['smiles'])} molecules")
    print(f"  Odor-Pair unique: {len(canonical_set)}")
    print(f"  Already in DB: {len(canonical_set) - len(missing)}")
    print(f"  Missing (need embeddings): {len(missing)}")
    
    # === Step 4: Generate embeddings for missing ===
    print(f"\n--- Step 4: Generate 256d embeddings for {len(missing)} molecules ---")
    
    model_dir = os.path.join(SERVER_DIR, 'models', 'openpom_ensemble')
    device = 'cpu'  # GPU may OOM with 10-model ensemble, CPU is safer for batch
    
    if len(missing) > 0:
        new_embeddings, new_smiles = generate_embeddings_batch(
            missing, model_dir, device=device, batch_size=64
        )
        
        if new_embeddings is not None and len(new_smiles) > 0:
            print(f"  Generated: {len(new_smiles)} new embeddings")
            
            # === Step 5: Merge and save expanded DB ===
            print("\n--- Step 5: Save expanded embedding DB ---")
            
            merged_embeddings = np.concatenate([
                existing['embeddings'],
                new_embeddings
            ], axis=0)
            
            merged_smiles = existing['smiles'] + new_smiles
            
            # Labels: extend with empty
            if existing['labels'] is not None:
                n_new = len(new_smiles)
                n_labels = existing['labels'].shape[1] if len(existing['labels'].shape) > 1 else 138
                new_labels = np.zeros((n_new, n_labels))
                merged_labels = np.concatenate([existing['labels'], new_labels], axis=0)
            else:
                merged_labels = np.zeros((len(merged_smiles), 138))
            
            # Save
            out_path = os.path.join(SERVER_DIR, 'models', 'openpom_ensemble', 'pom_embeddings_v2.npz')
            np.savez_compressed(out_path,
                embeddings=merged_embeddings,
                smiles=np.array(merged_smiles),
                labels=merged_labels
            )
            
            print(f"  Saved: {out_path}")
            print(f"  DB size: {len(existing['smiles'])} -> {len(merged_smiles)} molecules")
            print(f"  Embedding shape: {merged_embeddings.shape}")
        else:
            print("  [WARN] No new embeddings generated")
    else:
        print("  No missing molecules, DB is complete")
    
    # === Step 6: Build canonical mapping for Odor-Pair matching ===
    print("\n--- Step 6: Build canonical mapping ---")
    
    # Load final DB
    final_db_path = os.path.join(SERVER_DIR, 'models', 'openpom_ensemble', 'pom_embeddings_v2.npz')
    if not os.path.exists(final_db_path):
        final_db_path = emb_path
    
    final_data = np.load(final_db_path, allow_pickle=True)
    final_smiles = [str(s) for s in final_data['smiles']]
    
    # Build canonical -> index mapping
    can_to_idx = {}
    for i, smi in enumerate(final_smiles):
        can = canonicalize_smiles(smi)
        if can:
            can_to_idx[can] = i
        can_to_idx[smi] = i  # raw fallback
    
    # Test Odor-Pair matching rate with new DB
    with open(odor_pair_path, 'r') as f:
        test_data = json.load(f)
    
    matched_pairs = 0
    total_pairs = len(test_data)
    
    for item in test_data[:10000]:  # sample first 10K for speed
        s1 = canonicalize_smiles(item.get('mol1', ''))
        s2 = canonicalize_smiles(item.get('mol2', ''))
        if s1 and s2 and s1 in can_to_idx and s2 in can_to_idx:
            matched_pairs += 1
    
    match_rate = matched_pairs / min(total_pairs, 10000) * 100
    print(f"  Matching rate (sample 10K): {matched_pairs}/{min(total_pairs, 10000)} = {match_rate:.1f}%")
    
    # Save canonical mapping
    mapping_path = os.path.join(SERVER_DIR, 'data', 'pom_upgrade', 'canonical_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump({
            'total_molecules': len(final_smiles),
            'canonical_to_idx': {k: v for k, v in list(can_to_idx.items())[:100]},  # sample
            'match_rate_10k': match_rate,
        }, f, indent=2)
    
    elapsed = time.time() - t0
    print(f"\n[OK] Phase 1 complete ({elapsed:.1f}s)")
