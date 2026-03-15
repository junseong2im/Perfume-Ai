"""
===================================================================
Phase 2: POM Embedding Transfer Learning
===================================================================
1. Load 10 trained ensemble models from Phase 1
2. Extract 256d POM embeddings for all molecules (average across 10 models)
3. Train XGBoost on POM embeddings -> 138 odor descriptors
4. Compare AUROC: MPNN direct vs POM+XGBoost
===================================================================
"""
import os
import sys
import csv
import json
import time
import re
import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from rdkit import Chem

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

# Import shared config from train_openpom
TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
n_tasks = len(TASKS)
task_to_idx = {t: i for i, t in enumerate(TASKS)}
task_regexes = {t: re.compile(rf'\b{re.escape(t)}\b', re.IGNORECASE) for t in TASKS}


def canonicalize_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None


def load_gs_lf_merged():
    """Same data loading as train_openpom.py (canonical + regex)"""
    all_smiles = {}
    
    # Leffingwell
    lf_mol = os.path.join(DATA_DIR, 'pyrfume_all', 'leffingwell', 'molecules.csv')
    lf_beh = os.path.join(DATA_DIR, 'pyrfume_all', 'leffingwell', 'behavior.csv')
    cid_to_smiles = {}
    with open(lf_mol, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid, smi = row.get('CID', ''), row.get('IsomericSMILES', '')
            if cid and smi:
                c = canonicalize_smiles(smi)
                if c: cid_to_smiles[str(cid)] = c
    
    with open(lf_beh, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        lf_descs = [c for c in reader.fieldnames if c != 'Stimulus']
        for row in reader:
            smi = cid_to_smiles.get(str(row.get('Stimulus', '')), '')
            if not smi: continue
            label = np.zeros(n_tasks, dtype=np.float32)
            for desc in lf_descs:
                dl = desc.lower().strip()
                if dl in task_to_idx:
                    try:
                        if float(row.get(desc, '0')) > 0:
                            label[task_to_idx[dl]] = 1.0
                    except: pass
            if smi not in all_smiles: all_smiles[smi] = label
            else: all_smiles[smi] = np.maximum(all_smiles[smi], label)
    
    # GoodScents
    gs_mol = os.path.join(DATA_DIR, 'pyrfume_all', 'goodscents', 'molecules.csv')
    gs_beh = os.path.join(DATA_DIR, 'pyrfume_all', 'goodscents', 'behavior.csv')
    cas_map_path = os.path.join(DATA_DIR, 'pyrfume_all', 'goodscents', 'cas_to_cid.json')
    
    gs_cid_to_smiles = {}
    with open(gs_mol, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid, smi = row.get('CID', ''), row.get('IsomericSMILES', '')
            if cid and smi:
                c = canonicalize_smiles(smi)
                if c: gs_cid_to_smiles[str(cid)] = c
    
    cas_to_cid = {}
    if os.path.exists(cas_map_path):
        with open(cas_map_path, 'r') as f:
            cas_to_cid = json.load(f)
    
    with open(gs_beh, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            stimulus = row.get('Stimulus', '').strip()
            cid = cas_to_cid.get(stimulus, '')
            smi = gs_cid_to_smiles.get(str(cid), '') if cid else ''
            if not smi: continue
            desc_text = row.get('Descriptors', '')
            if not desc_text: continue
            label = np.zeros(n_tasks, dtype=np.float32)
            descs = [d.strip().lower() for d in desc_text.replace(';', ',').split(',')]
            matched = 0
            for d in descs:
                for task, rx in task_regexes.items():
                    if rx.search(d):
                        label[task_to_idx[task]] = 1.0
                        matched += 1
            if matched > 0:
                if smi not in all_smiles: all_smiles[smi] = label
                else: all_smiles[smi] = np.maximum(all_smiles[smi], label)
    
    smiles_list = list(all_smiles.keys())
    labels = np.array([all_smiles[s] for s in smiles_list], dtype=np.float32)
    return smiles_list, labels


def extract_pom_embeddings(smiles_list, labels, n_models=10):
    """Extract 256d POM embeddings from trained ensemble models"""
    print("\n" + "=" * 60)
    print("  [Phase 2] Extracting 256d POM Embeddings")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    featurizer = GraphFeaturizer()
    
    # Featurize all molecules
    print(f"  Featurizing {len(smiles_list)} molecules...")
    valid_features = []
    valid_labels = []
    valid_smi = []
    for i, smi in enumerate(smiles_list):
        try:
            feat = featurizer.featurize([smi])
            if feat is not None and len(feat) > 0:
                f = feat[0]
                if f is not None and hasattr(f, 'node_features') and f.node_features.shape[0] > 0:
                    valid_features.append(f)
                    valid_labels.append(labels[i])
                    valid_smi.append(smi)
        except: pass
    
    features = np.array(valid_features, dtype=object)
    labels_valid = np.array(valid_labels, dtype=np.float32)
    print(f"  Valid: {len(features)} molecules")
    
    # Create dataset
    dataset = dc.data.NumpyDataset(
        X=features, y=labels_valid,
        w=np.ones_like(labels_valid),
        ids=np.array(valid_smi))
    
    # Compute class imbalance for model init
    pos_counts = labels_valid.sum(axis=0)
    neg_counts = len(labels_valid) - pos_counts
    class_imbalance_ratio = list((neg_counts / (pos_counts + 1)).clip(max=50))
    
    # Extract embeddings from each model
    all_embeddings = []
    loaded_models = 0
    
    for i in range(n_models):
        model_dir = os.path.join(MODEL_DIR, 'openpom_ensemble', f'experiments_{i+1}')
        checkpoint_path = os.path.join(model_dir, 'checkpoint1.pt')
        
        if not os.path.exists(checkpoint_path):
            print(f"  Model {i+1}: checkpoint not found, skipping")
            continue
        
        # Create model with same architecture
        model = MPNNPOMModel(
            n_tasks=n_tasks,
            batch_size=128,
            class_imbalance_ratio=class_imbalance_ratio,
            loss_aggr_type='sum',
            node_out_feats=100,
            edge_hidden_feats=75,
            edge_out_feats=100,
            num_step_message_passing=5,
            mpnn_residual=True,
            message_aggregator_type='sum',
            mode='classification',
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1,
            readout_type='set2set',
            num_step_set2set=3,
            num_layer_set2set=2,
            ffn_hidden_list=[392, 392],
            ffn_embeddings=256,
            ffn_activation='relu',
            ffn_dropout_p=0.12,
            ffn_dropout_at_input_no_act=False,
            weight_decay=1e-5,
            self_loop=False,
            optimizer_name='adam',
            model_dir=model_dir,
            device_name=device,
        )
        
        # Restore best checkpoint
        model.restore(model_dir=model_dir)
        
        # Extract embeddings using predict_embedding
        embeddings = model.predict_embedding(dataset)
        
        if isinstance(embeddings, list):
            embeddings = embeddings[0]
        
        print(f"  Model {i+1}: embeddings shape = {embeddings.shape}")
        all_embeddings.append(embeddings)
        loaded_models += 1
    
    if loaded_models == 0:
        print("  ERROR: No models loaded!")
        return None, None, None, None
    
    # Average embeddings across models (ensemble POM)
    ensemble_embeddings = np.mean(all_embeddings, axis=0)
    print(f"\n  Ensemble POM: {ensemble_embeddings.shape} ({loaded_models} models averaged)")
    
    return ensemble_embeddings, labels_valid, valid_smi, features


def train_downstream_models(embeddings, labels, smiles_list, features):
    """Train XGBoost, LightGBM, RF, MLP on POM embeddings"""
    print("\n" + "=" * 60)
    print("  [Phase 2] Downstream Model Training")
    print("=" * 60)
    
    # Scaffold split (same seed=42 as Phase 1)
    dataset = dc.data.NumpyDataset(
        X=features, y=labels,
        w=np.ones_like(labels),
        ids=np.array(smiles_list))
    
    splitter = dc.splits.ScaffoldSplitter()
    try:
        train_ds, val_ds, test_ds = splitter.train_valid_test_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)
        train_idx = [list(smiles_list).index(s) for s in train_ds.ids]
        val_idx = [list(smiles_list).index(s) for s in val_ds.ids]
        test_idx = [list(smiles_list).index(s) for s in test_ds.ids]
    except Exception as e:
        print(f"  Scaffold split failed ({e}), using random")
        n = len(embeddings)
        idx = np.random.RandomState(42).permutation(n)
        n_train, n_val = int(n * 0.8), int(n * 0.1)
        train_idx = idx[:n_train].tolist()
        val_idx = idx[n_train:n_train+n_val].tolist()
        test_idx = idx[n_train+n_val:].tolist()
    
    X_train = embeddings[train_idx]
    X_val = embeddings[val_idx]
    X_test = embeddings[test_idx]
    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]
    
    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Embedding dim: {X_train.shape[1]}")
    
    results = {}
    
    # --- 1. XGBoost ---
    if HAS_XGB:
        print("\n  --- XGBoost ---")
        start = time.time()
        xgb_preds = np.zeros_like(y_test)
        xgb_aurocs = []
        
        for j in range(n_tasks):
            if len(np.unique(y_train[:, j])) < 2:
                continue
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric='logloss',
                verbosity=0, n_jobs=-1)
            clf.fit(X_train, y_train[:, j])
            pred = clf.predict_proba(X_test)
            if pred.shape[1] == 2:
                xgb_preds[:, j] = pred[:, 1]
            else:
                xgb_preds[:, j] = pred[:, 0]
            
            if len(np.unique(y_test[:, j])) >= 2:
                auc = roc_auc_score(y_test[:, j], xgb_preds[:, j])
                xgb_aurocs.append(auc)
        
        xgb_auroc = np.mean(xgb_aurocs) if xgb_aurocs else 0
        xgb_time = time.time() - start
        results['XGBoost'] = {'auroc': xgb_auroc, 'n_tasks': len(xgb_aurocs), 'time': xgb_time}
        print(f"  XGBoost Test AUROC: {xgb_auroc:.4f} ({len(xgb_aurocs)} tasks, {xgb_time:.1f}s)")
    
    # --- 2. Random Forest ---
    print("\n  --- Random Forest ---")
    start = time.time()
    rf_preds = np.zeros_like(y_test)
    rf_aurocs = []
    
    for j in range(n_tasks):
        if len(np.unique(y_train[:, j])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train[:, j])
        pred = clf.predict_proba(X_test)
        if pred.shape[1] == 2:
            rf_preds[:, j] = pred[:, 1]
        else:
            rf_preds[:, j] = pred[:, 0]
        
        if len(np.unique(y_test[:, j])) >= 2:
            auc = roc_auc_score(y_test[:, j], rf_preds[:, j])
            rf_aurocs.append(auc)
    
    rf_auroc = np.mean(rf_aurocs) if rf_aurocs else 0
    rf_time = time.time() - start
    results['RandomForest'] = {'auroc': rf_auroc, 'n_tasks': len(rf_aurocs), 'time': rf_time}
    print(f"  RF Test AUROC: {rf_auroc:.4f} ({len(rf_aurocs)} tasks, {rf_time:.1f}s)")
    
    # --- 3. MLP ---
    print("\n  --- MLP ---")
    start = time.time()
    mlp_preds = np.zeros_like(y_test)
    mlp_aurocs = []
    
    for j in range(n_tasks):
        if len(np.unique(y_train[:, j])) < 2:
            continue
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                           random_state=42, early_stopping=True, verbose=False)
        clf.fit(X_train, y_train[:, j])
        pred = clf.predict_proba(X_test)
        if pred.shape[1] == 2:
            mlp_preds[:, j] = pred[:, 1]
        else:
            mlp_preds[:, j] = pred[:, 0]
        
        if len(np.unique(y_test[:, j])) >= 2:
            auc = roc_auc_score(y_test[:, j], mlp_preds[:, j])
            mlp_aurocs.append(auc)
    
    mlp_auroc = np.mean(mlp_aurocs) if mlp_aurocs else 0
    mlp_time = time.time() - start
    results['MLP'] = {'auroc': mlp_auroc, 'n_tasks': len(mlp_aurocs), 'time': mlp_time}
    print(f"  MLP Test AUROC: {mlp_auroc:.4f} ({len(mlp_aurocs)} tasks, {mlp_time:.1f}s)")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("  [Phase 2] Results Summary")
    print("=" * 60)
    print(f"  {'Model':<20} {'AUROC':>8} {'Tasks':>6} {'Time':>8}")
    print(f"  {'-'*44}")
    print(f"  {'MPNN Ensemble (P1)':<20} {'0.7890':>8} {'128':>6} {'16.9m':>8}")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['auroc']):
        print(f"  {f'POM+{name}':<20} {r['auroc']:>8.4f} {r['n_tasks']:>6} {r['time']:>7.1f}s")
    
    # Save results
    results_path = os.path.join(MODEL_DIR, 'openpom_ensemble', 'phase2_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'phase1_ensemble_auroc': 0.7890,
            'phase2_results': {k: {'auroc': float(v['auroc']), 'n_tasks': v['n_tasks'], 
                                    'time': float(v['time'])} for k, v in results.items()},
            'embedding_dim': 256,
            'n_molecules': len(embeddings),
        }, f, indent=2)
    print(f"\n  Results saved: {results_path}")
    
    # Save embeddings for future use
    emb_path = os.path.join(MODEL_DIR, 'openpom_ensemble', 'pom_embeddings.npz')
    np.savez(emb_path, 
             embeddings=embeddings, 
             labels=labels,
             smiles=np.array(smiles_list))
    print(f"  Embeddings saved: {emb_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=int, default=10)
    args = parser.parse_args()
    
    # 1. Load data
    smiles_list, labels = load_gs_lf_merged()
    print(f"  Loaded {len(smiles_list)} molecules x {n_tasks} tasks")
    
    # 2. Extract POM embeddings
    embeddings, labels_valid, valid_smi, features = extract_pom_embeddings(
        smiles_list, labels, n_models=args.models)
    
    if embeddings is not None:
        # 3. Train downstream models
        results = train_downstream_models(embeddings, labels_valid, valid_smi, features)
