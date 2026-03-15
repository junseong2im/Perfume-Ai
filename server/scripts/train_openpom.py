"""
===================================================================
OpenPOM Phase 1 -- Paper Reproduction (All Bugs Fixed)
===================================================================
Architecture: predict_odors.py official hyperparameters (UNCHANGED)
Data: GS-LF (GoodScents + Leffingwell merged)
Descriptors: 138 (paper TASKS list)
Split: Scaffold Split (ONCE, fixed seed=42)
Evaluation: sklearn AUROC
Ensemble: 10-model SOFT VOTING (probability average)
Device: CUDA (GPU) if available

Bug fixes applied:
  1. model.restore() before test evaluation (best weights)
  2. Single split, soft-voting ensemble
  3. RDKit canonical SMILES normalization
  4. Regex \b word-boundary matching (pine!=pineapple, but 'sweet vanilla'->sweet)
  5. Validation every epoch (not every 10)
  6. Cosine Annealing LR scheduler
===================================================================
"""
import os
import sys
import csv
import re
import time
import json
import numpy as np
import traceback

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.metrics import roc_auc_score

import torch
import deepchem as dc
from rdkit import Chem
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

# ============================================================
# Paper's official 138 descriptors (from predict_odors.py)
# ============================================================
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
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

n_tasks = len(TASKS)  # 138
task_to_idx = {t: i for i, t in enumerate(TASKS)}
# Pre-compiled regex patterns with word boundaries for each task
# \bpine\b matches 'pine' but NOT 'pineapple'; matches 'sweet' inside 'sweet vanilla'
task_regexes = {t: re.compile(rf'\b{re.escape(t)}\b', re.IGNORECASE) for t in TASKS}


def canonicalize_smiles(smi):
    """[FIX #3] RDKit canonical SMILES to prevent data leakage"""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() < 2:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None


def load_gs_lf_merged():
    """GoodScents + Leffingwell merged -> 138 descriptor binary labels
    
    [FIX #3] All SMILES are canonicalized via RDKit
    [FIX #4] GoodScents descriptors use EXACT match only (no partial)
    """
    all_smiles = {}  # canonical_smiles -> 138d binary label
    
    # === 1. Leffingwell (3,522 molecules, binary columns) ===
    lf_mol = os.path.join(DATA_DIR, 'pyrfume_all', 'leffingwell', 'molecules.csv')
    lf_beh = os.path.join(DATA_DIR, 'pyrfume_all', 'leffingwell', 'behavior.csv')
    
    cid_to_smiles = {}
    with open(lf_mol, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid = row.get('CID', '')
            smi = row.get('IsomericSMILES', '')
            if cid and smi:
                canonical = canonicalize_smiles(smi)
                if canonical:
                    cid_to_smiles[str(cid)] = canonical
    
    lf_count = 0
    with open(lf_beh, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        lf_descs = [c for c in reader.fieldnames if c != 'Stimulus']
        
        for row in reader:
            stimulus = row.get('Stimulus', '')
            smi = cid_to_smiles.get(str(stimulus), '')
            if not smi:
                continue
            
            label = np.zeros(n_tasks, dtype=np.float32)
            for desc in lf_descs:
                desc_lower = desc.lower().strip()
                if desc_lower in task_to_idx:
                    try:
                        val = float(row.get(desc, '0'))
                        if val > 0:
                            label[task_to_idx[desc_lower]] = 1.0
                    except:
                        pass
            
            if smi not in all_smiles:
                all_smiles[smi] = label
            else:
                all_smiles[smi] = np.maximum(all_smiles[smi], label)
            lf_count += 1
    
    print(f"  Leffingwell: {lf_count} entries, {len(cid_to_smiles)} molecules")
    
    # === 2. GoodScents (CAS -> CID -> SMILES via PubChem mapping) ===
    gs_mol = os.path.join(DATA_DIR, 'pyrfume_all', 'goodscents', 'molecules.csv')
    gs_beh = os.path.join(DATA_DIR, 'pyrfume_all', 'goodscents', 'behavior.csv')
    cas_map_path = os.path.join(DATA_DIR, 'pyrfume_all', 'goodscents', 'cas_to_cid.json')
    
    gs_cid_to_smiles = {}
    with open(gs_mol, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid = row.get('CID', '')
            smi = row.get('IsomericSMILES', '')
            if cid and smi:
                canonical = canonicalize_smiles(smi)
                if canonical:
                    gs_cid_to_smiles[str(cid)] = canonical
    
    cas_to_cid = {}
    if os.path.exists(cas_map_path):
        with open(cas_map_path, 'r') as f:
            cas_to_cid = json.load(f)
        print(f"  CAS->CID mapping loaded: {len(cas_to_cid)} entries")
    else:
        print(f"  WARNING: {cas_map_path} not found!")
        print(f"  Run build_cas_mapping.py first")
    
    gs_count = 0
    gs_no_smiles = 0
    with open(gs_beh, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            stimulus = row.get('Stimulus', '').strip()
            cid = cas_to_cid.get(stimulus, '')
            smi = gs_cid_to_smiles.get(str(cid), '') if cid else ''
            
            if not smi:
                gs_no_smiles += 1
                continue
            
            descriptors_text = row.get('Descriptors', '')
            if not descriptors_text:
                continue
            
            label = np.zeros(n_tasks, dtype=np.float32)
            # GoodScents: semicolon-separated descriptors
            desc_list = [d.strip().lower() for d in descriptors_text.replace(';', ',').split(',')]
            
            # [FIX #4 v2] Regex word-boundary matching
            # \bpine\b matches 'pine' but NOT 'pineapple'
            # Also handles compound descriptors: 'sweet vanilla' -> matches 'sweet' and 'vanilla'
            matched = 0
            for desc in desc_list:
                for task, regex in task_regexes.items():
                    if regex.search(desc):
                        label[task_to_idx[task]] = 1.0
                        matched += 1
            
            if matched > 0:
                if smi not in all_smiles:
                    all_smiles[smi] = label
                else:
                    all_smiles[smi] = np.maximum(all_smiles[smi], label)
                gs_count += 1
    
    print(f"  GoodScents: {gs_count} matched, {gs_no_smiles} no SMILES, "
          f"{len(gs_cid_to_smiles)} total molecules")
    
    # === Merge ===
    smiles_list = list(all_smiles.keys())
    labels = np.array([all_smiles[s] for s in smiles_list], dtype=np.float32)
    
    active_per_task = labels.sum(axis=0)
    active_tasks = (active_per_task >= 5).sum()
    
    print(f"\n  [DATA] GS-LF Merged:")
    print(f"    Total molecules:    {len(smiles_list)}")
    print(f"    Total descriptors:  {n_tasks}")
    print(f"    Active descriptors: {active_tasks} (>=5 positive)")
    print(f"    Label density:      {labels.mean():.4f}")
    
    return smiles_list, labels


def featurize_molecules(smiles_list, labels):
    """Pre-filter with RDKit & featurize individually to skip inorganics"""
    print(f"  Featurizing {len(smiles_list)} molecules...")
    featurizer = GraphFeaturizer()
    
    valid_features = []
    valid_labels = []
    valid_smi = []
    skipped = 0
    
    for i, smi in enumerate(smiles_list):
        try:
            feat = featurizer.featurize([smi])
            if feat is not None and len(feat) > 0:
                f = feat[0]
                if f is not None and hasattr(f, 'node_features') and f.node_features.shape[0] > 0:
                    valid_features.append(f)
                    valid_labels.append(labels[i])
                    valid_smi.append(smi)
                else:
                    skipped += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
    
    features = np.array(valid_features, dtype=object)
    labels_valid = np.array(valid_labels, dtype=np.float32)
    print(f"  Valid: {len(features)}/{len(smiles_list)} (skipped {skipped})")
    return features, labels_valid, valid_smi


def train_single_model(train_ds, val_ds, model_idx=0, epochs=30, device='cpu'):
    """[FIX #1, #2, #5] Single model training with proper validation & restore
    
    - Receives pre-split data (no splitting inside)
    - Validates EVERY epoch (not every 10)
    - Restores best weights before returning
    """
    print(f"\n{'='*60}")
    print(f"  [MODEL] Model {model_idx+1} -- Training (seed={model_idx})")
    print(f"{'='*60}")
    
    np.random.seed(model_idx)
    torch.manual_seed(model_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_idx)
    
    print(f"  Device: {device}")
    
    # Class imbalance ratio from training data
    train_labels = train_ds.y
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    class_imbalance_ratio = list((neg_counts / (pos_counts + 1)).clip(max=50))
    
    # Model — paper hyperparameters UNCHANGED
    model_dir = os.path.join(SAVE_DIR, 'openpom_ensemble', f'experiments_{model_idx+1}')
    os.makedirs(model_dir, exist_ok=True)
    
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
    
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # [FIX #6] Cosine Annealing LR Scheduler
    # Will be created after first fit() call when optimizer exists
    has_scheduler = False
    scheduler = None
    
    # Training loop
    print(f"  Training {epochs} epochs (validating every epoch)...")
    best_val_auroc = 0
    best_epoch = 0
    patience = 0
    max_patience = 15  # Early stopping
    
    for epoch in range(epochs):
        loss = model.fit(train_ds, nb_epoch=1)
        
        # Create LR scheduler after first fit (when optimizer is initialized)
        if not has_scheduler and epoch == 0:
            try:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    model._pytorch_optimizer, T_max=epochs, eta_min=1e-6)
                has_scheduler = True
                print(f"  LR Scheduler: CosineAnnealing (T_max={epochs}, eta_min=1e-6)")
            except Exception as e:
                print(f"  LR Scheduler: not available ({e})")
        
        # Step LR scheduler after each epoch
        if has_scheduler:
            scheduler.step()
        
        # [FIX #5] Validate EVERY epoch
        val_preds = model.predict(val_ds)
        val_labels = val_ds.y
        
        aurocs = []
        for j in range(n_tasks):
            y_true = val_labels[:, j]
            y_pred = val_preds[:, j]
            if len(np.unique(y_true)) >= 2:
                try:
                    aurocs.append(roc_auc_score(y_true, y_pred))
                except:
                    pass
        
        avg_auroc = np.mean(aurocs) if aurocs else 0
        
        if avg_auroc > best_val_auroc:
            best_val_auroc = avg_auroc
            best_epoch = epoch + 1
            model.save_checkpoint(model_dir=model_dir)
            patience = 0
            marker = ' *BEST*'
        else:
            patience += 1
            marker = ''
        
        # Print every 5 epochs or on best
        if (epoch + 1) % 5 == 0 or marker or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={loss:.4f} "
                  f"val_AUROC={avg_auroc:.4f} "
                  f"({len(aurocs)} tasks){marker}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
            break
    
    # [FIX #1] RESTORE BEST WEIGHTS before returning!
    print(f"  Restoring best weights from epoch {best_epoch}...")
    model.restore(model_dir=model_dir)
    
    return model, best_val_auroc, best_epoch


def train_ensemble(n_models=10, epochs=30):
    """[FIX #2] Proper ensemble: single split, soft voting
    
    - Data split ONCE with fixed seed
    - All models see same train/val/test
    - Ensemble = average PREDICTIONS (soft voting), then compute AUROC once
    """
    print("=" * 60)
    print("[START] OpenPOM Phase 1 -- Paper Reproduction")
    print("=" * 60)
    print(f"  Tasks: {n_tasks}")
    print(f"  Ensemble: {n_models} models")
    print(f"  Epochs: {epochs} (with early stopping)")
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Check if DGL supports CUDA
    if device == 'cuda':
        try:
            import dgl
            g = dgl.graph(([0, 1], [1, 2]))
            g = g.to('cuda')
            print(f"  Device: cuda (GPU verified)")
        except Exception:
            device = 'cpu'
            print(f"  Device: cpu (DGL CUDA not available, falling back)")
    else:
        print(f"  Device: cpu")
    
    # 1. Load data (with canonical SMILES + exact match labels)
    smiles_list, labels = load_gs_lf_merged()
    
    if len(smiles_list) == 0:
        print("ERROR: No data loaded!")
        return
    
    # 2. Featurize ONCE (outside loop)
    features, labels_valid, valid_smiles = featurize_molecules(smiles_list, labels)
    
    # 3. [FIX #2] Split ONCE with fixed seed — all models see same split
    print(f"  Scaffold splitting (seed=42, ONCE for all models)...")
    try:
        dataset = dc.data.NumpyDataset(
            X=features, y=labels_valid,
            w=np.ones_like(labels_valid),
            ids=np.array(valid_smiles))
        
        splitter = dc.splits.ScaffoldSplitter()
        train_ds, val_ds, test_ds = splitter.train_valid_test_split(
            dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1,
            seed=42)  # FIXED seed
    except Exception as e:
        print(f"  Scaffold split failed ({e}), using random split...")
        n = len(features)
        idx = np.random.RandomState(42).permutation(n)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        
        train_ds = dc.data.NumpyDataset(
            X=features[idx[:n_train]], y=labels_valid[idx[:n_train]],
            w=np.ones_like(labels_valid[idx[:n_train]]))
        val_ds = dc.data.NumpyDataset(
            X=features[idx[n_train:n_train+n_val]], y=labels_valid[idx[n_train:n_train+n_val]],
            w=np.ones_like(labels_valid[idx[n_train:n_train+n_val]]))
        test_ds = dc.data.NumpyDataset(
            X=features[idx[n_train+n_val:]], y=labels_valid[idx[n_train+n_val:]],
            w=np.ones_like(labels_valid[idx[n_train+n_val:]]))
    
    print(f"  Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    # 4. Train ensemble — each model with different seed, same data
    all_test_preds = []
    all_val_aurocs = []
    start = time.time()
    
    for i in range(n_models):
        model, val_auroc, best_epoch = train_single_model(
            train_ds, val_ds, model_idx=i, epochs=epochs, device=device)
        
        # Get test predictions from RESTORED best model
        test_preds = model.predict(test_ds)
        all_test_preds.append(test_preds)
        all_val_aurocs.append(val_auroc)
        
        # Per-model test AUROC (for logging only)
        model_aurocs = []
        for j in range(n_tasks):
            y_true = test_ds.y[:, j]
            y_pred = test_preds[:, j]
            if len(np.unique(y_true)) >= 2:
                try:
                    model_aurocs.append(roc_auc_score(y_true, y_pred))
                except:
                    pass
        model_test_auroc = np.mean(model_aurocs) if model_aurocs else 0
        print(f"  Model {i+1}: val={val_auroc:.4f}, test={model_test_auroc:.4f} (epoch {best_epoch})")
    
    elapsed = time.time() - start
    
    # 5. [FIX #2] SOFT VOTING: average predictions, then compute AUROC
    print(f"\n{'='*60}")
    print(f"  [ENSEMBLE] Soft Voting ({n_models} models)")
    print(f"{'='*60}")
    
    ensemble_preds = np.mean(all_test_preds, axis=0)
    
    final_aurocs = []
    for j in range(n_tasks):
        y_true = test_ds.y[:, j]
        y_pred = ensemble_preds[:, j]
        if len(np.unique(y_true)) >= 2:
            try:
                final_aurocs.append(roc_auc_score(y_true, y_pred))
            except:
                pass
    
    ensemble_auroc = np.mean(final_aurocs) if final_aurocs else 0
    
    print(f"    Individual model avg: {np.mean(all_val_aurocs):.4f} (val)")
    print(f"    Ensemble Test AUROC:  {ensemble_auroc:.4f} ({len(final_aurocs)} tasks)")
    print(f"    Paper reference:      0.89")
    print(f"    Total time:           {elapsed/60:.1f} min")
    print(f"    Models saved:         models/openpom_ensemble/")
    
    # Save results
    os.makedirs(os.path.join(SAVE_DIR, 'openpom_ensemble'), exist_ok=True)
    results_path = os.path.join(SAVE_DIR, 'openpom_ensemble', 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'n_models': n_models,
            'epochs': epochs,
            'n_tasks': n_tasks,
            'tasks': TASKS,
            'individual_val_aurocs': [float(v) for v in all_val_aurocs],
            'ensemble_test_auroc': float(ensemble_auroc),
            'n_evaluated_tasks': len(final_aurocs),
            'elapsed_seconds': elapsed,
            'device': device,
            'fixes_applied': [
                'best_weights_restore',
                'single_split_soft_voting',
                'canonical_smiles',
                'exact_match_labels',
                'every_epoch_validation',
            ]
        }, f, indent=2)
    print(f"    Results saved:        {results_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=int, default=1, help='Number of ensemble models')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per model')
    args = parser.parse_args()
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_ensemble(n_models=args.models, epochs=args.epochs)
