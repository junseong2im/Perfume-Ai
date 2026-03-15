"""
===================================================================
Phase 3: Masked Multi-Task Learning (Mega Dataset)
===================================================================
7 datasets unified: Leffingwell + GoodScents + FlavorDB + FlavorNet + AromaDB + IFRA + Abraham
Masked BCE: each molecule only trained on tasks it has labels for
10-model ensemble + soft voting, Scaffold Split
===================================================================
"""
import os, sys, csv, re, json, time, argparse, traceback
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

TASKS = [
    'alcoholic','aldehydic','alliaceous','almond','amber','animal','anisic',
    'apple','apricot','aromatic','balsamic','banana','beefy','bergamot',
    'berry','bitter','black currant','brandy','burnt','buttery','cabbage',
    'camphoreous','caramellic','cedar','celery','chamomile','cheesy','cherry',
    'chocolate','cinnamon','citrus','clean','clove','cocoa','coconut','coffee',
    'cognac','cooked','cooling','cortex','coumarinic','creamy','cucumber',
    'dairy','dry','earthy','ethereal','fatty','fermented','fishy','floral',
    'fresh','fruit skin','fruity','garlic','gassy','geranium','grape',
    'grapefruit','grassy','green','hawthorn','hay','hazelnut','herbal',
    'honey','hyacinth','jasmin','juicy','ketonic','lactonic','lavender',
    'leafy','leathery','lemon','lily','malty','meaty','medicinal','melon',
    'metallic','milky','mint','muguet','mushroom','musk','musty','natural',
    'nutty','odorless','oily','onion','orange','orangeflower','orris','ozone',
    'peach','pear','phenolic','pine','pineapple','plum','popcorn','potato',
    'powdery','pungent','radish','raspberry','ripe','roasted','rose','rummy',
    'sandalwood','savory','sharp','smoky','soapy','solvent','sour','spicy',
    'strawberry','sulfurous','sweaty','sweet','tea','terpenic','tobacco',
    'tomato','tropical','vanilla','vegetable','vetiver','violet','warm',
    'waxy','weedy','winey','woody'
]

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_data')
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
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


def regex_match_descriptors(text):
    """Match text against TASKS using word boundary regex. Returns (label, mask) arrays."""
    label = np.zeros(n_tasks, dtype=np.float32)
    mask = np.zeros(n_tasks, dtype=np.float32)
    if not text:
        return label, mask
    
    descs = [d.strip().lower() for d in text.replace(';', ',').split(',')]
    for d in descs:
        for task, rx in task_regexes.items():
            if rx.search(d):
                label[task_to_idx[task]] = 1.0
                mask[task_to_idx[task]] = 1.0
    return label, mask


def load_cid_smiles(mol_path):
    """Load CID -> canonical SMILES from molecules.csv"""
    cid_map = {}
    with open(mol_path, 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid = row.get('CID', '').strip()
            smi = row.get('IsomericSMILES', '').strip()
            if cid and smi:
                c = canonicalize_smiles(smi)
                if c:
                    cid_map[str(cid)] = c
    return cid_map


def load_mega_dataset():
    """Load and merge all datasets with mask matrix"""
    # {smiles: {'label': np.array, 'mask': np.array}}
    all_data = {}
    stats = {}
    
    def add_molecule(smi, label, mask, source):
        if smi not in all_data:
            all_data[smi] = {'label': label.copy(), 'mask': mask.copy()}
        else:
            all_data[smi]['label'] = np.maximum(all_data[smi]['label'], label)
            all_data[smi]['mask'] = np.maximum(all_data[smi]['mask'], mask)
    
    # ========= 1. Leffingwell (binary columns) =========
    lf_dir = os.path.join(DATA_DIR, 'pyrfume_all', 'leffingwell')
    cid_map = load_cid_smiles(os.path.join(lf_dir, 'molecules.csv'))
    count = 0
    with open(os.path.join(lf_dir, 'behavior.csv'), 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        lf_descs = [c for c in reader.fieldnames if c != 'Stimulus']
        for row in reader:
            smi = cid_map.get(str(row.get('Stimulus', '')), '')
            if not smi: continue
            label = np.zeros(n_tasks, dtype=np.float32)
            mask = np.zeros(n_tasks, dtype=np.float32)
            for desc in lf_descs:
                dl = desc.lower().strip()
                if dl in task_to_idx:
                    mask[task_to_idx[dl]] = 1.0
                    try:
                        if float(row.get(desc, '0')) > 0:
                            label[task_to_idx[dl]] = 1.0
                    except: pass
            if mask.sum() > 0:
                add_molecule(smi, label, mask, 'leffingwell')
                count += 1
    stats['leffingwell'] = count
    print(f"  Leffingwell: {count} entries")
    
    # ========= 2. GoodScents (CAS -> CID -> SMILES) =========
    gs_dir = os.path.join(DATA_DIR, 'pyrfume_all', 'goodscents')
    gs_cid_map = load_cid_smiles(os.path.join(gs_dir, 'molecules.csv'))
    cas_map_path = os.path.join(gs_dir, 'cas_to_cid.json')
    cas_to_cid = {}
    if os.path.exists(cas_map_path):
        with open(cas_map_path, 'r') as f:
            cas_to_cid = json.load(f)
    count = 0
    with open(os.path.join(gs_dir, 'behavior.csv'), 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            stimulus = row.get('Stimulus', '').strip()
            cid = cas_to_cid.get(stimulus, '')
            smi = gs_cid_map.get(str(cid), '') if cid else ''
            if not smi: continue
            desc_text = row.get('Descriptors', '')
            if not desc_text: continue
            label, mask = regex_match_descriptors(desc_text)
            if mask.sum() > 0:
                add_molecule(smi, label, mask, 'goodscents')
                count += 1
    stats['goodscents'] = count
    print(f"  GoodScents: {count} entries")
    
    # ========= 3. FlavorDB (Odor + Flavor Percepts, CID-based) =========
    fdb_dir = os.path.join(DATA_DIR, 'pyrfume_all', 'flavordb')
    fdb_cid_map = load_cid_smiles(os.path.join(fdb_dir, 'molecules.csv'))
    count = 0
    with open(os.path.join(fdb_dir, 'behavior.csv'), 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            smi = fdb_cid_map.get(str(row.get('Stimulus', '')), '')
            if not smi: continue
            # Combine odor + flavor percepts
            text = ';'.join(filter(None, [
                row.get('Odor Percepts', ''),
                row.get('Flavor Percepts', '')
            ]))
            if not text: continue
            label, mask = regex_match_descriptors(text)
            if mask.sum() > 0:
                add_molecule(smi, label, mask, 'flavordb')
                count += 1
    stats['flavordb'] = count
    print(f"  FlavorDB: {count} entries")
    
    # ========= 4. FlavorNet (Descriptors, CID-based) =========
    fnet_dir = os.path.join(DATA_DIR, 'pyrfume_all', 'flavornet')
    fnet_cid_map = load_cid_smiles(os.path.join(fnet_dir, 'molecules.csv'))
    count = 0
    with open(os.path.join(fnet_dir, 'behavior.csv'), 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            smi = fnet_cid_map.get(str(row.get('Stimulus', '')), '')
            if not smi: continue
            text = row.get('Descriptors', '')
            if not text: continue
            label, mask = regex_match_descriptors(text)
            if mask.sum() > 0:
                add_molecule(smi, label, mask, 'flavornet')
                count += 1
    stats['flavornet'] = count
    print(f"  FlavorNet: {count} entries")
    
    # ========= 5. AromaDB (Filtered Descriptors, CID-based) =========
    adb_dir = os.path.join(DATA_DIR, 'pyrfume_all', 'aromadb')
    adb_cid_map = load_cid_smiles(os.path.join(adb_dir, 'molecules.csv'))
    count = 0
    with open(os.path.join(adb_dir, 'behavior.csv'), 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            smi = adb_cid_map.get(str(row.get('Stimulus', '')), '')
            if not smi: continue
            text = row.get('Filtered Descriptors', '')
            if not text: continue
            label, mask = regex_match_descriptors(text)
            if mask.sum() > 0:
                add_molecule(smi, label, mask, 'aromadb')
                count += 1
    stats['aromadb'] = count
    print(f"  AromaDB: {count} entries")
    
    # Summary
    smiles_list = list(all_data.keys())
    labels = np.array([all_data[s]['label'] for s in smiles_list], dtype=np.float32)
    masks = np.array([all_data[s]['mask'] for s in smiles_list], dtype=np.float32)
    
    active_tasks = (labels.sum(axis=0) >= 5).sum()
    label_density = labels.sum() / (labels.shape[0] * labels.shape[1])
    mask_density = masks.sum() / (masks.shape[0] * masks.shape[1])
    
    print(f"\n  [MEGA DATA] Summary:")
    print(f"    Total unique molecules: {len(smiles_list)}")
    print(f"    Total tasks:            {n_tasks}")
    print(f"    Active tasks (>=5 pos): {active_tasks}")
    print(f"    Label density:          {label_density:.4f}")
    print(f"    Mask density:           {mask_density:.4f}")
    print(f"    Sources: {stats}")
    
    return smiles_list, labels, masks


def train_single_model(train_ds, val_ds, test_ds, n_tasks, epochs, model_idx, device):
    """Train a single MPNN-POM model"""
    torch.manual_seed(model_idx)
    np.random.seed(model_idx)
    
    train_labels = train_ds.y
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    class_imbalance_ratio = list((neg_counts / (pos_counts + 1)).clip(max=50))
    
    model_dir = os.path.join(SAVE_DIR, 'openpom_mega', f'experiments_{model_idx+1}')
    os.makedirs(model_dir, exist_ok=True)
    
    model = MPNNPOMModel(
        n_tasks=n_tasks,
        batch_size=128,
        class_imbalance_ratio=class_imbalance_ratio,
        loss_aggr_type='sum',
        node_out_feats=100, edge_hidden_feats=75, edge_out_feats=100,
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
        model_dir=model_dir, device_name=device,
    )
    
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Parameters: {n_params:,}")
    
    # LR scheduler (deferred)
    has_scheduler = False
    scheduler = None
    
    print(f"  Training {epochs} epochs...")
    best_val = 0
    best_epoch = 0
    patience = 0
    max_patience = 15
    
    for epoch in range(epochs):
        loss = model.fit(train_ds, nb_epoch=1)
        
        if not has_scheduler and epoch == 0:
            try:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    model._pytorch_optimizer, T_max=epochs, eta_min=1e-6)
                has_scheduler = True
                print(f"  LR Scheduler: CosineAnnealing (T_max={epochs})")
            except Exception as e:
                print(f"  LR Scheduler: N/A ({e})")
        
        if has_scheduler:
            scheduler.step()
        
        # Validate
        val_preds = model.predict(val_ds)
        val_labels = val_ds.y
        val_weights = val_ds.w
        
        aurocs = []
        for j in range(n_tasks):
            # Only evaluate tasks where we have masked labels
            w = val_weights[:, j]
            idx = w > 0
            if idx.sum() < 5: continue
            y_true = val_labels[idx, j]
            y_pred = val_preds[idx, j]
            if len(np.unique(y_true)) >= 2:
                try:
                    aurocs.append(roc_auc_score(y_true, y_pred))
                except: pass
        
        val_auroc = np.mean(aurocs) if aurocs else 0
        
        if val_auroc > best_val:
            best_val = val_auroc
            best_epoch = epoch + 1
            patience = 0
            model.save_checkpoint(model_dir=model_dir)
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={loss:.4f} val_AUROC={val_auroc:.4f} ({len(aurocs)} tasks) *BEST*")
        else:
            patience += 1
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1:3d}/{epochs}: loss={loss:.4f} val_AUROC={val_auroc:.4f} ({len(aurocs)} tasks)")
        
        if patience >= max_patience:
            print(f"    Early stopping at epoch {epoch+1} (patience={max_patience})")
            break
    
    # Restore best weights
    print(f"  Restoring best weights from epoch {best_epoch}...")
    model.restore(model_dir=model_dir)
    
    # Test evaluation
    test_preds = model.predict(test_ds)
    test_labels = test_ds.y
    test_weights = test_ds.w
    test_aurocs = []
    for j in range(n_tasks):
        w = test_weights[:, j]
        idx = w > 0
        if idx.sum() < 5: continue
        y_true = test_labels[idx, j]
        y_pred = test_preds[idx, j]
        if len(np.unique(y_true)) >= 2:
            try:
                test_aurocs.append(roc_auc_score(y_true, y_pred))
            except: pass
    
    test_auroc = np.mean(test_aurocs) if test_aurocs else 0
    print(f"  Model {model_idx+1}: val={best_val:.4f}, test={test_auroc:.4f} (epoch {best_epoch})")
    
    return model, best_val, best_epoch, test_preds


def train_ensemble(n_models=10, epochs=30):
    """Train N-model ensemble with masked multi-task learning"""
    print("=" * 60)
    print("[START] Phase 3 -- Masked Multi-Task Mega Training")
    print("=" * 60)
    print(f"  Tasks: {n_tasks}")
    print(f"  Ensemble: {n_models} models")
    print(f"  Epochs: {epochs} (with early stopping)")
    
    device = 'cpu'
    try:
        if torch.cuda.is_available():
            import dgl
            g = dgl.graph(([0,1],[1,2]))
            g = g.to('cuda')
            device = 'cuda'
    except: pass
    print(f"  Device: {device}")
    
    # Load merged data
    smiles_list, labels, masks = load_mega_dataset()
    
    # Featurize
    print(f"  Featurizing {len(smiles_list)} molecules...")
    featurizer = GraphFeaturizer()
    valid_idx = []
    valid_features = []
    for i, smi in enumerate(smiles_list):
        try:
            feat = featurizer.featurize([smi])
            if feat is not None and len(feat) > 0:
                f = feat[0]
                if f is not None and hasattr(f, 'node_features') and f.node_features.shape[0] > 0:
                    valid_features.append(f)
                    valid_idx.append(i)
        except: pass
    
    features = np.array(valid_features, dtype=object)
    labels_valid = labels[valid_idx]
    masks_valid = masks[valid_idx]
    smiles_valid = [smiles_list[i] for i in valid_idx]
    print(f"  Valid: {len(features)}/{len(smiles_list)} (skipped {len(smiles_list)-len(features)})")
    
    # Create dataset WITH mask as weights
    dataset = dc.data.NumpyDataset(
        X=features, y=labels_valid,
        w=masks_valid,  # Mask matrix as weights -> only labeled tasks contribute to loss
        ids=np.array(smiles_valid))
    
    # Scaffold split (ONCE)
    print("  Scaffold splitting (seed=42)...")
    splitter = dc.splits.ScaffoldSplitter()
    train_ds, val_ds, test_ds = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)
    print(f"  Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    # Train models
    all_test_preds = []
    all_val_aurocs = []
    
    for i in range(n_models):
        print(f"\n{'='*60}")
        print(f"  [MODEL] Model {i+1} -- Training (seed={i})")
        print(f"{'='*60}")
        
        model, val_auroc, best_epoch, test_preds = train_single_model(
            train_ds, val_ds, test_ds, n_tasks, epochs, i, device)
        
        all_test_preds.append(test_preds)
        all_val_aurocs.append(val_auroc)
    
    # Ensemble soft voting
    print(f"\n{'='*60}")
    print(f"  [ENSEMBLE] Soft Voting ({n_models} models)")
    print(f"{'='*60}")
    
    ensemble_preds = np.mean(all_test_preds, axis=0)
    test_labels = test_ds.y
    test_weights = test_ds.w
    
    ensemble_aurocs = []
    for j in range(n_tasks):
        w = test_weights[:, j]
        idx = w > 0
        if idx.sum() < 5: continue
        y_true = test_labels[idx, j]
        y_pred = ensemble_preds[idx, j]
        if len(np.unique(y_true)) >= 2:
            try:
                ensemble_aurocs.append(roc_auc_score(y_true, y_pred))
            except: pass
    
    ensemble_auroc = np.mean(ensemble_aurocs) if ensemble_aurocs else 0
    
    print(f"    Individual model avg: {np.mean(all_val_aurocs):.4f} (val)")
    print(f"    Ensemble Test AUROC:  {ensemble_auroc:.4f} ({len(ensemble_aurocs)} tasks)")
    print(f"    Phase 1 reference:    0.7890 (GS-LF only)")
    
    # Save results
    results = {
        'phase': 3,
        'ensemble_test_auroc': float(ensemble_auroc),
        'individual_val_avg': float(np.mean(all_val_aurocs)),
        'n_models': n_models,
        'n_tasks_evaluated': len(ensemble_aurocs),
        'total_molecules': len(features),
        'phase1_reference': 0.7890,
    }
    results_path = os.path.join(SAVE_DIR, 'openpom_mega', 'results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"    Total time:           {total_time/60:.1f} min")
    print(f"    Models saved:         models/openpom_mega/")
    print(f"    Results saved:        {results_path}")


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    
    train_ensemble(n_models=args.models, epochs=args.epochs)
