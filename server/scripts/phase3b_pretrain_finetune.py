"""
===================================================================
Phase 3b: Pre-train on Mega (28K) → Fine-tune on GS-LF (5K)
===================================================================
Step 1: Use Phase 3 pre-trained model (28K molecules, structural understanding)
Step 2: Fine-tune on GS-LF clean data with LOW learning rate
Goal: Scaffold AUROC > 0.7890 (Phase 1 baseline)
===================================================================
"""
import os, sys, csv, re, json, time, argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

# Shared config
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
        if mol is None or mol.GetNumAtoms() < 2: return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except: return None


def load_gs_lf():
    """Load GS-LF (same as Phase 1)"""
    all_smiles = {}
    
    # Leffingwell
    lf_dir = os.path.join(DATA_DIR, 'pyrfume_all', 'leffingwell')
    cid_map = {}
    with open(os.path.join(lf_dir, 'molecules.csv'), 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid, smi = row.get('CID',''), row.get('IsomericSMILES','')
            if cid and smi:
                c = canonicalize_smiles(smi)
                if c: cid_map[str(cid)] = c
    
    lf_count = 0
    with open(os.path.join(lf_dir, 'behavior.csv'), 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        lf_descs = [c for c in reader.fieldnames if c != 'Stimulus']
        for row in reader:
            smi = cid_map.get(str(row.get('Stimulus','')), '')
            if not smi: continue
            label = np.zeros(n_tasks, dtype=np.float32)
            for desc in lf_descs:
                dl = desc.lower().strip()
                if dl in task_to_idx:
                    try:
                        if float(row.get(desc,'0')) > 0: label[task_to_idx[dl]] = 1.0
                    except: pass
            if smi not in all_smiles: all_smiles[smi] = label
            else: all_smiles[smi] = np.maximum(all_smiles[smi], label)
            lf_count += 1
    print(f"  Leffingwell: {lf_count}")
    
    # GoodScents
    gs_dir = os.path.join(DATA_DIR, 'pyrfume_all', 'goodscents')
    gs_cid = {}
    with open(os.path.join(gs_dir, 'molecules.csv'), 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            cid, smi = row.get('CID',''), row.get('IsomericSMILES','')
            if cid and smi:
                c = canonicalize_smiles(smi)
                if c: gs_cid[str(cid)] = c
    cas_map = {}
    cas_path = os.path.join(gs_dir, 'cas_to_cid.json')
    if os.path.exists(cas_path):
        with open(cas_path, 'r') as f: cas_map = json.load(f)
    
    gs_count = 0
    with open(os.path.join(gs_dir, 'behavior.csv'), 'r', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            stimulus = row.get('Stimulus','').strip()
            cid = cas_map.get(stimulus, '')
            smi = gs_cid.get(str(cid), '') if cid else ''
            if not smi: continue
            desc_text = row.get('Descriptors', '')
            if not desc_text: continue
            label = np.zeros(n_tasks, dtype=np.float32)
            descs = [d.strip().lower() for d in desc_text.replace(';',',').split(',')]
            matched = 0
            for d in descs:
                for task, rx in task_regexes.items():
                    if rx.search(d):
                        label[task_to_idx[task]] = 1.0
                        matched += 1
            if matched > 0:
                if smi not in all_smiles: all_smiles[smi] = label
                else: all_smiles[smi] = np.maximum(all_smiles[smi], label)
                gs_count += 1
    print(f"  GoodScents: {gs_count}")
    
    smiles_list = list(all_smiles.keys())
    labels = np.array([all_smiles[s] for s in smiles_list], dtype=np.float32)
    return smiles_list, labels


def pretrain_on_mega(epochs=30, device='cuda'):
    """Step 1: Pre-train on Mega dataset (import from phase3)"""
    from phase3_mega_train import load_mega_dataset
    
    print("=" * 60)
    print("  [Step 1] Pre-training on Mega Dataset (28K molecules)")
    print("=" * 60)
    
    smiles_list, labels, masks = load_mega_dataset()
    
    # Featurize
    print(f"  Featurizing {len(smiles_list)} molecules...")
    featurizer = GraphFeaturizer()
    valid_idx, valid_features = [], []
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
    labels_v = labels[valid_idx]
    masks_v = masks[valid_idx]
    smiles_v = [smiles_list[i] for i in valid_idx]
    print(f"  Valid: {len(features)}")
    
    dataset = dc.data.NumpyDataset(X=features, y=labels_v, w=masks_v, ids=np.array(smiles_v))
    splitter = dc.splits.ScaffoldSplitter()
    train_ds, val_ds, _ = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)
    
    pos_counts = train_ds.y.sum(axis=0)
    neg_counts = len(train_ds.y) - pos_counts
    cir = list((neg_counts / (pos_counts + 1)).clip(max=50))
    
    model_dir = os.path.join(SAVE_DIR, 'openpom_pretrained', 'mega_pretrain')
    os.makedirs(model_dir, exist_ok=True)
    
    model = MPNNPOMModel(
        n_tasks=n_tasks, batch_size=128, class_imbalance_ratio=cir,
        loss_aggr_type='sum', node_out_feats=100, edge_hidden_feats=75,
        edge_out_feats=100, num_step_message_passing=5, mpnn_residual=True,
        message_aggregator_type='sum', mode='classification',
        number_atom_features=GraphConvConstants.ATOM_FDIM,
        number_bond_features=GraphConvConstants.BOND_FDIM,
        n_classes=1, readout_type='set2set', num_step_set2set=3,
        num_layer_set2set=2, ffn_hidden_list=[392, 392], ffn_embeddings=256,
        ffn_activation='relu', ffn_dropout_p=0.12,
        ffn_dropout_at_input_no_act=False, weight_decay=1e-5,
        self_loop=False, optimizer_name='adam',
        model_dir=model_dir, device_name=device,
    )
    
    print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"  Pre-training {epochs} epochs...")
    
    best_val = 0
    best_epoch = 0
    for epoch in range(epochs):
        loss = model.fit(train_ds, nb_epoch=1)
        val_preds = model.predict(val_ds)
        aurocs = []
        for j in range(n_tasks):
            w = val_ds.w[:, j]
            idx = w > 0
            if idx.sum() < 5: continue
            y_true = val_ds.y[idx, j]
            y_pred = val_preds[idx, j]
            if len(np.unique(y_true)) >= 2:
                try: aurocs.append(roc_auc_score(y_true, y_pred))
                except: pass
        val_auroc = np.mean(aurocs) if aurocs else 0
        if val_auroc > best_val:
            best_val = val_auroc
            best_epoch = epoch + 1
            model.save_checkpoint(model_dir=model_dir)
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={loss:.4f} val={val_auroc:.4f} *BEST*")
        elif (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={loss:.4f} val={val_auroc:.4f}")
    
    print(f"  Pre-train done: best val={best_val:.4f} (epoch {best_epoch})")
    model.restore(model_dir=model_dir)
    return model_dir, best_val


def finetune_on_gslf(pretrained_dir, n_models=10, epochs=20, device='cuda'):
    """Step 2: Fine-tune pre-trained model on GS-LF with low LR"""
    print("\n" + "=" * 60)
    print("  [Step 2] Fine-tuning on GS-LF (5K molecules, low LR)")
    print("=" * 60)
    
    smiles_list, labels = load_gs_lf()
    print(f"  GS-LF: {len(smiles_list)} molecules x {n_tasks} tasks")
    
    # Featurize
    featurizer = GraphFeaturizer()
    valid_features, valid_labels, valid_smi = [], [], []
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
    labels_v = np.array(valid_labels, dtype=np.float32)
    print(f"  Valid: {len(features)}")
    
    dataset = dc.data.NumpyDataset(
        X=features, y=labels_v, w=np.ones_like(labels_v), ids=np.array(valid_smi))
    
    # Same scaffold split as Phase 1
    splitter = dc.splits.ScaffoldSplitter()
    train_ds, val_ds, test_ds = splitter.train_valid_test_split(
        dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42)
    print(f"  Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    pos_counts = train_ds.y.sum(axis=0)
    neg_counts = len(train_ds.y) - pos_counts
    cir = list((neg_counts / (pos_counts + 1)).clip(max=50))
    
    all_test_preds = []
    all_val_aurocs = []
    
    for model_idx in range(n_models):
        print(f"\n{'='*60}")
        print(f"  [FINETUNE] Model {model_idx+1} -- (seed={model_idx})")
        print(f"{'='*60}")
        
        torch.manual_seed(model_idx)
        np.random.seed(model_idx)
        
        ft_dir = os.path.join(SAVE_DIR, 'openpom_pretrained', f'finetune_{model_idx+1}')
        os.makedirs(ft_dir, exist_ok=True)
        
        # Create model with LOW learning rate for fine-tuning
        model = MPNNPOMModel(
            n_tasks=n_tasks, batch_size=128, class_imbalance_ratio=cir,
            loss_aggr_type='sum', node_out_feats=100, edge_hidden_feats=75,
            edge_out_feats=100, num_step_message_passing=5, mpnn_residual=True,
            message_aggregator_type='sum', mode='classification',
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1, readout_type='set2set', num_step_set2set=3,
            num_layer_set2set=2, ffn_hidden_list=[392, 392], ffn_embeddings=256,
            ffn_activation='relu', ffn_dropout_p=0.12,
            ffn_dropout_at_input_no_act=False, weight_decay=1e-5,
            self_loop=False, optimizer_name='adam',
            learning_rate=0.00005,  # 🔑 5x lower than default (1e-4 → 5e-5)
            model_dir=ft_dir, device_name=device,
        )
        
        # Load pre-trained weights from Mega
        print(f"  Loading pre-trained weights from Mega...")
        model.restore(model_dir=pretrained_dir)
        print(f"  Pre-trained weights loaded!")
        
        # Fine-tune
        best_val = 0
        best_epoch = 0
        patience = 0
        max_patience = 10
        
        for epoch in range(epochs):
            loss = model.fit(train_ds, nb_epoch=1)
            val_preds = model.predict(val_ds)
            aurocs = []
            for j in range(n_tasks):
                y_true = val_ds.y[:, j]
                y_pred = val_preds[:, j]
                if len(np.unique(y_true)) >= 2:
                    try: aurocs.append(roc_auc_score(y_true, y_pred))
                    except: pass
            val_auroc = np.mean(aurocs) if aurocs else 0
            
            if val_auroc > best_val:
                best_val = val_auroc
                best_epoch = epoch + 1
                patience = 0
                model.save_checkpoint(model_dir=ft_dir)
                print(f"    Epoch {epoch+1:3d}/{epochs}: loss={loss:.4f} val={val_auroc:.4f} ({len(aurocs)} tasks) *BEST*")
            else:
                patience += 1
                if (epoch+1) % 5 == 0 or epoch == epochs-1:
                    print(f"    Epoch {epoch+1:3d}/{epochs}: loss={loss:.4f} val={val_auroc:.4f} ({len(aurocs)} tasks)")
            
            if patience >= max_patience:
                print(f"    Early stop at epoch {epoch+1}")
                break
        
        # Restore best + test
        model.restore(model_dir=ft_dir)
        test_preds = model.predict(test_ds)
        test_aurocs = []
        for j in range(n_tasks):
            y_true = test_ds.y[:, j]
            y_pred = test_preds[:, j]
            if len(np.unique(y_true)) >= 2:
                try: test_aurocs.append(roc_auc_score(y_true, y_pred))
                except: pass
        test_auroc = np.mean(test_aurocs) if test_aurocs else 0
        print(f"  Model {model_idx+1}: val={best_val:.4f}, test={test_auroc:.4f} (epoch {best_epoch})")
        
        all_test_preds.append(test_preds)
        all_val_aurocs.append(best_val)
    
    # Ensemble
    print(f"\n{'='*60}")
    print(f"  [ENSEMBLE] Soft Voting ({n_models} models)")
    print(f"{'='*60}")
    
    ensemble_preds = np.mean(all_test_preds, axis=0)
    test_labels = test_ds.y
    ensemble_aurocs = []
    for j in range(n_tasks):
        y_true = test_labels[:, j]
        y_pred = ensemble_preds[:, j]
        if len(np.unique(y_true)) >= 2:
            try: ensemble_aurocs.append(roc_auc_score(y_true, y_pred))
            except: pass
    
    ensemble_auroc = np.mean(ensemble_aurocs) if ensemble_aurocs else 0
    
    print(f"    Individual avg: {np.mean(all_val_aurocs):.4f} (val)")
    print(f"    Ensemble Test:  {ensemble_auroc:.4f} ({len(ensemble_aurocs)} tasks)")
    print(f"    Phase 1 ref:    0.7890")
    print(f"    Improvement:    {ensemble_auroc - 0.7890:+.4f}")
    
    results = {
        'approach': 'pretrain_mega_finetune_gslf',
        'ensemble_test_auroc': float(ensemble_auroc),
        'individual_val_avg': float(np.mean(all_val_aurocs)),
        'n_models': n_models,
        'phase1_reference': 0.7890,
        'improvement': float(ensemble_auroc - 0.7890),
    }
    rp = os.path.join(SAVE_DIR, 'openpom_pretrained', 'results.json')
    with open(rp, 'w') as f: json.dump(results, f, indent=2)
    print(f"    Results: {rp}")
    
    return ensemble_auroc


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=int, default=10)
    parser.add_argument('--pretrain-epochs', type=int, default=30)
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip pre-training, use existing checkpoint')
    args = parser.parse_args()
    
    device = 'cpu'
    try:
        if torch.cuda.is_available():
            import dgl
            g = dgl.graph(([0,1],[1,2])); g.to('cuda')
            device = 'cuda'
    except: pass
    print(f"  Device: {device}")
    
    pretrained_dir = os.path.join(SAVE_DIR, 'openpom_mega', 'experiments_1')
    
    if not args.skip_pretrain:
        # Step 1: Pre-train on Mega
        pretrained_dir, pretrain_val = pretrain_on_mega(
            epochs=args.pretrain_epochs, device=device)
    else:
        print(f"  Skipping pre-train, using: {pretrained_dir}")
    
    # Step 2: Fine-tune on GS-LF
    ensemble_auroc = finetune_on_gslf(
        pretrained_dir, n_models=args.models,
        epochs=args.finetune_epochs, device=device)
    
    total = time.time() - start
    print(f"\n  Total time: {total/60:.1f} min")
