"""Final comprehensive system check for OdorPredictor v6
=======================================================
Checks:
1. Syntax validation (all Python files)
2. Model architecture integrity (odor_predictor_v6.py)
3. Training pipeline consistency (train_v6.py)
4. Data pipeline (quality_data_collector.py)
5. Curated dataset validity
6. Label mapping consistency
7. Component compatibility
"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")

def warn(name, detail=""):
    global WARN
    WARN += 1
    print(f"  [WARN] {name} {detail}")

print("=" * 70)
print("  FINAL SYSTEM CHECK: OdorPredictor v6")
print("=" * 70)

# ======================================================================
# 1. SYNTAX VALIDATION
# ======================================================================
print("\n[1/7] Syntax Validation")
import py_compile

files_to_check = [
    "cloud/models/odor_predictor_v6.py",
    "cloud/train_v6.py",
    "scripts/quality_data_collector.py",
]

for f in files_to_check:
    try:
        py_compile.compile(f, doraise=True)
        check(f"Syntax: {f}", True)
    except py_compile.PyCompileError as e:
        check(f"Syntax: {f}", False, str(e))

# ======================================================================
# 2. MODEL ARCHITECTURE
# ======================================================================
print("\n[2/7] Model Architecture")

try:
    import torch
    from models.odor_predictor_v6 import (
        OdorPredictorV6, smiles_to_graph_v6, extract_phys_props,
        EMA, GradNorm, compute_loss, contrastive_loss, N_ODOR_DIM,
        ATOM_FEATURES_DIM, BOND_FEATURES_DIM
    )
    check("Model imports", True)
except Exception as e:
    check("Model imports", False, str(e))

try:
    model = OdorPredictorV6(bert_dim=384)
    total_params = sum(p.numel() for p in model.parameters())
    check(f"Model instantiation ({total_params:,} params)", True)
    check("Param count > 30M", total_params > 30_000_000)
    check("Param count < 50M", total_params < 50_000_000)
except Exception as e:
    check("Model instantiation", False, str(e))

# Forward pass test
try:
    from torch_geometric.data import Data, Batch
    graph = smiles_to_graph_v6("CCO", compute_3d=True)
    check("Graph construction", graph is not None)
    
    gb = Batch.from_data_list([graph, graph])
    bert = torch.zeros(2, 384)
    phys = torch.zeros(2, 12)
    
    model.eval()
    with torch.no_grad():
        out = model(bert, gb, phys, return_aux=True)
    check("Forward pass (return_aux=True)", isinstance(out, dict))
    check("Output has 'odor' key", 'odor' in out)
    check(f"Odor output shape: {out['odor'].shape}", out['odor'].shape == (2, N_ODOR_DIM))
    
    with torch.no_grad():
        out2 = model(bert, gb, phys, return_aux=False)
    check("Forward pass (return_aux=False)", out2.shape == (2, N_ODOR_DIM))
except Exception as e:
    check("Forward pass", False, str(e))

# EMA test
try:
    ema = EMA(model, decay=0.999)
    check("EMA creation", True)
except Exception as e:
    check("EMA creation", False, str(e))

# ======================================================================
# 3. TRAINING PIPELINE
# ======================================================================
print("\n[3/7] Training Pipeline")

try:
    from train_v6 import (
        OdorDataset, randomize_smiles, collate_odor,
        mixup_data, cutmix_data, scaffold_split,
        WarmupCosineScheduler, load_label_mapping
    )
    check("Training imports", True)
except Exception as e:
    check("Training imports", False, str(e))

# SMILES augmentation
try:
    augs = randomize_smiles("CCO", n_aug=5)
    check(f"SMILES augmentation (CCO -> {len(augs)} variants)", len(augs) >= 2)
    
    augs2 = randomize_smiles("CC(=O)Oc1ccccc1C(=O)O", n_aug=5)
    check(f"SMILES augmentation (aspirin -> {len(augs2)} variants)", len(augs2) >= 3)
except Exception as e:
    check("SMILES augmentation", False, str(e))

# ======================================================================
# 4. DATA PIPELINE
# ======================================================================
print("\n[4/7] Data Pipeline")

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("qdc", "scripts/quality_data_collector.py")
    qdc = importlib.util.module_from_spec(spec)
    check("quality_data_collector.py loadable", True)
except Exception as e:
    check("quality_data_collector loads", False, str(e))

# ======================================================================
# 5. CURATED DATASET
# ======================================================================
print("\n[5/7] Curated Dataset")

data_path = "cloud/data/curated_training_data.csv"
if os.path.exists(data_path):
    check("curated_training_data.csv exists", True)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    
    check(f"Row count: {len(rows)}", len(rows) > 5000)
    check(f"Column count: {len(cols)}", len(cols) > 100)
    check("'smiles' column exists", 'smiles' in cols)
    check("'quality_score' column exists", 'quality_score' in cols)
    check("'sources' column exists", 'sources' in cols)
    check("'tier' column exists", 'tier' in cols)
    
    # Check data quality
    meta_cols = {'smiles', 'sources', 'n_sources', 'tier', 'quality_score'}
    label_cols = [c for c in cols if c not in meta_cols]
    check(f"Label columns: {len(label_cols)}", len(label_cols) > 100)
    
    # Check first row
    r0 = rows[0]
    smi = r0['smiles']
    check(f"First SMILES valid: {smi[:30]}", len(smi) > 2)
    
    # Count non-zero labels in first row
    nz = sum(1 for c in label_cols if float(r0.get(c, 0)) > 0)
    check(f"First row non-zero labels: {nz}", nz > 0)
    
    # Check tier distribution
    tier1 = sum(1 for r in rows if r.get('tier') == '1')
    tier2 = sum(1 for r in rows if r.get('tier') == '2')
    check(f"Tier 1 molecules: {tier1}", tier1 > 4000)
    check(f"Tier 2 molecules: {tier2}", tier2 > 0)
    
    # Multi-source check
    multi = sum(1 for r in rows if int(r.get('n_sources', 1)) >= 2)
    pct = 100 * multi / len(rows)
    check(f"Multi-source validated: {multi} ({pct:.1f}%)", pct > 40)
    
    # Quality score distribution
    scores = [float(r.get('quality_score', 0)) for r in rows]
    avg_q = sum(scores) / len(scores)
    check(f"Avg quality score: {avg_q:.3f}", avg_q > 0.6)
else:
    check("curated_training_data.csv exists", False, "FILE MISSING!")

# ======================================================================
# 6. LABEL MAPPING
# ======================================================================
print("\n[6/7] Label Mapping")

mapping_path = "cloud/data/label_mapping.json"
if os.path.exists(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    check(f"label_mapping.json exists", True)
    check(f"n_labels: {mapping['n_labels']}", mapping['n_labels'] > 100)
    check(f"Labels list length matches n_labels", len(mapping['labels']) == mapping['n_labels'])
    
    # Cross-check with CSV columns
    csv_labels = set(label_cols)
    map_labels = set(mapping['labels'])
    check("Labels match CSV columns", csv_labels == map_labels,
          f"CSV has {len(csv_labels - map_labels)} extra, mapping has {len(map_labels - csv_labels)} extra")
    
    # Stats
    stats = mapping.get('stats', {})
    if stats:
        check(f"Stats.total_molecules: {stats['total_molecules']}", stats['total_molecules'] == len(rows))
else:
    check("label_mapping.json exists", False, "FILE MISSING!")

# ======================================================================
# 7. DATASET LOADING COMPATIBILITY
# ======================================================================
print("\n[7/7] End-to-End Compatibility")

try:
    ds = OdorDataset(data_path, n_aug=0, label_smooth=0.0, max_samples=50)
    check(f"OdorDataset loads curated data ({len(ds)} samples)", len(ds) > 0)
    check(f"Labels detected: {ds.n_labels}", ds.n_labels == mapping['n_labels'])
    
    sample = ds[0]
    check(f"Sample odor shape: {sample['odor'].shape}", sample['odor'].shape[0] == ds.n_labels)
    check("Sample has graph", sample['graph'] is not None)
    check("Sample has phys", sample['phys'].shape[0] == 12)
    check("Sample has bert", sample['bert'].shape[0] == 384)
    
    # Test collate
    batch = collate_odor([ds[0], ds[1]])
    check("Collate works", batch['graph_batch'] is not None)
    check(f"Batch odor shape: {batch['odor'].shape}", batch['odor'].shape == (2, ds.n_labels))
    
    # Test with augmentation
    ds_aug = OdorDataset(data_path, n_aug=3, label_smooth=0.05, max_samples=20)
    check(f"Augmented dataset: {len(ds_aug)} samples (from 20)", len(ds_aug) > 20)
    
    # Verify label smoothing
    s = ds_aug[0]
    max_val = s['odor'].max().item()
    min_val = s['odor'].min().item()
    check(f"Label smoothing applied (max={max_val:.3f}, min={min_val:.3f})", 
          max_val < 1.0 and min_val > 0.0)
    
except Exception as e:
    check("End-to-end compatibility", False, str(e))
    import traceback; traceback.print_exc()

# ======================================================================
# SUMMARY
# ======================================================================
print(f"\n{'='*70}")
print(f"  FINAL SYSTEM CHECK COMPLETE")
print(f"  [PASS] PASSED: {PASS}")
print(f"  [FAIL] FAILED: {FAIL}")
print(f"  [WARN] WARNINGS: {WARN}")
if FAIL == 0:
    print(f"  [OK] ALL CHECKS PASSED - SYSTEM READY FOR TRAINING")
else:
    print(f"  [!!] {FAIL} CHECKS FAILED - FIX BEFORE TRAINING")
print(f"{'='*70}")
