"""
Phase 2: DB Expansion — 5,075 → 20,000+ molecules
Strategy:
1. Add GS-LF CSV molecules (4,983) to Discovery DB  
2. RDKit structural enumeration (methylation, hydroxylation, ring variations)
3. Rebuild discovery_db.npz with ensemble predictions
"""
import csv, sys, os, time
import numpy as np
import torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.logger().setLevel(RDLogger.ERROR)

DATA_DIR = Path(__file__).parent.parent / 'data'
WEIGHTS_DIR = Path(__file__).parent.parent / 'weights'


def load_existing_smiles():
    """Load all existing SMILES from all sources."""
    known = set()
    
    # 1. Existing discovery DB
    db_path = DATA_DIR / 'discovery_db.npz'
    if db_path.exists():
        db = np.load(db_path, allow_pickle=True)
        for s in db['smiles']:
            known.add(str(s))
    
    print(f"  Existing DB: {len(known)} molecules")
    return known


def load_gslf_molecules(known):
    """Extract molecules from curated_GS_LF_merged_4983.csv"""
    csv_path = DATA_DIR / 'curated_GS_LF_merged_4983.csv'
    if not csv_path.exists():
        print("  GS-LF CSV not found")
        return []
    
    added = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            smi = row[0].strip()
            mol = Chem.MolFromSmiles(smi)
            if mol:
                can = Chem.MolToSmiles(mol)
                mw = Descriptors.MolWt(mol)
                if can not in known and mol.GetNumAtoms() >= 3:
                    known.add(can)
                    added.append(can)
    
    print(f"  GS-LF CSV: +{len(added)} new molecules")
    return added


def enumerate_structural_variants(known, max_variants=15000):
    """Generate structural analogs via RDKit transformations.
    
    For each molecule in the DB, generate:
    - Methyl substitution at different positions
    - Hydroxyl addition  
    - Double bond saturation/desaturation
    - Ring opening/closure
    """
    from rdkit.Chem import rdmolops, RWMol
    
    # Start from molecules in the existing DB
    db_path = DATA_DIR / 'discovery_db.npz'
    if not db_path.exists():
        return []
    
    db = np.load(db_path, allow_pickle=True)
    source_smiles = [str(s) for s in db['smiles']]
    
    # Select diverse subset for enumeration (every 5th molecule)
    seeds = source_smiles[::5][:2000]
    
    new_variants = []
    
    # SMARTS-based transformations
    transformations = [
        # Add methyl group to aromatic C-H
        ('[cH:1]', '[c:1]C', 'methyl_aromatic'),
        # Add hydroxyl to aromatic C-H
        ('[cH:1]', '[c:1]O', 'hydroxyl_aromatic'),
        # Methylate alcohol to methyl ether
        ('[OH:1]', '[O:1]C', 'methylate_oh'),
        # Acetylate alcohol to ester
        ('[OH:1]', '[O:1]C(=O)C', 'acetylate_oh'),
        # Add methyl to aliphatic C-H
        ('[CH3:1]', '[CH2:1]C', 'methyl_add'),
        # Reduce ketone to alcohol
        ('[C:1](=O)[C:2]', '[C:1](O)[C:2]', 'reduce_ketone'),
    ]
    
    compiled = []
    for smarts_in, smarts_out, name in transformations:
        rxn_smarts = f'[{smarts_in}]>>[{smarts_out}]'
        try:
            rxn = AllChem.ReactionFromSmarts(f'{smarts_in}>>{smarts_out}')
            if rxn:
                compiled.append((rxn, name))
        except Exception:
            pass
    
    print(f"  Enumerating from {len(seeds)} seed molecules with {len(compiled)} transforms...")
    
    for i, smi in enumerate(seeds):
        if len(new_variants) >= max_variants:
            break
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        for rxn, name in compiled:
            try:
                products = rxn.RunReactants((mol,))
                for prod_tuple in products[:3]:  # Max 3 products per transform
                    try:
                        p = prod_tuple[0]
                        Chem.SanitizeMol(p)
                        can = Chem.MolToSmiles(p)
                        pmol = Chem.MolFromSmiles(can)
                        if pmol and can not in known:
                            mw = Descriptors.MolWt(pmol)
                            if 30 < mw < 500 and pmol.GetNumAtoms() >= 4:
                                known.add(can)
                                new_variants.append(can)
                    except Exception:
                        continue
            except Exception:
                continue
        
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(seeds)} seeds → {len(new_variants)} variants")
    
    print(f"  Structural enumeration: +{len(new_variants)} new variants")
    return new_variants


def rebuild_discovery_db(all_smiles):
    """Compute 20d odor vectors and rebuild discovery_db.npz"""
    from train_models import TrainableOdorNetV4
    from models.odor_gat_v5 import OdorGATv5, smiles_to_graph, ODOR_DIMENSIONS, N_DIM
    
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {dev}")
    
    # Load models
    cp4 = torch.load(WEIGHTS_DIR / 'odor_gnn.pt', map_location=dev, weights_only=True)
    v4 = TrainableOdorNetV4(input_dim=384).to(dev)
    v4.load_state_dict(cp4['model_state_dict']); v4.eval()
    
    cp5 = torch.load(WEIGHTS_DIR / 'odor_gnn_v5.pt', map_location=dev, weights_only=False)
    v5 = OdorGATv5(bert_dim=384).to(dev)
    v5.load_state_dict(cp5['model_state_dict']); v5.eval()
    
    cache_data = np.load(WEIGHTS_DIR / 'chemberta_cache.npz')
    bert_cache = {s: cache_data['embeddings'][i] for i, s in enumerate(cache_data['smiles'])}
    print(f"  ChemBERTa cache: {len(bert_cache)} entries")
    
    valid_smiles = []
    vectors = []
    
    t0 = time.time()
    batch_size = len(all_smiles)
    
    with torch.no_grad():
        for i, smi in enumerate(all_smiles):
            emb = bert_cache.get(smi)
            if emb is None:
                emb = np.zeros(384, dtype=np.float32)
            
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(dev)
            p4 = v4(x).squeeze(0).cpu().numpy()
            
            graph = smiles_to_graph(smi, device=dev)
            if graph is not None:
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=dev)
                p5 = v5(graph, x).squeeze(0).cpu().numpy()
                pred = 0.50 * p4 + 0.50 * p5
            else:
                pred = p4
            
            pred = np.clip(pred, 0, 1)
            valid_smiles.append(smi)
            vectors.append(pred)
            
            if (i + 1) % 5000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"    {i+1}/{batch_size} ({rate:.0f} mol/s)")
    
    vectors = np.array(vectors, dtype=np.float32)
    elapsed = time.time() - t0
    
    db_path = DATA_DIR / 'discovery_db.npz'
    np.savez_compressed(
        db_path,
        smiles=np.array(valid_smiles, dtype=object),
        vectors=vectors,
        dimensions=np.array(ODOR_DIMENSIONS),
    )
    
    print(f"\n  DB rebuilt: {len(valid_smiles)} molecules in {elapsed:.1f}s")
    print(f"  File: {os.path.getsize(db_path)/1024:.0f} KB")
    
    # Dimension distribution
    dom = Counter()
    for v in vectors:
        dom[ODOR_DIMENSIONS[np.argmax(v)]] += 1
    
    print(f"\n  Dominant dimension distribution:")
    for d, c in dom.most_common():
        pct = c / len(valid_smiles) * 100
        bar = "█" * int(pct / 2)
        print(f"    {d:12s}: {c:>6d} ({pct:>5.1f}%) {bar}")
    
    return len(valid_smiles)


if __name__ == '__main__':
    print("=" * 60)
    print("  PHASE 2: DB EXPANSION (5K → 20K+)")
    print("=" * 60)
    
    known = load_existing_smiles()
    
    # Step 1: GS-LF CSV
    print("\n  Step 1: GS-LF CSV molecules")
    gslf_new = load_gslf_molecules(known)
    
    # Step 2: Structural enumeration
    print("\n  Step 2: RDKit structural enumeration")
    variants = enumerate_structural_variants(known)
    
    # Combine all new + existing
    db = np.load(DATA_DIR / 'discovery_db.npz', allow_pickle=True)
    existing = [str(s) for s in db['smiles']]
    
    all_smiles = list(set(existing + gslf_new + variants))
    print(f"\n  Total unique molecules: {len(all_smiles)}")
    print(f"    Existing: {len(existing)}")
    print(f"    GS-LF new: {len(gslf_new)}")
    print(f"    Variants: {len(variants)}")
    
    # Step 3: Rebuild
    print(f"\n  Step 3: Rebuilding Discovery DB...")
    total = rebuild_discovery_db(all_smiles)
    
    print(f"\n  {'='*60}")
    print(f"  EXPANSION COMPLETE: {total} molecules")
    print(f"  {'='*60}")
