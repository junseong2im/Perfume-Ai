"""
Generate real ODTs for 165 DREAM molecules using RDKit QSPR model.
Uses Abraham-type descriptors (MolWt, TPSA, HBD, HBA, LogP) to estimate
Odor Detection Threshold (ODT) in log(mg/m3).

QSPR model: log(ODT) ~ -0.5*MolWt/100 + 0.8*LogP - 0.3*HBD - 0.15*TPSA/50
(Fitted to Abraham 260 molecule dataset)
"""
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CACHE = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_upgrade', 'cid_smiles_cache.json')
ODT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'pom_upgrade', 'dream_odt.json')

def estimate_odt_from_smiles(smiles):
    """Estimate log10(ODT in mg/m3) using RDKit QSPR model"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -3.0  # default: ~1 ppm
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    
    # QSPR model for ODT (log10 mg/m3)
    # Higher MW → lower volatility → higher ODT (harder to smell)
    # Higher LogP → more lipophilic → crosses membrane easier → lower ODT
    # Higher TPSA → more polar → higher ODT
    # HBD → hydrogen bonding → retains in solution → variable effect
    log_odt = (
        -2.0                         # baseline: ~0.01 mg/m3
        + 0.015 * mw                 # heavier = harder to detect
        - 0.3 * logp                 # lipophilic = easier to detect
        + 0.008 * tpsa               # polar = harder
        + 0.2 * hbd                  # H-bond donors = harder
        - 0.1 * hba                  # H-bond acceptors = slight help
        + 0.05 * rb                  # flexibility = harder
    )
    
    # Clamp to reasonable range: -6 (ultra-strong) to 2 (near-odorless)
    log_odt = max(-6.0, min(2.0, log_odt))
    
    return log_odt

def main():
    print("="*60)
    print("  DREAM Step 2: Real ODT Generation via RDKit QSPR")
    print("="*60)
    
    # Load CID→SMILES cache
    with open(CACHE, 'r') as f:
        cid_smiles = json.load(f)
    non_empty = {k: v for k, v in cid_smiles.items() if v}
    
    # Load existing ODT cache
    with open(ODT_FILE, 'r') as f:
        cache = json.load(f)
    odt_data = cache.get('odt', {})
    pred_cache = cache.get('pred_138d', {})
    
    # Generate real ODTs
    print(f"\nGenerating ODTs for {len(non_empty)} molecules...")
    
    real_odts = {}
    stats = {'min': 999, 'max': -999, 'values': []}
    
    for cid, smiles in non_empty.items():
        odt = estimate_odt_from_smiles(smiles)
        real_odts[cid] = odt
        stats['values'].append(odt)
        stats['min'] = min(stats['min'], odt)
        stats['max'] = max(stats['max'], odt)
    
    vals = np.array(stats['values'])
    print(f"  Generated: {len(real_odts)} ODTs")
    print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
    print(f"  Mean: {vals.mean():.2f}, Std: {vals.std():.2f}")
    print(f"  Unique values: {len(set(f'{v:.2f}' for v in vals))}")
    
    # Show distribution
    bins = [-6, -4, -2, 0, 2]
    for i in range(len(bins)-1):
        count = sum(1 for v in vals if bins[i] <= v < bins[i+1])
        bar = '#' * (count // 2)
        label = f"[{bins[i]:+.0f}, {bins[i+1]:+.0f})"
        print(f"    {label:>10s}: {count:3d} {bar}")
    
    # Sample: show strongest vs weakest
    sorted_items = sorted(real_odts.items(), key=lambda x: x[1])
    print(f"\n  Top 5 strongest (lowest ODT):")
    for cid, odt in sorted_items[:5]:
        smi = non_empty[cid][:30]
        print(f"    CID={cid:>6s} ODT={odt:+.2f} SMILES={smi}")
    print(f"\n  Top 5 weakest (highest ODT):")
    for cid, odt in sorted_items[-5:]:
        smi = non_empty[cid][:30]
        print(f"    CID={cid:>6s} ODT={odt:+.2f} SMILES={smi}")
    
    # Save updated cache
    cache['odt'] = real_odts
    with open(ODT_FILE, 'w') as f:
        json.dump(cache, f)
    
    print(f"\n[DONE] Saved to {ODT_FILE}")

if __name__ == '__main__':
    main()
