"""
Fix 5: HedonicFunction data-driven weight fitting
Uses 300 famous perfume ratings to learn SMARTS pattern weights via ridge regression.
Saves fitted weights to data/hedonic_weights.json for runtime use.
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rdkit import Chem
from rdkit.Chem import Descriptors


# ========== Feature extraction ==========
SMARTS_FEATURES = [
    # Pleasant
    ('[CX3](=O)O[CX4]', 'ester'),
    ('c1ccccc1', 'benzene_ring'),
    ('[CX3H1](=O)', 'aldehyde'),
    ('[OX2H]', 'alcohol'),
    ('C=C(C)C', 'isoprene'),
    ('[CX3](=O)[CX4]', 'ketone'),
    ('c1cc(O)ccc1', 'phenol_mild'),
    ('OC(=O)c1ccccc1', 'benzoate'),
    ('C/C=C/C(=O)', 'enone'),
    # Unpleasant
    ('[#16]', 'sulfur'),
    ('[SX2H]', 'thiol'),
    ('[NX3H2]', 'primary_amine'),
    ('S=O', 'sulfoxide'),
    ('[NX2]=[NX2]', 'diazo'),
    ('[N+](=O)[O-]', 'nitro'),
    ('C#N', 'nitrile'),
    ('[Cl,Br,I]', 'halogen'),
]

COMPILED_SMARTS = [(Chem.MolFromSmarts(s), name) for s, name in SMARTS_FEATURES]


def extract_features(smiles):
    """Extract SMARTS match + molecular descriptor features"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    # SMARTS matches (binary)
    for pattern, name in COMPILED_SMARTS:
        if pattern and mol.HasSubstructMatch(pattern):
            features.append(1.0)
        else:
            features.append(0.0)
    
    # Continuous descriptors (normalized)
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    features.append(max(0, min(1, (logp - 0) / 6)))       # logP 0~6 → 0~1
    features.append(max(0, min(1, (mw - 50) / 400)))       # MW 50~450 → 0~1
    features.append(max(0, min(1, Descriptors.TPSA(mol) / 100)))  # TPSA
    features.append(Descriptors.NumAromaticRings(mol) / 4)  # aromatic rings
    features.append(Descriptors.NumHDonors(mol) / 4)        # HBD
    features.append(Descriptors.NumHAcceptors(mol) / 6)     # HBA
    features.append(Descriptors.NumRotatableBonds(mol) / 10) # rotatable
    
    return features


# Category SMILES (simplified)
CAT_SMILES = {
    'citrus': 'CC(=CCC/C(=C\\C)C)C',
    'floral': 'OCC=C(C)CCC=C(C)C',
    'woody': 'CC1CCC2(C)C(O)CCC12',
    'oriental': 'O=Cc1ccc(O)c(OC)c1',
    'musk': 'O=C1CCCCCCCCCCCCC1',
    'fresh': 'CC(=O)OCC=C(C)C',
    'spicy': 'C=CCc1ccc(O)c(OC)c1',
    'green': 'CC/C=C\\CCO',
    'fruity': 'CCCCOC(=O)CC',
    'amber': 'CC1(C)CCCC2(C)C1CCC1=CC(=O)CCC12',
    'vanilla': 'O=Cc1ccc(O)c(OC)c1',
    'leather': 'c1ccc2c(c1)c(=O)[nH]c2=O',
    'gourmand': 'CC1=CC(=O)OC1',
    'aquatic': 'O=CC1=CC(C)CC1',
}


def note_to_smiles(note_id):
    for key, smi in CAT_SMILES.items():
        if key in note_id.lower():
            return smi
    return 'CCCCCCO'


def perfume_features(perfume):
    """Average features over all notes in a perfume"""
    all_notes = (perfume.get('top_notes', []) + 
                 perfume.get('middle_notes', []) + 
                 perfume.get('base_notes', []))
    
    n_feat = len(SMARTS_FEATURES) + 7  # SMARTS + 7 descriptors
    feat_sum = np.zeros(n_feat)
    count = 0
    
    for note in all_notes:
        smi = note_to_smiles(note)
        f = extract_features(smi)
        if f:
            feat_sum += np.array(f)
            count += 1
    
    if count > 0:
        return feat_sum / count
    return None


def fit_weights():
    """Ridge regression: features → rating"""
    # Load perfumes
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'famous_perfumes.json')
    with open(path, 'r', encoding='utf-8') as f:
        perfumes = json.load(f)
    
    X = []
    y = []
    
    for p in perfumes:
        feat = perfume_features(p)
        if feat is not None:
            X.append(feat)
            y.append((p['rating'] - 1) / 4)  # 1~5 → 0~1
    
    X = np.array(X)
    y = np.array(y)
    print(f"Training data: {X.shape[0]} perfumes, {X.shape[1]} features")
    
    # Ridge regression (manual, no sklearn dependency)
    alpha = 1.0  # regularization
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    weights = np.linalg.solve(XtX, Xty)
    
    # Evaluate
    y_pred = X @ weights
    residuals = y - y_pred
    mse = np.mean(residuals ** 2)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    corr = np.corrcoef(y, y_pred)[0, 1]
    
    print(f"\nRidge Regression Results:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Corr: {corr:.4f}")
    
    # Feature importance
    feature_names = [name for _, name in SMARTS_FEATURES] + ['logP', 'MW', 'TPSA', 'aromatic_rings', 'HBD', 'HBA', 'rotatable']
    print(f"\nFeature weights (sorted by abs):")
    ranked = sorted(zip(feature_names, weights), key=lambda x: -abs(x[1]))
    for name, w in ranked:
        direction = "+" if w > 0 else "-"
        print(f"  {direction} {name:20s}: {w:+.4f}")
    
    # Save
    out = {
        'feature_names': feature_names,
        'weights': weights.tolist(),
        'bias': float(np.mean(y)),
        'alpha': alpha,
        'r2': round(r2, 4),
        'correlation': round(corr, 4),
        'mse': round(mse, 4),
        'n_samples': len(y),
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'hedonic_weights.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")
    
    return out


if __name__ == '__main__':
    fit_weights()
