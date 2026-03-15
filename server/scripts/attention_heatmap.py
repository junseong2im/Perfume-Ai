"""
Attention Heatmap: GATConv atom attention → RDKit 2D molecular visualization
Uses rdMolDraw2D with SimilarityMaps for Gaussian-smoothed heatmaps.
"""
import io, os, base64, sys
import numpy as np
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.odor_gat_v5 import OdorGATv5, smiles_to_graph, ODOR_DIMENSIONS


def generate_attention_heatmap(
    smiles: str,
    v5_model: OdorGATv5,
    device='cuda',
    width=500,
    height=400,
) -> dict:
    """Generate attention heatmap for a molecule.
    
    Returns dict with 'image_base64', 'atom_importance', 'top_atoms'.
    """
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'error': 'Invalid SMILES'}
    
    AllChem.Compute2DCoords(mol)
    
    # Build graph and extract attention
    graph = smiles_to_graph(smiles, device=dev)
    if graph is None:
        return {'error': 'Failed to build graph'}
    
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=dev)
    v5_model.eval()
    v5_model.to(dev)
    
    atom_importance, _ = v5_model.extract_attention(graph)
    importance = atom_importance.numpy().astype(float)
    
    n_atoms = mol.GetNumAtoms()
    if len(importance) < n_atoms:
        importance = np.pad(importance, (0, n_atoms - len(importance)))
    importance = importance[:n_atoms]
    
    # Create MolDraw2D canvas
    d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
    
    # Render via SimilarityMaps
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        importance.tolist(),
        draw2d=d2d,
        colorMap='RdYlGn_r',
        sigma=0.4,
    )
    d2d.FinishDrawing()
    
    png_data = d2d.GetDrawingText()
    img_b64 = base64.b64encode(png_data).decode('utf-8')
    
    # Build atom info
    atom_info = []
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_info.append({
            'idx': i,
            'symbol': atom.GetSymbol(),
            'importance': round(float(importance[i]), 3),
            'atomic_num': atom.GetAtomicNum(),
            'is_aromatic': atom.GetIsAromatic(),
        })
    
    sorted_atoms = sorted(atom_info, key=lambda a: a['importance'], reverse=True)
    
    return {
        'image_base64': img_b64,
        'image_mime': 'image/png',
        'atom_importance': atom_info,
        'top_atoms': sorted_atoms[:5],
        'n_atoms': n_atoms,
        'smiles': smiles,
    }


if __name__ == '__main__':
    """Quick test with famous molecules"""
    from train_models import WEIGHTS_DIR
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cp = torch.load(WEIGHTS_DIR / 'odor_gnn_v5.pt', map_location=device, weights_only=False)
    model = OdorGATv5(bert_dim=384).to(device)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()
    
    test_molecules = {
        'Vanillin': 'O=Cc1ccc(O)c(OC)c1',
        'Limonene': 'CC(=C)C1CCC(C)CC1',
        'Eugenol': 'COc1cc(CC=C)ccc1O',
        'Allyl_sulfide': 'C=CCSSCC=C',
        'Benzaldehyde': 'O=Cc1ccccc1',
        'Linalool': 'CC(=CCC/C(=C/CO)C)C',
    }
    
    save_dir = Path(__file__).parent.parent / 'data' / 'attention_maps'
    save_dir.mkdir(exist_ok=True)
    
    for name, smiles in test_molecules.items():
        result = generate_attention_heatmap(smiles, model, device=str(device))
        
        if 'error' in result:
            print(f"  x {name}: {result['error']}")
            continue
        
        png_path = save_dir / f'{name}.png'
        with open(png_path, 'wb') as f:
            f.write(base64.b64decode(result['image_base64']))
        
        top3 = result['top_atoms'][:3]
        top_str = ', '.join(f"{a['symbol']}({a['importance']:.2f})" for a in top3)
        print(f"  OK {name}: {result['n_atoms']} atoms | Top: {top_str}")
    
    print(f"\n  Saved to: {save_dir}")
