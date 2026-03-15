"""
UMAP Odor Space Visualization
4,718 molecules × 20d → UMAP 2D → Interactive Plotly HTML

Proves the model understands "chemical geography of scents"
by showing clusters of floral, woody, citrus, etc.
"""
import csv, sys, os, json, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_odor_space(device='cuda'):
    """Build 20d odor vectors for all molecules using ensemble."""
    from train_models import WEIGHTS_DIR, TrainableOdorNetV4
    from models.odor_gat_v5 import OdorGATv5, smiles_to_graph, ODOR_DIMENSIONS, N_DIM
    from rdkit import Chem
    
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load models
    print("  Loading models...")
    cp4 = torch.load(WEIGHTS_DIR / 'odor_gnn.pt', map_location=dev, weights_only=True)
    v4 = TrainableOdorNetV4(input_dim=384).to(dev)
    v4.load_state_dict(cp4['model_state_dict'])
    v4.eval()
    
    cp5 = torch.load(WEIGHTS_DIR / 'odor_gnn_v5.pt', map_location=dev, weights_only=False)
    v5 = OdorGATv5(bert_dim=384).to(dev)
    v5.load_state_dict(cp5['model_state_dict'])
    v5.eval()
    
    # ChemBERTa cache
    cache = np.load(WEIGHTS_DIR / 'chemberta_cache.npz')
    bert_cache = {s: cache['embeddings'][i] for i, s in enumerate(cache['smiles'])}
    
    # Load all molecules
    gs_path = Path(__file__).parent.parent / 'data' / 'curated_GS_LF_merged_4983.csv'
    with open(gs_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        odor_cols = header[1:]
        rows = list(reader)
    
    print(f"  Processing {len(rows)} molecules...")
    t0 = time.time()
    
    results = []
    skipped = 0
    
    with torch.no_grad():
        for i, row in enumerate(rows):
            smiles = row[0]
            
            # Get canonical SMILES
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                skipped += 1
                continue
            can = Chem.MolToSmiles(mol)
            
            # ChemBERTa embedding
            emb = bert_cache.get(can, bert_cache.get(smiles))
            if emb is None:
                skipped += 1
                continue
            
            # v4 prediction
            x4 = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(dev)
            p4 = v4(x4).squeeze(0).cpu().numpy()
            
            # v5 prediction
            graph = smiles_to_graph(can, device=dev)
            if graph is None:
                # v4 only
                pred = p4
            else:
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=dev)
                x5 = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(dev)
                p5 = v5(graph, x5).squeeze(0).cpu().numpy()
                pred = 0.50 * p4 + 0.50 * p5  # Ensemble
            
            pred = np.clip(pred, 0, 1)
            
            # Determine dominant dimensions
            top_idx = np.argsort(pred)[::-1]
            dominant = ODOR_DIMENSIONS[top_idx[0]]
            second = ODOR_DIMENSIONS[top_idx[1]]
            
            # Ground truth labels
            active = [col for col, val in zip(odor_cols, row[1:]) if val == '1']
            
            results.append({
                'smiles': can,
                'vector': pred.tolist(),
                'dominant': dominant,
                'second': second,
                'labels': active[:5],
                'strength': float(pred[top_idx[0]]),
            })
            
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(rows)} processed...")
    
    elapsed = time.time() - t0
    print(f"  Done: {len(results)} molecules in {elapsed:.1f}s (skipped {skipped})")
    
    return results, ODOR_DIMENSIONS


def generate_umap_html(results, odor_dims, output_path):
    """Generate interactive Plotly HTML with UMAP 2D projection."""
    import umap
    import plotly.graph_objects as go
    
    vectors = np.array([r['vector'] for r in results])
    
    print(f"  Running UMAP on {vectors.shape} matrix...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        random_state=42,
        n_components=2,
    )
    embedding = reducer.fit_transform(vectors)
    print(f"  UMAP done in {time.time()-t0:.1f}s")
    
    # Color assignment by dominant dimension
    color_map = {
        'sweet': '#FF6B9D',    'sour': '#FFC107',     'woody': '#8D6E63',
        'floral': '#E91E63',   'citrus': '#FF9800',   'spicy': '#F44336',
        'musk': '#9C27B0',     'fresh': '#00BCD4',    'green': '#4CAF50',
        'warm': '#FF5722',     'fruity': '#FF4081',    'smoky': '#607D8B',
        'powdery': '#E1BEE7',  'aquatic': '#03A9F4',  'herbal': '#66BB6A',
        'amber': '#FFA000',    'leather': '#5D4037',   'earthy': '#795548',
        'ozonic': '#B3E5FC',   'metallic': '#90A4AE',
    }
    
    # Group by dominant dimension
    groups = {}
    for i, r in enumerate(results):
        d = r['dominant']
        if d not in groups:
            groups[d] = {'x': [], 'y': [], 'text': [], 'color': color_map.get(d, '#999')}
        groups[d]['x'].append(float(embedding[i, 0]))
        groups[d]['y'].append(float(embedding[i, 1]))
        labels = ', '.join(r['labels'][:3]) if r['labels'] else 'unknown'
        groups[d]['text'].append(
            f"SMILES: {r['smiles'][:40]}<br>"
            f"Dominant: {r['dominant']} ({r['strength']:.2f})<br>"
            f"Labels: {labels}"
        )
    
    fig = go.Figure()
    
    # Sort groups by size (largest first for better legend)
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]['x']), reverse=True)
    
    for dim, data in sorted_groups:
        fig.add_trace(go.Scattergl(
            x=data['x'],
            y=data['y'],
            mode='markers',
            name=f"{dim} ({len(data['x'])})",
            marker=dict(
                color=data['color'],
                size=4,
                opacity=0.7,
                line=dict(width=0.3, color='white'),
            ),
            text=data['text'],
            hoverinfo='text',
        ))
    
    fig.update_layout(
        title=dict(
            text=f"AI Odor Space — {len(results):,} Molecules × 20d → UMAP 2D",
            font=dict(size=18, family='Inter, sans-serif'),
        ),
        xaxis=dict(title='UMAP-1', showgrid=False, zeroline=False),
        yaxis=dict(title='UMAP-2', showgrid=False, zeroline=False),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#16213e',
        font=dict(color='#e0e0e0', family='Inter, sans-serif'),
        legend=dict(
            font=dict(size=10),
            bgcolor='rgba(22,33,62,0.8)',
            bordercolor='rgba(255,255,255,0.1)',
        ),
        width=1200,
        height=800,
        hovermode='closest',
    )
    
    # Save HTML
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"  Saved: {output_path}")
    
    # Save data for API
    data_path = output_path.with_suffix('.json')
    save_data = {
        'embedding': embedding.tolist(),
        'molecules': [{
            'smiles': r['smiles'],
            'dominant': r['dominant'],
            'second': r['second'],
            'strength': r['strength'],
            'labels': r['labels'],
        } for r in results],
        'n_molecules': len(results),
        'dimensions': list(odor_dims),
    }
    with open(data_path, 'w') as f:
        json.dump(save_data, f)
    print(f"  Data saved: {data_path}")
    
    return embedding


if __name__ == '__main__':
    output = Path(__file__).parent.parent / 'data' / 'odor_space_umap.html'
    
    results, odor_dims = build_odor_space()
    generate_umap_html(results, odor_dims, output)
    
    # Stats
    from collections import Counter
    dom_counts = Counter(r['dominant'] for r in results)
    print(f"\n  Top dominant dimensions:")
    for dim, count in dom_counts.most_common(10):
        print(f"    {dim:12s}: {count:4d} ({count/len(results)*100:.1f}%)")
