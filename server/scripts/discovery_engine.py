"""
Odor Discovery Engine — PubChem novel molecule search
Uses sequential CID ranges (1-50000) for reliable downloads,
then computes 20d odor vectors via ensemble for cosine similarity search.

v2.1: FAISS HNSW index for O(log N) vector search (~2ms vs ~2s)
"""
import csv, json, sys, os, time, logging
import numpy as np
import torch
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.odor_gat_v5 import OdorGATv5, smiles_to_graph, ODOR_DIMENSIONS, N_DIM
from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

log = logging.getLogger("perfumery.discovery")

DATA_DIR = Path(__file__).parent.parent / 'data'
DISCOVERY_DB = DATA_DIR / 'discovery_db.npz'

# ===== Module-level Singleton Cache =====
_db_cache = None  # (smiles, vectors, dims)
_idf_cache = None
_faiss_index = None  # FAISS HNSW index
_faiss_idf_index = None  # FAISS index for IDF-weighted vectors 
_db_mtime = 0  # Track file modification for invalidation


def _get_db():
    """Load and cache discovery DB (singleton). Invalidates on file change."""
    global _db_cache, _db_mtime, _faiss_index, _faiss_idf_index, _idf_cache
    if not DISCOVERY_DB.exists():
        return None, None, None
    
    mtime = os.path.getmtime(DISCOVERY_DB)
    if _db_cache is not None and mtime == _db_mtime:
        return _db_cache
    
    db = np.load(DISCOVERY_DB, allow_pickle=True)
    _db_cache = (db['smiles'], db['vectors'], db['dimensions'])
    _db_mtime = mtime
    # Invalidate FAISS indexes when DB changes
    _faiss_index = None
    _faiss_idf_index = None
    _idf_cache = None
    log.info(f"Discovery DB loaded: {len(_db_cache[0])} molecules")
    return _db_cache


def _get_idf():
    """Cached IDF weights (recomputes only when DB changes)."""
    global _idf_cache, _db_mtime
    _, vectors, _ = _get_db()
    if vectors is None:
        return np.ones(N_DIM, dtype=np.float32)
    
    # Use DB mtime as cache key
    if _idf_cache is not None:
        return _idf_cache
    
    n = len(vectors)
    threshold = 0.02
    counts = np.sum(vectors > threshold, axis=0)
    idf = np.log(n / (1.0 + counts))
    idf = idf / (idf.mean() + 1e-8)
    _idf_cache = idf.astype(np.float32)
    return _idf_cache


def download_pubchem_sequential(target=5000, batch_size=200, start_cid=1):
    """Download from PubChem using sequential CID ranges (reliable)"""
    import urllib.request
    
    cache_path = DATA_DIR / 'pubchem_novel_smiles.txt'
    if cache_path.exists():
        with open(cache_path) as f:
            smiles = [l.strip() for l in f if l.strip()]
        if len(smiles) >= target // 2:
            print(f"  Using cached {len(smiles)} SMILES from {cache_path.name}")
            return smiles
    
    all_smiles = []
    cid = start_cid
    max_cid = 60000  # PubChem small CIDs are dense
    
    # Also load existing known molecules to exclude them
    known = set()
    gs_path = DATA_DIR / 'curated_GS_LF_merged_4983.csv'
    if gs_path.exists():
        with open(gs_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                mol = Chem.MolFromSmiles(row[0])
                if mol:
                    known.add(Chem.MolToSmiles(mol))
    print(f"  Excluding {len(known)} known molecules")
    
    print(f"  Downloading PubChem CIDs {start_cid}~{max_cid} (target: {target})...")
    
    while cid < max_cid and len(all_smiles) < target:
        cid_list = list(range(cid, min(cid + batch_size, max_cid)))
        cid_str = ','.join(str(c) for c in cid_list)
        
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
               f"{cid_str}/property/CanonicalSMILES,MolecularWeight/JSON")
        
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            
            props = data.get('PropertyTable', {}).get('Properties', [])
            for p in props:
                smi = p.get('ConnectivitySMILES', p.get('CanonicalSMILES', ''))
                mw_str = p.get('MolecularWeight', '0')
                try:
                    mw = float(mw_str)
                except (ValueError, TypeError):
                    mw = 0
                
                if smi and 40 < mw < 600:
                    mol = Chem.MolFromSmiles(smi)
                    if mol and mol.GetNumAtoms() >= 3:
                        can = Chem.MolToSmiles(mol)
                        if can not in known:
                            all_smiles.append(can)
                            known.add(can)
        except Exception:
            pass
        
        cid += batch_size
        
        if (cid - start_cid) % 2000 == 0:
            print(f"    CID {cid}: {len(all_smiles)} novel molecules")
        
        time.sleep(0.15)  # Rate limit
    
    print(f"  Downloaded {len(all_smiles)} novel molecules")
    
    with open(cache_path, 'w') as f:
        for smi in all_smiles:
            f.write(smi + '\n')
    
    return all_smiles


def build_discovery_db(smiles_list, device='cuda'):
    """Compute 20d odor vectors for all molecules via ensemble"""
    from train_models import WEIGHTS_DIR, TrainableOdorNetV4
    
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print("  Loading ensemble models...")
    cp4 = torch.load(WEIGHTS_DIR / 'odor_gnn.pt', map_location=dev, weights_only=True)
    v4 = TrainableOdorNetV4(input_dim=384).to(dev)
    v4.load_state_dict(cp4['model_state_dict'])
    v4.eval()
    
    cp5 = torch.load(WEIGHTS_DIR / 'odor_gnn_v5.pt', map_location=dev, weights_only=False)
    v5 = OdorGATv5(bert_dim=384).to(dev)
    v5.load_state_dict(cp5['model_state_dict'])
    v5.eval()
    
    cache_data = np.load(WEIGHTS_DIR / 'chemberta_cache.npz')
    bert_cache = {s: cache_data['embeddings'][i] for i, s in enumerate(cache_data['smiles'])}
    
    print(f"  Computing odor vectors for {len(smiles_list)} molecules...")
    t0 = time.time()
    
    valid_smiles = []
    vectors = []
    
    with torch.no_grad():
        for i, smiles in enumerate(smiles_list):
            emb = bert_cache.get(smiles)
            if emb is None:
                emb = np.zeros(384, dtype=np.float32)
            
            x4 = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(dev)
            p4 = v4(x4).squeeze(0).cpu().numpy()
            
            graph = smiles_to_graph(smiles, device=dev)
            if graph is not None:
                graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=dev)
                x5 = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(dev)
                p5 = v5(graph, x5).squeeze(0).cpu().numpy()
                pred = 0.50 * p4 + 0.50 * p5
            else:
                pred = p4
            
            pred = np.clip(pred, 0, 1)
            valid_smiles.append(smiles)
            vectors.append(pred)
            
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(smiles_list)} processed")
    
    vectors = np.array(vectors, dtype=np.float32)
    elapsed = time.time() - t0
    
    np.savez_compressed(
        DISCOVERY_DB,
        smiles=np.array(valid_smiles, dtype=object),
        vectors=vectors,
        dimensions=np.array(ODOR_DIMENSIONS),
    )
    
    print(f"  Done: {len(valid_smiles)} molecules in {elapsed:.1f}s")
    print(f"  DB: {os.path.getsize(DISCOVERY_DB)/1024:.0f} KB")
    
    return valid_smiles, vectors


def _compute_idf_weights():
    """Compute IDF weights (cached via _get_idf singleton)."""
    return _get_idf()


def _orthogonal_remove(query, remove_dims):
    """Remove specified dimensions via orthogonal projection.
    
    Instead of naive subtraction (which creates negatives),
    we zero out the component along the unwanted direction.
    
    This is the 'Negative Prompting' technique.
    """
    result = query.copy()
    for dim_idx in remove_dims:
        # Create unit vector for this dimension
        direction = np.zeros(N_DIM, dtype=np.float32)
        direction[dim_idx] = 1.0
        
        # Remove projection onto this direction
        proj = np.dot(result, direction) * direction
        result = result - proj
    
    # Ensure non-negative
    result = np.clip(result, 0, None)
    return result


def _get_faiss_index(use_idf=True):
    """Build or retrieve FAISS HNSW index (singleton, auto-rebuilt on DB change).
    
    HNSW (Hierarchical Navigable Small World) provides O(log N) search
    vs O(N) brute-force cosine similarity.
    """
    global _faiss_index, _faiss_idf_index
    
    if not FAISS_AVAILABLE:
        return None
    
    # Check if we already have a cached index
    if use_idf and _faiss_idf_index is not None:
        return _faiss_idf_index
    if not use_idf and _faiss_index is not None:
        return _faiss_index
    
    _, vectors, _ = _get_db()
    if vectors is None:
        return None
    
    t0 = time.time()
    
    # Prepare vectors (L2-normalize for cosine → inner product)
    if use_idf:
        idf = _compute_idf_weights()
        v = vectors * idf[np.newaxis, :]
    else:
        v = vectors.copy()
    
    v = v.astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    v_normalized = v / norms
    
    d = v_normalized.shape[1]  # 20 dimensions
    n = v_normalized.shape[0]  # ~16K molecules
    
    # HNSW index — Inner Product on L2-normalized vectors = Cosine Similarity
    index = faiss.IndexHNSWFlat(d, 32)  # M=32 connections per node
    index.hnsw.efConstruction = 200     # Build quality
    index.hnsw.efSearch = 64            # Search quality
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.add(v_normalized)
    
    elapsed = time.time() - t0
    log.info(f"FAISS HNSW index built: {n} vectors × {d}d in {elapsed:.2f}s (IDF={use_idf})")
    print(f"[Discovery] FAISS HNSW index: {n} molecules, {elapsed:.3f}s (IDF={use_idf})")
    
    if use_idf:
        _faiss_idf_index = index
    else:
        _faiss_index = index
    
    return index


def search_by_odor_profile(query_vector, top_k=10, use_idf=True, 
                            negative_dims=None):
    """Find molecules matching a desired odor profile.
    
    Uses FAISS HNSW index for O(log N) search when available,
    falls back to brute-force cosine similarity otherwise.
    
    Args:
        query_vector: 20d target odor profile
        top_k: number of results
        use_idf: apply TF-IDF weighting (reduces sweet bias)
        negative_dims: list of dimension indices to remove via orthogonal projection
    """
    smiles, vectors, dims = _get_db()  # Cached singleton
    if smiles is None:
        return {'error': 'DB not built yet'}
    
    query = np.array(query_vector, dtype=np.float32)
    
    # --- Negative prompting: remove unwanted dimensions ---
    if negative_dims:
        query = _orthogonal_remove(query, negative_dims)
    
    t0 = time.time()
    search_method = 'brute_force'
    
    # --- Try FAISS first ---
    faiss_index = _get_faiss_index(use_idf=use_idf)
    
    if faiss_index is not None:
        search_method = 'faiss_hnsw'
        # Apply same weighting to query
        if use_idf:
            idf = _compute_idf_weights()
            q_weighted = query * idf
        else:
            q_weighted = query
        
        # L2-normalize query
        q_norm = q_weighted / (np.linalg.norm(q_weighted) + 1e-8)
        q_norm = q_norm.reshape(1, -1).astype(np.float32)
        
        # FAISS search — returns (distances, indices)
        sims_top, top_idx = faiss_index.search(q_norm, min(top_k, len(smiles)))
        sims_top = sims_top[0]  # Flatten
        top_idx = top_idx[0]
        
        # Filter out invalid indices (-1)
        valid = top_idx >= 0
        top_idx = top_idx[valid]
        sims_top = sims_top[valid]
    else:
        # Fallback: brute-force cosine similarity
        if use_idf:
            idf = _compute_idf_weights()
            q_weighted = query * idf
            v_weighted = vectors * idf[np.newaxis, :]
        else:
            q_weighted = query
            v_weighted = vectors
        
        q_norm = q_weighted / (np.linalg.norm(q_weighted) + 1e-8)
        v_norms = np.linalg.norm(v_weighted, axis=1, keepdims=True) + 1e-8
        v_norm = v_weighted / v_norms
        sims_all = v_norm @ q_norm
        
        top_idx = np.argsort(sims_all)[::-1][:top_k]
        sims_top = sims_all[top_idx]
    
    search_time_ms = (time.time() - t0) * 1000
    
    results = []
    idf_w = _compute_idf_weights() if use_idf else np.ones(N_DIM, dtype=np.float32)
    for i, idx in enumerate(top_idx):
        idx = int(idx)
        v = vectors[idx]  # Raw values
        v_idf = v * idf_w  # IDF-weighted for ranking dimensions
        td = np.argsort(v_idf)[::-1][:5]
        sim = float(sims_top[i]) if i < len(sims_top) else 0.0
        results.append({
            'smiles': str(smiles[idx]),
            'similarity': sim,
            'dominant': str(dims[td[0]]),
            'top_dimensions': [[str(dims[d]), round(float(v[d]), 3)] for d in td],
        })
    
    return {
        'results': results, 
        'db_size': int(len(smiles)),
        'query_desc': _desc(query, dims),
        'idf_applied': use_idf,
        'negative_dims': [str(dims[d]) for d in (negative_dims or [])],
        'search_method': search_method,
        'search_time_ms': round(search_time_ms, 2),
    }


def search_by_text(description, top_k=10):
    """Advanced text → 20d → search with TF-IDF + negative prompting.
    
    Supports:
        - 'sweet 60% woody 40%'       → weighted query
        - 'woody -sweet'               → negative prompting (remove sweet)
        - 'floral NOT sweet citrus'    → negative prompting
        - 'citrus fresh'               → simple query
    """
    import re
    query = np.zeros(N_DIM, dtype=np.float32)
    negative_dims = []
    dl = description.lower()
    
    # Parse negative prompts: "-sweet" or "NOT sweet"
    # Handle "NOT dim" syntax
    for m in re.finditer(r'(?:not|NOT|-)\s*(\w+)', dl):
        word = m.group(1)
        for i, dim in enumerate(ODOR_DIMENSIONS):
            if dim == word or word in dim:
                negative_dims.append(i)
    
    # Remove negative terms from description for positive parsing
    clean_dl = re.sub(r'(?:not|NOT|-)\s*\w+', '', dl).strip()
    
    # Parse positive terms with percentages: "sweet 60%"
    for m in re.finditer(r'(\w+)\s+(\d+)%', clean_dl):
        w, p = m.group(1), int(m.group(2))
        for i, dim in enumerate(ODOR_DIMENSIONS):
            if dim in w or w in dim:
                query[i] = p / 100.0
    
    # Parse remaining bare words: "citrus fresh"
    remaining_words = re.sub(r'\w+\s+\d+%', '', clean_dl).split()
    for word in remaining_words:
        word = word.strip()
        if len(word) < 3:
            continue
        for i, dim in enumerate(ODOR_DIMENSIONS):
            if dim == word or word in dim or dim in word:
                if query[i] == 0:  # Don't override percentage values
                    query[i] = 0.5
    
    if query.sum() == 0:
        return {'error': 'No positive odor dimensions found in: "%s"' % description}
    
    return search_by_odor_profile(query.tolist(), top_k, 
                                   use_idf=True, negative_dims=negative_dims)


def _desc(q, dims):
    parts = []
    active = [(float(q[i]), str(dims[i])) for i in range(min(len(dims), len(q))) if q[i] > 0.01]
    active.sort(reverse=True)
    return ' + '.join(f"{d} {v*100:.0f}%" for v, d in active[:5])


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  DISCOVERY ENGINE v2 — IDF + Negative Prompting")
    print("=" * 60)
    
    # Show IDF weights
    idf = _compute_idf_weights()
    print("\n  IDF Weights (higher = rarer = more important):")
    ranked = sorted(zip(ODOR_DIMENSIONS, idf), key=lambda x: x[1])
    for dim, w in ranked:
        bar = "█" * int(w * 10)
        print(f"    {dim:12s}: {w:.3f} {bar}")
    
    # Run search tests: compare IDF on vs off
    tests = [
        ("Sweet+Woody", "sweet 60% woody 40%"),
        ("Citrus Fresh", "citrus fresh"),
        ("Smoky Leather", "smoky"),
        ("Floral Light", "floral green"),
        ("Woody NOT Sweet", "woody -sweet"),
        ("Musky Warm", "musk warm"),
    ]
    
    print(f"\n{'='*60}")
    print(f"  SEARCH COMPARISON: Plain vs IDF-Weighted")
    print(f"{'='*60}")
    
    for name, q in tests:
        # Plain search (no IDF)
        old_query = np.zeros(N_DIM, dtype=np.float32)
        for i, dim in enumerate(ODOR_DIMENSIONS):
            if dim in q.lower():
                old_query[i] = 0.5
        
        import re
        for m in re.finditer(r'(\w+)\s+(\d+)%', q.lower()):
            w, p = m.group(1), int(m.group(2))
            for i, dim in enumerate(ODOR_DIMENSIONS):
                if dim in w or w in dim:
                    old_query[i] = p / 100.0
        
        plain = search_by_odor_profile(old_query.tolist(), top_k=3, use_idf=False)
        idf_result = search_by_text(q, top_k=3)
        
        print(f"\n  {name}: '{q}'")
        
        if 'error' not in plain:
            print(f"    [PLAIN]  ", end="")
            for m in plain['results'][:3]:
                print(f" {m['dominant']:8s}({m['similarity']:.3f})", end="")
            print()
        
        if 'error' not in idf_result:
            neg = idf_result.get('negative_dims', [])
            neg_str = f" (-{','.join(neg)})" if neg else ""
            print(f"    [IDF{neg_str}] ", end="")
            for m in idf_result['results'][:3]:
                print(f" {m['dominant']:8s}({m['similarity']:.3f})", end="")
            print()

