"""
Retrain OdorGNN v4 with REAL experimental odor labels
from GoodScents + Leffingwell curated dataset (4,983 molecules, 138 descriptors)
"""
import os, sys, csv, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights")
CACHE_PATH  = os.path.join(WEIGHTS_DIR, "chemberta_cache.npz")
DATA_PATH   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "curated_GS_LF_merged_4983.csv")
SAVE_PATH   = os.path.join(WEIGHTS_DIR, "odor_gnn.pt")

# Our 22d odor space (matches train_models.py)
ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
    'fatty', 'waxy',  # NEW: co-occurrence upgraded
]

# ================================================================
# Map 138 GoodScents descriptors → 20d odor dimensions
# Each GS label contributes to one or more of our dimensions
# Weights based on semantic closeness
# ================================================================
LABEL_MAP = {
    # → sweet (idx 0)
    'sweet':       {'sweet': 1.0},
    'caramellic':  {'sweet': 0.8, 'warm': 0.3},
    'honey':       {'sweet': 0.7, 'floral': 0.2, 'warm': 0.2},
    'vanilla':     {'sweet': 0.8, 'warm': 0.3},
    'sugary':      {'sweet': 0.9},
    'chocolate':   {'sweet': 0.6, 'warm': 0.3},
    'butterscotch':{'sweet': 0.7, 'warm': 0.3},
    'maple':       {'sweet': 0.7, 'woody': 0.2},
    'cotton candy':{'sweet': 0.9},
    'jammy':       {'sweet': 0.6, 'fruity': 0.4},
    
    # → sour (idx 1)
    'sour':        {'sour': 1.0},
    'acidic':      {'sour': 0.8},
    'acetic':      {'sour': 0.7},
    'pungent':     {'sour': 0.5, 'spicy': 0.3},
    'sharp':       {'sour': 0.4, 'fresh': 0.3},
    'vinegar':     {'sour': 0.9},
    'fermented':   {'sour': 0.5, 'earthy': 0.3},
    
    # → woody (idx 2)
    'woody':       {'woody': 1.0},
    'cedarwood':   {'woody': 0.9, 'warm': 0.2},
    'sandalwood':  {'woody': 0.7, 'warm': 0.4, 'sweet': 0.2},
    'pine':        {'woody': 0.7, 'fresh': 0.4, 'green': 0.2},
    'balsamic':    {'woody': 0.5, 'warm': 0.4, 'sweet': 0.2},
    'camphoreous': {'woody': 0.4, 'fresh': 0.5, 'herbal': 0.3},
    'bark':        {'woody': 0.8, 'earthy': 0.2},
    'resinous':    {'woody': 0.6, 'warm': 0.3},
    
    # → floral (idx 3)
    'floral':      {'floral': 1.0},
    'rose':        {'floral': 0.9, 'sweet': 0.2},
    'jasmine':     {'floral': 0.8, 'sweet': 0.3},
    'lily':        {'floral': 0.8, 'fresh': 0.2},
    'violet':      {'floral': 0.7, 'powdery': 0.3, 'sweet': 0.2},
    'orchid':      {'floral': 0.7, 'sweet': 0.2},
    'muguet':      {'floral': 0.8, 'fresh': 0.3, 'green': 0.2},
    'gardenia':    {'floral': 0.8, 'sweet': 0.3},
    'magnolia':    {'floral': 0.7, 'fresh': 0.2},
    'chrysanthemum': {'floral': 0.6, 'herbal': 0.3},
    'geranium':    {'floral': 0.6, 'green': 0.3, 'herbal': 0.2},
    'lavender':    {'floral': 0.5, 'herbal': 0.5, 'fresh': 0.3},
    'hyacinth':    {'floral': 0.7, 'green': 0.3},
    
    # → citrus (idx 4)
    'citrus':      {'citrus': 1.0},
    'lemon':       {'citrus': 0.9, 'fresh': 0.3},
    'orange':      {'citrus': 0.7, 'sweet': 0.3, 'fruity': 0.2},
    'grapefruit':  {'citrus': 0.8, 'sour': 0.2},
    'lime':        {'citrus': 0.8, 'fresh': 0.3, 'sour': 0.2},
    'bergamot':    {'citrus': 0.7, 'floral': 0.2, 'fresh': 0.2},
    'mandarin':    {'citrus': 0.7, 'sweet': 0.3},
    'tangerine':   {'citrus': 0.7, 'sweet': 0.3},
    'aldehydic':   {'citrus': 0.3, 'fresh': 0.4, 'powdery': 0.3},
    
    # → spicy (idx 5)
    'spicy':       {'spicy': 1.0},
    'cinnamon':    {'spicy': 0.8, 'warm': 0.4, 'sweet': 0.2},
    'clove':       {'spicy': 0.8, 'warm': 0.3},
    'pepper':      {'spicy': 0.7, 'warm': 0.2},
    'anise':       {'spicy': 0.6, 'herbal': 0.3, 'sweet': 0.2},
    'ginger':      {'spicy': 0.6, 'fresh': 0.3, 'citrus': 0.2},
    'cardamom':    {'spicy': 0.5, 'warm': 0.3, 'herbal': 0.2},
    'nutmeg':      {'spicy': 0.6, 'warm': 0.4},
    'savory':      {'spicy': 0.4, 'herbal': 0.4},
    
    # → musk (idx 6)
    'musk':        {'musk': 1.0},
    'musky':       {'musk': 0.9, 'warm': 0.2},
    'animalic':    {'musk': 0.6, 'leather': 0.3, 'warm': 0.2},
    'ambery':      {'musk': 0.3, 'amber': 0.7, 'warm': 0.3},
    
    # → fresh (idx 7)
    'fresh':       {'fresh': 1.0},
    'clean':       {'fresh': 0.8, 'aquatic': 0.2},
    'cooling':     {'fresh': 0.7, 'ozonic': 0.3},
    'airy':        {'fresh': 0.6, 'ozonic': 0.3},
    'ethereal':    {'fresh': 0.5, 'ozonic': 0.3},
    
    # → green (idx 8)
    'green':       {'green': 1.0},
    'grassy':      {'green': 0.8, 'fresh': 0.2},
    'leafy':       {'green': 0.8, 'herbal': 0.2},
    'vegetable':   {'green': 0.6, 'earthy': 0.3},
    'mossy':       {'green': 0.5, 'earthy': 0.4, 'woody': 0.2},
    'tea':         {'green': 0.5, 'herbal': 0.4, 'fresh': 0.2},
    'hay':         {'green': 0.5, 'warm': 0.3, 'sweet': 0.2},
    
    # → warm (idx 9)
    'warm':        {'warm': 1.0},
    'bready':      {'warm': 0.6, 'sweet': 0.3},
    'toasted':     {'warm': 0.6, 'smoky': 0.3},
    'roasted':     {'warm': 0.5, 'smoky': 0.4, 'earthy': 0.2},
    'cooked':      {'warm': 0.5, 'sweet': 0.2},
    'burnt':       {'warm': 0.3, 'smoky': 0.7},
    
    # → fruity (idx 10)
    'fruity':      {'fruity': 1.0},
    'apple':       {'fruity': 0.8, 'fresh': 0.2, 'sweet': 0.2},
    'banana':      {'fruity': 0.8, 'sweet': 0.3},
    'pear':        {'fruity': 0.7, 'fresh': 0.3, 'green': 0.2},
    'peach':       {'fruity': 0.8, 'sweet': 0.3},
    'plum':        {'fruity': 0.7, 'sweet': 0.3},
    'grape':       {'fruity': 0.7, 'sweet': 0.3},
    'berry':       {'fruity': 0.7, 'sweet': 0.3},
    'tropical':    {'fruity': 0.7, 'sweet': 0.3, 'fresh': 0.2},
    'pineapple':   {'fruity': 0.8, 'citrus': 0.2, 'sweet': 0.2},
    'melon':       {'fruity': 0.7, 'fresh': 0.3, 'green': 0.2},
    'cherry':      {'fruity': 0.7, 'sweet': 0.3},
    'coconut':     {'fruity': 0.5, 'sweet': 0.4, 'warm': 0.2},
    'cassis':      {'fruity': 0.6, 'green': 0.3},
    'black currant':{'fruity': 0.6, 'green': 0.3},
    'apricot':     {'fruity': 0.8, 'sweet': 0.3},
    'raspberry':   {'fruity': 0.7, 'sweet': 0.3},
    'strawberry':  {'fruity': 0.7, 'sweet': 0.3},
    'winey':       {'fruity': 0.4, 'warm': 0.3, 'sweet': 0.2},
    'alcoholic':   {'fruity': 0.2, 'warm': 0.3, 'fresh': 0.2},
    'rum':         {'fruity': 0.3, 'warm': 0.4, 'sweet': 0.3},
    'brandy':      {'fruity': 0.3, 'warm': 0.4, 'woody': 0.2},
    
    # → smoky (idx 11)
    'smoky':       {'smoky': 1.0},
    'phenolic':    {'smoky': 0.6, 'warm': 0.2},
    'tarry':       {'smoky': 0.7, 'warm': 0.2},
    'creosote':    {'smoky': 0.8},
    
    # → powdery (idx 12)
    'powdery':     {'powdery': 1.0},
    'dusty':       {'powdery': 0.6, 'earthy': 0.3},
    'chalky':      {'powdery': 0.7},
    'soft':        {'powdery': 0.5, 'floral': 0.3, 'sweet': 0.2},
    
    # → aquatic (idx 13)
    'marine':      {'aquatic': 0.9, 'fresh': 0.3},
    'watery':      {'aquatic': 0.7, 'fresh': 0.3},
    'oceanic':     {'aquatic': 0.9},
    'ozone':       {'aquatic': 0.3, 'ozonic': 0.7, 'fresh': 0.3},
    
    # → herbal (idx 14)
    'herbal':      {'herbal': 1.0},
    'minty':       {'herbal': 0.5, 'fresh': 0.5},
    'mint':        {'herbal': 0.5, 'fresh': 0.5},
    'basil':       {'herbal': 0.7, 'green': 0.3, 'spicy': 0.2},
    'thyme':       {'herbal': 0.7, 'spicy': 0.2, 'green': 0.2},
    'rosemary':    {'herbal': 0.6, 'woody': 0.2, 'fresh': 0.2},
    'medicinal':   {'herbal': 0.4, 'fresh': 0.3},
    'aromatic':    {'herbal': 0.5, 'spicy': 0.3},
    
    # → amber (idx 15)
    'amber':       {'amber': 1.0},
    'resin':       {'amber': 0.5, 'woody': 0.4, 'warm': 0.3},
    'incense':     {'amber': 0.5, 'smoky': 0.3, 'woody': 0.3},
    'oriental':    {'amber': 0.5, 'warm': 0.4, 'spicy': 0.3},
    
    # → leather (idx 16)
    'leather':     {'leather': 1.0},
    'suede':       {'leather': 0.7, 'powdery': 0.2},
    'tobacco':     {'leather': 0.4, 'warm': 0.3, 'smoky': 0.2, 'sweet': 0.2},
    
    # → earthy (idx 17)
    'earthy':      {'earthy': 1.0},
    'mushroom':    {'earthy': 0.8, 'warm': 0.2},
    'musty':       {'earthy': 0.6, 'warm': 0.2},
    'soil':        {'earthy': 0.9},
    'damp':        {'earthy': 0.5, 'aquatic': 0.3},
    'fungal':      {'earthy': 0.7},
    'mossy':       {'earthy': 0.5, 'green': 0.4},
    'nutty':       {'earthy': 0.4, 'warm': 0.3, 'sweet': 0.2},
    'oily':        {'earthy': 0.3, 'warm': 0.2, 'fruity': 0.1},
    'fatty':       {'earthy': 0.2, 'warm': 0.3},
    'waxy':        {'earthy': 0.2, 'warm': 0.2, 'sweet': 0.1},
    
    # → ozonic (idx 18)
    'ozonic':      {'ozonic': 1.0},
    'ozone':       {'ozonic': 0.8, 'fresh': 0.3},
    
    # → metallic (idx 19)
    'metallic':    {'metallic': 1.0},
    'sulfurous':   {'metallic': 0.5, 'sour': 0.3},
    'chemical':    {'metallic': 0.4, 'fresh': 0.2},
    'solvent':     {'metallic': 0.3, 'fresh': 0.3},
    
    # === FOOD / MISC — distribute across related dims ===
    'meaty':       {'warm': 0.4, 'smoky': 0.3, 'earthy': 0.3},
    'beefy':       {'warm': 0.4, 'smoky': 0.3, 'earthy': 0.3},
    'garlic':      {'spicy': 0.3, 'earthy': 0.3, 'metallic': 0.2},
    'onion':       {'spicy': 0.3, 'earthy': 0.3, 'sour': 0.2},
    'alliaceous':  {'spicy': 0.3, 'earthy': 0.3, 'metallic': 0.2},
    'fishy':       {'aquatic': 0.3, 'metallic': 0.3},
    'cheesy':      {'sour': 0.3, 'warm': 0.3, 'earthy': 0.2},
    'milky':       {'sweet': 0.4, 'warm': 0.3},
    'creamy':      {'sweet': 0.4, 'warm': 0.3, 'powdery': 0.2},
    'buttery':     {'sweet': 0.3, 'warm': 0.4},
    'corn':        {'sweet': 0.3, 'green': 0.2, 'warm': 0.2},
    'popcorn':     {'warm': 0.4, 'sweet': 0.3},
    'coffee':      {'warm': 0.4, 'smoky': 0.3, 'earthy': 0.2},
    'odorless':    {},  # zero vector

    # → fatty (idx 20)
    'fatty':       {'fatty': 1.0},
    'oily':        {'fatty': 0.8, 'warm': 0.2},
    'greasy':      {'fatty': 0.7},
    'sebaceous':   {'fatty': 0.6, 'warm': 0.2},
    'lard':        {'fatty': 0.7, 'warm': 0.3},
    'tallow':      {'fatty': 0.8, 'warm': 0.2},
    'rancid':      {'fatty': 0.6, 'sour': 0.3},

    # → waxy (idx 21)
    'waxy':        {'waxy': 1.0},
    'paraffin':    {'waxy': 0.8},
    'candle':      {'waxy': 0.7, 'warm': 0.2},
    'beeswax':     {'waxy': 0.8, 'sweet': 0.2},
    'ceraceous':   {'waxy': 0.9},
    'petroleum':   {'waxy': 0.5, 'smoky': 0.3},
}


def load_dataset():
    """Load the curated GoodScents+Leffingwell dataset and ChemBERTa cache"""
    
    # Load ChemBERTa cache
    print("[1] Loading ChemBERTa cache ...")
    cache_data = np.load(CACHE_PATH, allow_pickle=True)
    emb_arr = cache_data['embeddings']
    smiles_arr = list(cache_data['smiles'])
    hidden_size = int(cache_data['hidden_size'])
    
    # Build canonical SMILES → index map
    from rdkit import Chem
    canon_map = {}
    for i, s in enumerate(smiles_arr):
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            canonical = Chem.MolToSmiles(mol)
            canon_map[canonical] = i
    
    print(f"    Cache: {len(smiles_arr)} molecules, {hidden_size}d")
    
    # Load curated odor dataset
    print("[2] Loading curated GS+LF dataset ...")
    with open(DATA_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    
    odor_cols = header[1:]  # first column is SMILES
    print(f"    Dataset: {len(rows)} molecules, {len(odor_cols)} descriptors")
    
    # Build training data
    print("[3] Building training pairs ...")
    X_list = []  # 384d ChemBERTa features
    Y_list = []  # 22d odor labels (soft)
    matched = 0
    unmatched = 0
    
    for row in rows:
        smiles = row[0]
        
        # Canonicalize
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            unmatched += 1
            continue
        canonical = Chem.MolToSmiles(mol)
        
        # Look up in ChemBERTa cache
        if canonical not in canon_map:
            unmatched += 1
            continue
        
        emb_idx = canon_map[canonical]
        features = emb_arr[emb_idx]
        
        # Convert 138 binary labels → 20d soft vector
        active_labels = [col.lower().strip() for col, val in zip(odor_cols, row[1:]) if val == "1"]
        
        vec_20d = np.zeros(len(ODOR_DIMENSIONS), dtype=np.float32)
        for label in active_labels:
            if label in LABEL_MAP:
                for dim_name, weight in LABEL_MAP[label].items():
                    dim_idx = ODOR_DIMENSIONS.index(dim_name)
                    vec_20d[dim_idx] += weight
        
        # Normalize to [0, 1]
        max_val = vec_20d.max()
        if max_val > 0:
            vec_20d = vec_20d / max_val
        else:
            unmatched += 1
            continue
        
        X_list.append(features)
        Y_list.append(vec_20d)
        matched += 1
    
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    n_mapped = sum(1 for row in rows 
                  for col, val in zip(odor_cols, row[1:]) 
                  if val == "1" and col.lower().strip() in LABEL_MAP)
    n_total_labels = sum(1 for row in rows 
                        for val in row[1:] 
                        if val == "1")
    coverage = n_mapped / n_total_labels * 100 if n_total_labels > 0 else 0
    
    print(f"    Matched: {matched} molecules")
    print(f"    Unmatched: {unmatched}")
    print(f"    Label coverage: {coverage:.1f}% ({n_mapped}/{n_total_labels})")
    print(f"    Feature shape: {X.shape}")
    print(f"    Label shape: {Y.shape}")
    
    return X, Y


def train_v4_real(epochs=400, lr=0.002, batch_size=64):
    """Train OdorGNN v4 with REAL experimental labels"""
    from train_models import TrainableOdorNetV4
    
    X, Y = load_dataset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[4] Training OdorGNN v4 (REAL labels) on {device}")
    
    # Create dataset
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    dataset = TensorDataset(X_t, Y_t)
    
    # 80/20 split
    n = len(dataset)
    n_val = int(n * 0.2)
    n_train = n - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], 
                                       generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    print(f"    Train: {n_train}, Val: {n_val}")
    
    # Model
    model = TrainableOdorNetV4(input_dim=384).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Model params: {n_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss: MSE + cosine similarity  
    mse_loss = nn.MSELoss()
    cos_loss = nn.CosineEmbeddingLoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 40
    patience_counter = 0
    t0 = time.time()
    
    print(f"\n{'='*60}")
    print(f"  OdorGNN v4 Training (REAL GoodScents+Leffingwell labels)")
    print(f"  {n_train} train / {n_val} val | {epochs} epochs | lr={lr}")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = mse_loss(pred, yb) + 0.3 * (1 - nn.functional.cosine_similarity(pred, yb).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_cos_sim = 0
        val_count = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = mse_loss(pred, yb)
                cos_sim = nn.functional.cosine_similarity(pred, yb).mean()
                val_loss += loss.item()
                val_cos_sim += cos_sim.item()
                val_count += 1
        val_loss /= val_count
        val_cos_sim /= val_count
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_cos_sim': val_cos_sim,
                'input_dim': 384,
                'n_train': n_train,
                'n_val': n_val,
                'n_params': n_params,
                'data_source': 'GoodScents+Leffingwell_real',
            }, SAVE_PATH)
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]['lr']
            saved = " *" if patience_counter == 0 else ""
            print(f"  Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | CosSim: {val_cos_sim:.3f} | LR: {lr_now:.6f} | {elapsed:.1f}s{saved}")
        
        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  DONE! Best epoch: {best_epoch}, Val loss: {best_val_loss:.4f}")
    print(f"  Best CosSim: {val_cos_sim:.3f}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Saved: {SAVE_PATH}")
    print(f"{'='*60}")
    
    return model


if __name__ == "__main__":
    train_v4_real()
