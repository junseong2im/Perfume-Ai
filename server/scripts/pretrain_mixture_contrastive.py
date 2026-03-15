"""
Priority 2: Contrastive Pre-training with 5330 Molecule Pairs
==============================================================
Train MixtureEncoder to predict pairwise distances BEFORE fine-tuning on DREAM.
Uses our 5330-molecule embedding DB to generate thousands of training pairs.

Architecture:
  MixtureEncoder(mol_A) → 64d 
  MixtureEncoder(mol_B) → 64d
  DistanceHead(zA, zB) → predicted_dist
  Target: cosine_distance(138d_A, 138d_B) 
  Loss: MSE(predicted_dist, target_dist)
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the existing architecture
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))
from dream_optimized_backtest import MixtureEncoder, SiameseMixtureNet

BASE = os.path.join(os.path.dirname(__file__), '..')
ODT_FILE = os.path.join(BASE, 'data', 'pom_upgrade', 'dream_odt.json')
PRETRAINED_PATH = os.path.join(BASE, 'models', 'mixture_encoder_pretrained.pt')

def generate_pairs_from_138d(engine, n_pairs=20000):
    """Generate contrastive pairs from our 138d prediction DB"""
    print(f"  Generating {n_pairs} contrastive pairs from POM DB...")
    
    # Get all molecules with SMILES from fragrance DB
    molecules = []
    for name, entry in engine._fragrance_db.items():
        smi = entry.get('smiles', '')
        if smi:
            try:
                pred = engine.predict_138d(smi)
                if pred is not None and np.any(pred > 0):
                    odt = entry.get('odt_log', -3.0)
                    molecules.append({
                        'smiles': smi,
                        'pred_138d': pred,
                        'odt_log': odt if odt is not None else -3.0,
                    })
            except:
                pass
    
    print(f"  Valid molecules: {len(molecules)}")
    
    if len(molecules) < 50:
        print("  [WARN] Too few molecules for pre-training")
        return []
    
    # Generate random pairs
    pairs = []
    for _ in range(n_pairs):
        i, j = np.random.choice(len(molecules), 2, replace=False)
        a = molecules[i]
        b = molecules[j]
        
        # Cosine distance as target
        va = a['pred_138d']
        vb = b['pred_138d']
        cos_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        target_dist = (1.0 - cos_sim) / 2.0  # Scale to ~0-1
        
        pairs.append({
            'pred_a': va,
            'pred_b': vb,
            'odt_a': a['odt_log'],
            'odt_b': b['odt_log'],
            'target': target_dist,
        })
    
    return pairs

def generate_pairs_from_cache(pred_cache, odt_cache, n_pairs=20000):
    """Faster: generate from cached 138d predictions"""
    print(f"  Generating {n_pairs} pairs from prediction cache ({len(pred_cache)} molecules)...")
    
    keys = [k for k, v in pred_cache.items() if v and len(v) >= 138]
    if len(keys) < 50:
        return []
    
    pairs = []
    for _ in range(n_pairs):
        i, j = np.random.choice(len(keys), 2, replace=False)
        ka, kb = keys[i], keys[j]
        
        va = np.array(pred_cache[ka][:138])
        vb = np.array(pred_cache[kb][:138])
        
        cos_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9)
        target_dist = (1.0 - cos_sim) / 2.0
        
        pairs.append({
            'pred_a': va,
            'pred_b': vb,
            'odt_a': odt_cache.get(ka, -3.0),
            'odt_b': odt_cache.get(kb, -3.0),
            'target': target_dist,
        })
    
    return pairs

def pretrain_encoder(model, pairs, epochs=50, lr=0.002, batch_size=64):
    """Pre-train MixtureEncoder with contrastive pairs"""
    device = torch.device('cpu')
    model = model.to(device)
    
    # Prepare tensors (single molecule = N=1 mixture)
    n = len(pairs)
    X_a = torch.FloatTensor(np.array([p['pred_a'][:138] for p in pairs]))  # [N, 138]
    X_b = torch.FloatTensor(np.array([p['pred_b'][:138] for p in pairs]))
    W_a = torch.FloatTensor(np.array([1.0 / (10**p['odt_a'] + 1e-6) for p in pairs]))
    W_b = torch.FloatTensor(np.array([1.0 / (10**p['odt_b'] + 1e-6) for p in pairs]))
    targets = torch.FloatTensor(np.array([p['target'] for p in pairs]))
    
    # Reshape for MixtureEncoder (single molecule = [B, 1, 138])
    X_a = X_a.unsqueeze(1)  # [N, 1, 138]
    X_b = X_b.unsqueeze(1)
    W_a = W_a.unsqueeze(1)  # [N, 1]
    W_b = W_b.unsqueeze(1)
    M = torch.zeros(n, 1, dtype=torch.bool)  # No padding
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Split: 90% train, 10% val
    split = int(0.9 * n)
    train_idx = list(range(split))
    val_idx = list(range(split, n))
    
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)
        
        train_loss = 0
        n_batches = 0
        
        for batch_start in range(0, len(train_idx), batch_size):
            idx = train_idx[batch_start:batch_start+batch_size]
            
            xa = X_a[idx]
            xb = X_b[idx]
            wa = W_a[idx]
            wb = W_b[idx]
            m = M[idx]
            t = targets[idx]
            
            optimizer.zero_grad()
            pred = model(xa, wa, m, xb, wb, m)
            loss = F.mse_loss(pred, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_a[val_idx], W_a[val_idx], M[val_idx],
                           X_b[val_idx], W_b[val_idx], M[val_idx])
            val_loss = F.mse_loss(val_pred, targets[val_idx]).item()
            
            val_r = np.corrcoef(val_pred.numpy(), targets[val_idx].numpy())[0, 1]
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss/n_batches:.4f} val_loss={val_loss:.4f} val_r={val_r:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model

def main():
    t0 = time.time()
    print("="*60)
    print("  Priority 2: Contrastive Pre-training")
    print("="*60)
    
    # Load prediction cache (faster than running engine)
    with open(ODT_FILE, 'r') as f:
        cache = json.load(f)
    pred_cache = cache.get('pred_138d', {})
    odt_cache = cache.get('odt', {})
    
    # Also load the bigger fragrance DB predictions if available
    # Generate pairs from ALL available 138d predictions
    print(f"\n[1] Loading prediction DB...")
    
    # Try to augment with more molecules from the engine
    try:
        from pom_engine import POMEngine
        engine = POMEngine()
        engine.load()
        
        # Add all fragrance DB molecules
        added = 0
        for name, entry in engine._fragrance_db.items():
            smi = entry.get('smiles', '')
            if smi and smi not in pred_cache.values():
                try:
                    pred = engine.predict_138d(smi)
                    if pred is not None and np.any(pred > 0):
                        key = f"fdb_{name}"
                        pred_cache[key] = pred.tolist()
                        odt_log = entry.get('odt_log')
                        odt_cache[key] = odt_log if odt_log is not None else -3.0
                        added += 1
                except:
                    pass
        print(f"  Added {added} molecules from fragrance DB")
    except Exception as e:
        print(f"  Engine load skipped: {e}")
    
    print(f"  Total molecules for pre-training: {len(pred_cache)}")
    
    # Generate pairs
    print(f"\n[2] Generating contrastive pairs...")
    n_pairs = min(20000, len(pred_cache) * (len(pred_cache) - 1) // 2)
    pairs = generate_pairs_from_cache(pred_cache, odt_cache, n_pairs=n_pairs)
    print(f"  Generated: {len(pairs)} pairs")
    
    if len(pairs) < 100:
        print("  [FAIL] Not enough pairs for pre-training")
        return
    
    # Build model
    print(f"\n[3] Pre-training MixtureEncoder...")
    model = SiameseMixtureNet(input_dim=138, hidden_dim=128, output_dim=64)
    model = pretrain_encoder(model, pairs, epochs=80, lr=0.003, batch_size=128)
    
    # Save pre-trained weights
    os.makedirs(os.path.dirname(PRETRAINED_PATH), exist_ok=True)
    torch.save(model.encoder.state_dict(), PRETRAINED_PATH)
    
    elapsed = time.time() - t0
    print(f"\n[DONE] Pre-trained encoder saved to {PRETRAINED_PATH}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Pairs used: {len(pairs)}")

if __name__ == '__main__':
    main()
