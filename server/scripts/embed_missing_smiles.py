"""embed_missing_smiles.py — 누락된 SMILES의 ChemBERTa 임베딩 생성

unified_training_data.csv에서 chemberta_cache.npz에 없는 SMILES를 찾아
DeepChem/ChemBERTa-77M-MTR로 임베딩 → 캐시에 추가
"""
import os, sys, csv, time
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
BATCH_SIZE = 64


def main():
    t0 = time.time()
    print("=" * 60)
    print("  Embedding Missing SMILES")
    print("=" * 60)

    # 1. Load existing cache
    cache_path = os.path.join(WEIGHTS_DIR, 'chemberta_cache.npz')
    cache_data = np.load(cache_path, allow_pickle=True)
    existing_smiles = set(str(s) for s in cache_data['smiles'])
    existing_emb = cache_data['embeddings']
    hidden_size = int(cache_data['hidden_size'])
    print(f"  Existing cache: {len(existing_smiles):,} SMILES × {hidden_size}d")

    # 2. Find missing SMILES from unified CSV
    csv_path = os.path.join(DATA_DIR, 'unified_training_data.csv')
    missing = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row['smiles'].strip()
            if smi and smi not in existing_smiles:
                missing.append(smi)

    missing = list(set(missing))  # Deduplicate
    print(f"  Missing SMILES: {len(missing)}")

    if len(missing) == 0:
        print("  Nothing to embed!")
        return

    # 3. Load ChemBERTa model
    print(f"\n  Loading {MODEL_NAME}...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    actual_hidden = model.config.hidden_size
    print(f"  Model hidden size: {actual_hidden}")
    
    if actual_hidden != hidden_size:
        print(f"  WARNING: Model is {actual_hidden}d but cache is {hidden_size}d!")
        print(f"  Will project {actual_hidden}d → {hidden_size}d")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Device: {device}")

    # 4. Embed in batches
    new_embeddings = []
    valid_smiles = []
    errors = 0

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i+BATCH_SIZE]
        try:
            tokens = tokenizer(batch, return_tensors='pt', padding=True,
                             truncation=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            with torch.no_grad():
                outputs = model(**tokens)
                # Mean pooling
                emb = outputs.last_hidden_state.mean(dim=1)  # [B, hidden]
                
                # Project if needed
                if actual_hidden != hidden_size:
                    emb = emb[:, :hidden_size]  # Simple truncation
                
                new_embeddings.append(emb.cpu().numpy())
                valid_smiles.extend(batch)
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error at batch {i}: {e}")

        if (i // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+len(batch)}/{len(missing)} embedded ({elapsed:.0f}s)")

    if not new_embeddings:
        print("  No embeddings generated!")
        return

    new_emb_arr = np.concatenate(new_embeddings, axis=0).astype(np.float32)
    print(f"\n  New embeddings: {new_emb_arr.shape}")

    # 5. Merge with existing cache
    all_smiles = list(cache_data['smiles']) + valid_smiles
    all_emb = np.concatenate([existing_emb, new_emb_arr], axis=0)
    
    print(f"  Combined cache: {len(all_smiles):,} SMILES × {all_emb.shape[1]}d")

    # 6. Save updated cache
    backup_path = cache_path + '.bak'
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(cache_path, backup_path)
        print(f"  Backup: {backup_path}")

    np.savez_compressed(cache_path,
                       smiles=np.array(all_smiles, dtype=object),
                       embeddings=all_emb.astype(np.float32),
                       hidden_size=np.array(hidden_size),
                       model_name=np.array(MODEL_NAME))

    elapsed = time.time() - t0
    print(f"\n  DONE! {len(valid_smiles)} new + {len(existing_smiles)} existing = {len(all_smiles):,} total")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved: {cache_path}")


if __name__ == '__main__':
    main()
