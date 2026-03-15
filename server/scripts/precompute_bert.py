# precompute_bert.py - ChemBERTa 임베딩 사전 계산 (Freeze & Cache)
# ==================================================================
# DB의 모든 SMILES를 ChemBERTa에 통과시켜 384d 벡터를 추출하고
# weights/chemberta_cache.npz에 저장. 런타임에는 이 파일만 로드.
#
# v2: SMILES Randomization 증강 + LoRA 파인튜닝 모델 지원
#     --augment 10  → 각 SMILES당 10개 랜덤 변형 생성
#     --lora PATH   → LoRA 어댑터 로드 후 임베딩 추출
# ==================================================================

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import database as db

# ChemBERTa-77M-MTR: PubChem 77M 분자 학습, 384d, multi-task regression
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
CACHE_PATH = Path(__file__).parent.parent / 'weights' / 'chemberta_cache.npz'
BATCH_SIZE = 64


def extract_embeddings(n_augment=0, lora_path=None):
    """ChemBERTa 임베딩 추출 (+ SMILES 증강 + LoRA 지원)
    
    Args:
        n_augment: SMILES 랜덤화 변형 수 (0이면 비활성화)
        lora_path: LoRA 어댑터 경로 (None이면 기본 모델)
    """
    from transformers import AutoTokenizer, AutoModel

    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aug_label = f" + Augment×{n_augment}" if n_augment > 0 else ""
    lora_label = f" + LoRA({lora_path})" if lora_path else ""
    print("=" * 60)
    print(f"  ChemBERTa Embedding Extraction{aug_label}{lora_label}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {device}")
    print("=" * 60)

    # 1. DB에서 SMILES 로드
    print("\n  Loading molecules from DB...")
    molecules = db.get_all_molecules()
    original_smiles = [m['smiles'] for m in molecules if m.get('smiles')]
    print(f"  Loaded {len(original_smiles)} original SMILES")

    # 1b. SMILES 증강 (랜덤화)
    if n_augment > 0:
        from smiles_augment import randomize_smiles
        smiles_list = []
        for smi in original_smiles:
            variants = randomize_smiles(smi, n_augment=n_augment)
            smiles_list.extend(variants)
        # 중복 제거 (순서 보존)
        seen = set()
        unique_smiles = []
        for s in smiles_list:
            if s not in seen:
                seen.add(s)
                unique_smiles.append(s)
        smiles_list = unique_smiles
        print(f"  Augmented: {len(original_smiles)} → {len(smiles_list)} "
              f"unique SMILES ({len(smiles_list)/len(original_smiles):.1f}x)")
    else:
        smiles_list = original_smiles

    # 2. 모델 로드 (LoRA 지원)
    print(f"\n  Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
    if lora_path and Path(lora_path).exists():
        try:
            from peft import PeftModel
            print(f"  Loading LoRA adapters from {lora_path}...")
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()  # LoRA를 기본 모델에 병합
            print(f"  LoRA adapters merged successfully")
        except ImportError:
            print(f"  WARNING: peft not installed, using base model")
        except Exception as e:
            print(f"  WARNING: LoRA load failed ({e}), using base model")
    
    model = model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    print(f"  Hidden size: {hidden_size}d")

    # 3. 배치 처리로 임베딩 추출
    all_embeddings = []
    all_smiles = []
    errors = 0

    print(f"\n  Extracting embeddings (batch={BATCH_SIZE}, total={len(smiles_list)})...")

    with torch.no_grad():
        for i in range(0, len(smiles_list), BATCH_SIZE):
            batch_smiles = smiles_list[i:i + BATCH_SIZE]

            try:
                inputs = tokenizer(
                    batch_smiles,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                ).to(device)

                outputs = model(**inputs)
                # [CLS] token embedding = 분자의 전체 문맥 표현
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                all_embeddings.append(cls_embeddings)
                all_smiles.extend(batch_smiles)
            except Exception as e:
                errors += len(batch_smiles)
                print(f"    Error at batch {i}: {e}")

            if (i // BATCH_SIZE + 1) % 20 == 0:
                elapsed = time.time() - start
                print(f"    {i + len(batch_smiles)}/{len(smiles_list)} "
                      f"({elapsed:.1f}s)")

    # 4. 저장
    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # SMILES → index 매핑을 위해 SMILES 배열도 저장
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        embeddings=embeddings,
        smiles=np.array(all_smiles),
        model_name=np.array(MODEL_NAME),
        hidden_size=np.array(hidden_size),
    )

    elapsed = time.time() - start
    file_size = CACHE_PATH.stat().st_size / (1024 * 1024)

    print(f"\n  {'=' * 40}")
    print(f"  Done!")
    print(f"  Embeddings: {embeddings.shape} ({hidden_size}d)")
    print(f"  SMILES cached: {len(all_smiles)}")
    if n_augment > 0:
        print(f"  Augmentation: {len(original_smiles)} → {len(all_smiles)} ({n_augment}x target)")
    print(f"  Errors: {errors}")
    print(f"  File: {CACHE_PATH} ({file_size:.1f} MB)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  {'=' * 40}")

    return embeddings, all_smiles


class ChemBERTaCache:
    """런타임에서 사용하는 ChemBERTa 캐시 로더

    사용법:
        cache = ChemBERTaCache()
        embedding = cache.get("CCO")  # 384d numpy array
    """

    def __init__(self, cache_path=None):
        self.cache_path = Path(cache_path) if cache_path else CACHE_PATH
        self._embeddings = None
        self._smiles_to_idx = None
        self._hidden_size = 384
        self._loaded = False

    def load(self):
        if self._loaded:
            return True

        if not self.cache_path.exists():
            print(f"[ChemBERTaCache] Cache not found: {self.cache_path}")
            return False

        data = np.load(self.cache_path, allow_pickle=True)
        self._embeddings = data['embeddings']
        smiles_arr = data['smiles']
        self._hidden_size = int(data['hidden_size'])

        self._smiles_to_idx = {s: i for i, s in enumerate(smiles_arr)}
        self._loaded = True

        print(f"[ChemBERTaCache] Loaded {len(self._smiles_to_idx)} embeddings "
              f"({self._hidden_size}d) from {self.cache_path.name}")
        return True

    def get(self, smiles):
        """SMILES -> embedding (384d numpy array) or None"""
        if not self._loaded:
            self.load()
        if not self._loaded or smiles not in self._smiles_to_idx:
            return None
        idx = self._smiles_to_idx[smiles]
        return self._embeddings[idx]

    def get_batch(self, smiles_list):
        """여러 SMILES → (embeddings, hit_mask)"""
        if not self._loaded:
            self.load()

        results = np.zeros((len(smiles_list), self._hidden_size), dtype=np.float32)
        hits = np.zeros(len(smiles_list), dtype=bool)

        if self._loaded:
            for i, s in enumerate(smiles_list):
                if s in self._smiles_to_idx:
                    results[i] = self._embeddings[self._smiles_to_idx[s]]
                    hits[i] = True

        return results, hits

    @property
    def hidden_size(self):
        if not self._loaded:
            self.load()
        return self._hidden_size

    @property
    def size(self):
        if not self._loaded:
            return 0
        return len(self._smiles_to_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChemBERTa embedding extraction')
    parser.add_argument('--augment', type=int, default=0,
                        help='SMILES randomization augmentation factor (0=off, 10=recommended)')
    parser.add_argument('--lora', type=str, default=None,
                        help='Path to LoRA adapter directory')
    args = parser.parse_args()
    extract_embeddings(n_augment=args.augment, lora_path=args.lora)
