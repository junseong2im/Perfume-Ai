# -*- coding: utf-8 -*-
"""
ChemBERTa LoRA Fine-tuning for Odor Prediction
===============================================
ChemBERTa-77M-MTR 모델에 LoRA (Low-Rank Adaptation) 어댑터를 적용하여
냄새 예측 태스크에 특화된 임베딩을 생성.

LoRA rank=4~8 → 학습 파라미터 ~50K 추가 → 5.7K 데이터에서도 안전.

Usage:
    python train_chemberta_lora.py --epochs 20 --rank 4
    python train_chemberta_lora.py --epochs 30 --rank 8 --lr 2e-5

After training:
    python precompute_bert.py --lora weights/chemberta_lora
"""

import sys
import os
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import database as db

# Constants
MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
WEIGHTS_DIR = Path(__file__).parent.parent / 'weights'
LORA_SAVE_DIR = WEIGHTS_DIR / 'chemberta_lora'

ODOR_DIMENSIONS = [
    'sweet', 'sour', 'woody', 'floral', 'citrus',
    'spicy', 'musk', 'fresh', 'green', 'warm',
    'fruity', 'smoky', 'powdery', 'aquatic', 'herbal',
    'amber', 'leather', 'earthy', 'ozonic', 'metallic',
]
N_DIM = len(ODOR_DIMENSIONS)


# ================================================================
# Descriptor → 20d mapping (import from train_models)
# ================================================================

from train_models import _descriptor_to_20d_target, _get_scaffold


# ================================================================
# Dataset for LoRA fine-tuning
# ================================================================

class ChemBERTaOdorDataset(Dataset):
    """SMILES + 20d odor labels for ChemBERTa fine-tuning"""
    
    def __init__(self, molecules, tokenizer, max_length=128, n_augment=0):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        skipped = 0
        
        # SMILES augmentation
        do_augment = n_augment > 0
        if do_augment:
            try:
                from smiles_augment import randomize_smiles
            except ImportError:
                do_augment = False
        
        for mol in molecules:
            smiles = mol.get('smiles', '')
            labels = mol.get('odor_labels', [])
            
            if not smiles or not labels or labels == ['odorless']:
                skipped += 1
                continue
            
            target = _descriptor_to_20d_target(labels)
            if target.max() == 0:
                skipped += 1
                continue
            
            self.data.append((smiles, target))
            
            # SMILES augmentation
            if do_augment:
                variants = randomize_smiles(smiles, n_augment=n_augment)
                for v in variants:
                    if v != smiles:
                        self.data.append((v, target))
        
        print(f"  [LoRA Dataset] {len(self.data)} samples (skipped={skipped})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles, target = self.data[idx]
        encoding = self.tokenizer(
            smiles,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(target, dtype=torch.float32),
        }


# ================================================================
# Scaffold Split for LoRA dataset
# ================================================================

def scaffold_split_lora(dataset, val_ratio=0.2, seed=42):
    """Scaffold split for ChemBERTaOdorDataset"""
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    
    scaffold_to_indices = defaultdict(list)
    for i in range(len(dataset.data)):
        smiles = dataset.data[i][0]
        scaffold = _get_scaffold(smiles)
        scaffold_to_indices[scaffold].append(i)
    
    scaffolds = list(scaffold_to_indices.values())
    rng.shuffle(scaffolds)
    scaffolds.sort(key=len, reverse=True)
    
    n_val_target = int(len(dataset) * val_ratio)
    train_idx, val_idx = [], []
    
    for group in scaffolds:
        if len(val_idx) < n_val_target:
            val_idx.extend(group)
        else:
            train_idx.extend(group)
    
    print(f"  [Scaffold Split] {len(scaffold_to_indices)} unique scaffolds")
    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    return train_set, val_set


# ================================================================
# LoRA fine-tuning head
# ================================================================

class OdorPredictionHead(nn.Module):
    """[CLS] embedding → 20d odor vector"""
    def __init__(self, hidden_size=384):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, N_DIM),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.head(x)


# ================================================================
# ASL Loss (from train_models.py)
# ================================================================

class AsymmetricLoss(nn.Module):
    """ASL for multi-label classification"""
    def __init__(self, gamma_pos=1, gamma_neg=4, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
    
    def forward(self, preds, targets):
        eps = 1e-7
        p = preds.clamp(eps, 1 - eps)
        pos_loss = targets * ((1 - p) ** self.gamma_pos) * torch.log(p)
        p_neg = (p - self.clip).clamp(min=0)
        neg_loss = (1 - targets) * (p_neg ** self.gamma_neg) * torch.log(1 - p)
        loss = -(pos_loss + neg_loss)
        return loss.mean()


# ================================================================
# Training
# ================================================================

def train_lora(
    epochs=20,
    rank=4,
    alpha=16,
    lr=2e-5,
    batch_size=16,
    n_augment=5,
    seed=42,
):
    """ChemBERTa LoRA 파인튜닝
    
    Args:
        epochs: 학습 에폭 수
        rank: LoRA rank (4~8 권장)
        alpha: LoRA alpha (보통 rank * 4)
        lr: 학습률
        batch_size: 배치 크기
        n_augment: SMILES 증강 수 (5~10 권장)
        seed: 랜덤 시드
    """
    from transformers import AutoTokenizer, AutoModel
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print(f"  ChemBERTa LoRA Fine-tuning")
    print(f"  Model: {MODEL_NAME}")
    print(f"  LoRA: rank={rank}, alpha={alpha}")
    print(f"  Device: {device}")
    print(f"  Augmentation: {n_augment}x")
    print("=" * 60)
    
    # 1. Data
    print("\n  Loading data from DB...")
    molecules = db.get_all_molecules()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    dataset = ChemBERTaOdorDataset(
        molecules, tokenizer,
        n_augment=n_augment
    )
    
    train_set, val_set = scaffold_split_lora(dataset, val_ratio=0.2, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    
    # 2. Model + LoRA
    print(f"\n  Loading {MODEL_NAME}...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.1,
            target_modules=["query", "value"],  # attention Q, V만 적용
            bias="none",
        )
        
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
    except ImportError:
        print("  ERROR: peft not installed!")
        print("  Install with: pip install peft>=0.7.0")
        return None
    
    model = model.to(device)
    
    # Prediction head
    hidden_size = base_model.config.hidden_size
    pred_head = OdorPredictionHead(hidden_size).to(device)
    
    # 3. Optimizer (LoRA params + head params)
    all_params = list(model.parameters()) + list(pred_head.parameters())
    trainable_params = [p for p in all_params if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    
    # Cosine scheduler with warmup
    warmup_steps = max(5, epochs // 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return (epoch + 1) / warmup_steps
        progress = (epoch - warmup_steps) / max(1, epochs - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = AsymmetricLoss(gamma_pos=1, gamma_neg=4, clip=0.05)
    
    n_total = sum(p.numel() for p in trainable_params)
    print(f"\n  Trainable parameters: {n_total:,}")
    print(f"  Data: train={len(train_set)}, val={len(val_set)}")
    
    # 4. Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        pred_head.train()
        train_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, hidden]
            preds = pred_head(cls_emb)  # [B, 20]
            
            loss = criterion(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= max(1, n_batches)
        
        # --- Validate ---
        model.eval()
        pred_head.eval()
        val_loss = 0
        val_cos_sim = 0
        n_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_emb = outputs.last_hidden_state[:, 0, :]
                preds = pred_head(cls_emb)
                
                val_loss += criterion(preds, labels).item()
                
                # Cosine similarity
                cos = nn.functional.cosine_similarity(preds, labels, dim=1)
                val_cos_sim += cos.sum().item()
                n_val += labels.size(0)
        
        val_loss /= max(1, len(val_loader))
        val_cos_sim /= max(1, n_val)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save LoRA adapters
            LORA_SAVE_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(LORA_SAVE_DIR))
            torch.save(pred_head.state_dict(), LORA_SAVE_DIR / 'pred_head.pt')
            
            # Save training info
            info = {
                'epoch': epoch,
                'val_loss': val_loss,
                'val_cos_sim': val_cos_sim,
                'rank': rank,
                'alpha': alpha,
                'lr': lr,
                'n_train': len(train_set),
                'n_val': len(val_set),
                'n_augment': n_augment,
                'trainable_params': n_total,
            }
            with open(LORA_SAVE_DIR / 'training_info.json', 'w') as f:
                json.dump(info, f, indent=2)
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"CosSim: {val_cos_sim:.3f} | Best: {best_val_loss:.4f} | "
                  f"{elapsed:.1f}s")
        
        if patience_counter >= 15:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    print(f"\n  {'='*40}")
    print(f"  ✅ LoRA fine-tuning complete!")
    print(f"  Best epoch: {best_epoch+1} | Val loss: {best_val_loss:.4f}")
    print(f"  CosSim: {val_cos_sim:.3f}")
    print(f"  Saved to: {LORA_SAVE_DIR}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  {'='*40}")
    print(f"\n  Next step: python precompute_bert.py --lora {LORA_SAVE_DIR}")
    
    return model, pred_head


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChemBERTa LoRA Fine-tuning')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--rank', type=int, default=4,
                        help='LoRA rank (4~8 recommended)')
    parser.add_argument('--alpha', type=int, default=16,
                        help='LoRA alpha (usually rank * 4)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--augment', type=int, default=5,
                        help='SMILES augmentation factor')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    train_lora(
        epochs=args.epochs,
        rank=args.rank,
        alpha=args.alpha,
        lr=args.lr,
        batch_size=args.batch_size,
        n_augment=args.augment,
        seed=args.seed,
    )
