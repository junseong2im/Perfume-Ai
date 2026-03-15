"""
Odor-Text Contrastive Learning — CLIP 스타일 대조 학습
=======================================================
분자의 화학 구조(OdorGNN 벡터)와 냄새 텍스트 묘사(Sentence-BERT)를
의미 공간에서 정렬하여 희귀 노트(ozonic, metallic 등) 예측 보정.

사용법:
    from scripts.contrastive_learner import OdorTextContrastiveLearner
    learner = OdorTextContrastiveLearner()
    learner.build_dataset()
    learner.train(epochs=20)
    learner.save_projection_head('models/odor_text_projection.pt')
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ProjectionHead(nn.Module):
    """OdorGNN 20d → 384d 투영 (Sentence-BERT 공간으로)"""
    def __init__(self, input_dim=20, output_dim=384, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TextEncoder:
    """Sentence-BERT로 텍스트 → 384d 임베딩"""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = None
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _ensure_loaded(self):
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"[TextEncoder] Loaded {self.model_name}")
        except ImportError:
            print("[TextEncoder] sentence-transformers not installed, using fallback")
            self.model = None
    
    def encode(self, texts):
        """텍스트 리스트 → (N, 384) 텐서"""
        self._ensure_loaded()
        if self.model is not None:
            with torch.no_grad():
                embeddings = self.model.encode(texts, convert_to_tensor=True,
                                              normalize_embeddings=True)
            return embeddings.to(self.device)
        else:
            return self._fallback_encode(texts)
    
    def _fallback_encode(self, texts):
        """Sentence-BERT 없을 때 TF-IDF 스타일 fallback"""
        # 간단한 해싱 기반 임베딩
        embeddings = []
        for text in texts:
            words = text.lower().split()
            vec = np.zeros(384)
            for i, w in enumerate(words):
                h = hash(w) % 384
                vec[h] += 1.0 / (i + 1)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
        return torch.tensor(np.array(embeddings), dtype=torch.float32, device=self.device)


def labels_to_text(odor_labels):
    """냄새 descriptor 리스트 → 자연어 문장
    
    ['sweet', 'floral', 'vanilla'] → "This molecule smells sweet, floral, with vanilla notes"
    """
    if not odor_labels or odor_labels == ['{}']:
        return "This molecule has no distinctive odor"
    
    labels = [l.strip() for l in odor_labels if l and l.strip() and l != '{}']
    if not labels:
        return "This molecule has no distinctive odor"
    
    if len(labels) == 1:
        return f"This molecule smells {labels[0]}"
    
    primary = labels[0]
    secondary = ', '.join(labels[1:])
    return f"This molecule smells {primary}, with {secondary} notes"


class InfoNCELoss(nn.Module):
    """InfoNCE Contrastive Loss (CLIP-style)"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_chem, z_text):
        """
        Args:
            z_chem: (B, D) 화학 벡터 (normalized)
            z_text: (B, D) 텍스트 벡터 (normalized)
        
        같은 인덱스 = positive pair, 다른 인덱스 = negative
        """
        # 코사인 유사도 행렬
        logits = torch.mm(z_chem, z_text.T) / self.temperature
        labels = torch.arange(len(z_chem), device=z_chem.device)
        
        # 양방향 InfoNCE
        loss_ct = F.cross_entropy(logits, labels)
        loss_tc = F.cross_entropy(logits.T, labels)
        return (loss_ct + loss_tc) / 2


class OdorTextContrastiveLearner:
    """Odor-Text 대조 학습기"""
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.projection = ProjectionHead().to(self.device)
        self.text_encoder = TextEncoder()
        self.criterion = InfoNCELoss(temperature=0.07)
        
        self.chem_vectors = []   # (N, 20) OdorGNN 벡터
        self.text_embeddings = [] # (N, 384) Sentence-BERT 벡터
        self.smiles_list = []
        self.text_list = []
    
    def build_dataset(self, limit=None):
        """DB에서 (분자 냄새 벡터, 텍스트 묘사) 쌍 구축"""
        import database as db
        from odor_engine import OdorGNN
        
        gnn = OdorGNN(device=self.device)
        molecules = db.get_all_molecules(limit=limit)
        
        valid_pairs = []
        for mol in molecules:
            smiles = mol.get('smiles')
            labels = mol.get('odor_labels', [])
            if not smiles or not labels or labels == ['{}']:
                continue
            
            # OdorGNN 벡터
            vec = gnn.encode(smiles)
            if vec is None:
                continue
            
            # 텍스트 묘사 생성
            text = labels_to_text(labels)
            valid_pairs.append((smiles, vec, text))
        
        if not valid_pairs:
            print("[ContrastiveLearner] No valid pairs found")
            return 0
        
        # 텍스트 인코딩 (배치)
        texts = [p[2] for p in valid_pairs]
        text_embs = self.text_encoder.encode(texts)
        
        for i, (smiles, vec, text) in enumerate(valid_pairs):
            self.chem_vectors.append(torch.tensor(vec, dtype=torch.float32))
            self.text_embeddings.append(text_embs[i])
            self.smiles_list.append(smiles)
            self.text_list.append(text)
        
        print(f"[ContrastiveLearner] Built {len(valid_pairs)} pairs")
        return len(valid_pairs)
    
    def train(self, epochs=20, batch_size=64, lr=1e-3):
        """대조 학습 실행"""
        if len(self.chem_vectors) < batch_size:
            print(f"[ContrastiveLearner] Not enough data ({len(self.chem_vectors)})")
            return []
        
        chem_all = torch.stack(self.chem_vectors).to(self.device)
        text_all = torch.stack(self.text_embeddings).to(self.device)
        
        optimizer = torch.optim.AdamW(self.projection.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        n = len(chem_all)
        losses = []
        
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                if len(idx) < 4:
                    continue
                
                chem_batch = chem_all[idx]
                text_batch = text_all[idx]
                
                # 투영
                z_chem = self.projection(chem_batch)
                z_text = F.normalize(text_batch, dim=-1)
                
                loss = self.criterion(z_chem, z_text)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.projection.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        print(f"[ContrastiveLearner] Training done. Final loss: {losses[-1]:.4f}")
        return losses
    
    def save_projection_head(self, path):
        """Projection head 저장"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(self.projection.state_dict(), path)
        print(f"[ContrastiveLearner] Saved to {path}")
    
    def load_projection_head(self, path):
        """Projection head 로드"""
        if os.path.exists(path):
            self.projection.load_state_dict(torch.load(path, map_location=self.device))
            print(f"[ContrastiveLearner] Loaded from {path}")
    
    def get_text_similarity(self, odor_vector, text_query):
        """냄새 벡터와 텍스트 쿼리의 유사도 계산"""
        self.projection.eval()
        with torch.no_grad():
            chem_t = torch.tensor(odor_vector, dtype=torch.float32, device=self.device)
            z_chem = self.projection(chem_t.unsqueeze(0))
            z_text = self.text_encoder.encode([text_query])
            z_text = F.normalize(z_text, dim=-1)
            sim = F.cosine_similarity(z_chem, z_text).item()
        return sim


if __name__ == '__main__':
    print("=== Odor-Text Contrastive Learning ===")
    learner = OdorTextContrastiveLearner()
    n = learner.build_dataset(limit=500)
    if n > 0:
        losses = learner.train(epochs=20, batch_size=32)
        learner.save_projection_head('models/odor_text_projection.pt')
