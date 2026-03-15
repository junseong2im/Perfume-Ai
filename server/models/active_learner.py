# active_learner.py — Pillar 4: nn.Embedding + nn.GRU (PyTorch CUDA)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChemLangModel(nn.Module):
    """nn.Embedding → nn.GRU → 냄새 분류"""
    def __init__(self, vocab_size, embed_dim=32, gru_hidden=64, num_labels=20):
        super().__init__()
        # ★ nn.Embedding — 네이티브 임베딩 (O(1) lookup, GPU 가속)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0  # <PAD> = 0 마스킹 (build_vocab guarantees <PAD>=0)
        )

        # ★ nn.GRU — cuDNN 가속 시퀀스 인코딩
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=2,         # 2층 GRU
            batch_first=True,
            dropout=0.1,
            bidirectional=True    # 양방향 GRU
        )

        self.odor_embed_layer = nn.Linear(gru_hidden * 2, embed_dim)  # 양방향 → embed_dim
        self.classifier = nn.Sequential(
            nn.Tanh(),
            nn.Linear(embed_dim, num_labels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, SeqLen] (int64 정수 시퀀스)
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()  # actual sequence lengths
        emb = self.embedding(x)         # [B, SeqLen, EmbedDim]
        # Pack padded sequences so GRU doesn't process padding tokens
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        output, hidden = self.gru(packed)  # hidden: [4, B, GRU_H] (2 layers × 2 dirs)
        # 마지막 hidden state (양방향 concat)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [B, 2*GRU_H]
        odor_emb = self.odor_embed_layer(h)             # [B, embed_dim]
        return self.classifier(odor_emb)                # [B, num_labels]

    def get_embedding(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        output, hidden = self.gru(packed)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.odor_embed_layer(h)  # [B, embed_dim]


class UncertaintyModel(nn.Module):
    """MC Dropout + Embedding"""
    def __init__(self, vocab_size, embed_dim=16, num_labels=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 48), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, num_labels), nn.Sigmoid()
        )

    def forward(self, x):
        emb = self.embedding(x)              # [B, SeqLen, EmbedDim]
        # Masked mean: only average over non-padding positions
        mask = (x != 0).unsqueeze(-1).float()  # [B, SeqLen, 1]
        emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, EmbedDim]
        return self.net(emb)


class ActiveLearner:
    """능동 학습 + GRU CLM + 임베딩 Zero-shot"""

    ODOR_LABELS = [
        'floral', 'citrus', 'woody', 'spicy', 'sweet', 'fresh',
        'green', 'warm', 'musk', 'fruity', 'rose', 'jasmine',
        'cedar', 'vanilla', 'amber', 'clean', 'smoky', 'powdery',
        'aquatic', 'herbal'
    ]

    TEXT_SEEDS = {
        '장미': ['floral', 'rose'], '재스민': ['floral', 'jasmine'],
        '레몬': ['citrus', 'fresh'], '오렌지': ['citrus', 'fruity'],
        '백단향': ['woody', 'warm'], '시더': ['woody', 'cedar'],
        '바닐라': ['sweet', 'vanilla'], '꿀': ['sweet', 'warm'],
        '라벤더': ['herbal', 'fresh'], '민트': ['fresh', 'green'],
        '계피': ['spicy', 'warm'], '후추': ['spicy'],
        '바다': ['aquatic', 'fresh'], '이끼': ['green', 'woody'],
        '머스크': ['musk', 'powdery'], '앰버': ['amber', 'warm'],
        '연기': ['smoky', 'woody'], '비누': ['clean', 'fresh'],
        '복숭아': ['fruity', 'sweet'], '사과': ['fruity', 'green'],
        '숲': ['green', 'woody', 'fresh'], '꽃': ['floral'],
        '향신료': ['spicy'], '과일': ['fruity'], '나무': ['woody'],
        '달콤': ['sweet'], '신선': ['fresh'], '깨끗': ['clean'],
        '따뜻': ['warm'], '부드러운': ['powdery', 'musk'],
    }

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.token_map = {}
        self.vocab_size = 0
        self.max_seq_len = 64
        self.embed_dim = 32
        self.clm = None
        self.unc_model = None
        self.odor_embeddings = {}
        self.trained = False

    def build_vocab(self, molecules):
        # ★ Fixed: Use deterministic list ordering so <PAD>=0 matches padding_idx=0
        special = ['<PAD>', '<START>', '<END>', '<UNK>']  # indices 0,1,2,3
        char_tokens = set()
        for mol in molecules:
            for t in self._tokenize(mol.get('smiles', '')):
                char_tokens.add(t)
        all_tokens = special + sorted(char_tokens - set(special))
        self.token_map = {t: i for i, t in enumerate(all_tokens)}
        self.vocab_size = len(self.token_map)

    def _tokenize(self, smiles):
        tokens = []
        i = 0
        while i < len(smiles):
            if smiles[i] == '[':
                close = smiles.find(']', i)
                if close > i:
                    tokens.append(smiles[i:close + 1])
                    i = close + 1; continue
            if i + 1 < len(smiles) and smiles[i:i + 2] in ('Cl', 'Br', 'Si', 'Na', 'Li', 'Se', 'Te'):
                tokens.append(smiles[i:i + 2])
                i += 2; continue
            tokens.append(smiles[i])
            i += 1
        return tokens

    def encode_smiles(self, smiles):
        tokens = self._tokenize(smiles)
        seq = [self.token_map.get('<START>', 1)]
        for t in tokens:
            seq.append(self.token_map.get(t, self.token_map.get('<UNK>', 3)))
        seq.append(self.token_map.get('<END>', 2))
        while len(seq) < self.max_seq_len:
            seq.append(0)
        return seq[:self.max_seq_len]

    def train(self, molecular_engine, epochs=50, on_progress=None):
        molecules = molecular_engine.get_all() if hasattr(molecular_engine, 'get_all') else molecular_engine
        if not molecules:
            return

        self.build_vocab(molecules)

        self.clm = ChemLangModel(
            self.vocab_size, self.embed_dim, 64, len(self.ODOR_LABELS)
        ).to(self.device)

        self.unc_model = UncertaintyModel(
            self.vocab_size, 16, len(self.ODOR_LABELS)
        ).to(self.device)

        # 데이터
        inputs, targets = [], []
        for mol in molecules:
            inputs.append(self.encode_smiles(mol.get('smiles', '')))
            targets.append([1.0 if l in mol.get('odor_labels', []) else 0.0 for l in self.ODOR_LABELS])

        # 증강 5x
        aug_in, aug_tgt = list(inputs), list(targets)
        for _ in range(5):
            for i in range(len(inputs)):
                noisy = [0 if np.random.random() > 0.9 else v for v in inputs[i]]
                aug_in.append(noisy)
                aug_tgt.append(targets[i])

        X = torch.tensor(aug_in, dtype=torch.long).to(self.device)
        Y = torch.tensor(aug_tgt, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        # CLM 학습
        opt = torch.optim.Adam(self.clm.parameters(), lr=0.001)
        crit = nn.BCELoss()
        clm_epochs = int(epochs * 0.6)

        self.clm.train()
        for ep in range(clm_epochs):
            total_loss = 0
            for bx, by in loader:
                opt.zero_grad()
                out = self.clm(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if on_progress:
                on_progress(ep + 1, f"CLM: loss={total_loss / len(loader):.4f}")

        # Uncertainty 학습
        opt2 = torch.optim.Adam(self.unc_model.parameters(), lr=0.001)
        unc_epochs = int(epochs * 0.4)

        self.unc_model.train()
        for ep in range(unc_epochs):
            total_loss = 0
            for bx, by in loader:
                opt2.zero_grad()
                out = self.unc_model(bx)
                loss = crit(out, by)
                loss.backward()
                opt2.step()
                total_loss += loss.item()
            if on_progress:
                on_progress(ep + 1, f"Uncertainty: loss={total_loss / len(loader):.4f}")

        self._extract_embeddings(molecules)
        self.trained = True

    @torch.no_grad()
    def _extract_embeddings(self, molecules):
        self.clm.eval()
        for label in self.ODOR_LABELS:
            mols = [m for m in molecules if label in m.get('odor_labels', [])]
            if not mols:
                self.odor_embeddings[label] = [0.0] * self.embed_dim
                continue
            seqs = [self.encode_smiles(m.get('smiles', '')) for m in mols]
            X = torch.tensor(seqs, dtype=torch.long).to(self.device)
            embs = self.clm.get_embedding(X)
            self.odor_embeddings[label] = embs.mean(dim=0).cpu().tolist()

    @torch.no_grad()
    def estimate_uncertainty(self, smiles, num_samples=20):
        if self.unc_model is None:
            return None
        seq = self.encode_smiles(smiles)
        X = torch.tensor([seq], dtype=torch.long).to(self.device)

        self.unc_model.train()  # MC Dropout 활성화
        preds = []
        for _ in range(num_samples):
            out = self.unc_model(X).cpu().numpy()[0]
            preds.append(out)

        preds = np.array(preds)
        mean = preds.mean(axis=0)
        var = preds.var(axis=0)
        total_unc = float(var.mean())

        return {
            'predictions': [
                {'label': l, 'mean': float(mean[i]), 'variance': float(var[i]),
                 'confidence': 1 - float(np.sqrt(var[i]))}
                for i, l in enumerate(self.ODOR_LABELS) if mean[i] > 0.2
            ],
            'totalUncertainty': total_unc,
            'isHighUncertainty': total_unc > 0.1
        }

    def predict_from_text(self, text):
        seed_labels = set()
        for kw, odors in self.TEXT_SEEDS.items():
            if kw in text:
                seed_labels.update(odors)

        if not seed_labels or not self.odor_embeddings:
            return self._fallback(text, seed_labels)

        query = np.zeros(self.embed_dim)
        for label in seed_labels:
            emb = self.odor_embeddings.get(label, [0] * self.embed_dim)
            query += np.array(emb)
        query /= (np.linalg.norm(query) + 1e-8)

        results = []
        for label in self.ODOR_LABELS:
            emb = np.array(self.odor_embeddings.get(label, [0] * self.embed_dim))
            sim = float(np.dot(query, emb) / (np.linalg.norm(emb) + 1e-8))
            bonus = 0.2 if label in seed_labels else 0
            results.append({'label': label, 'score': max(0, min(1, (sim + 1) / 2 + bonus)), 'isDirectMatch': label in seed_labels})

        results.sort(key=lambda x: x['score'], reverse=True)
        return {'query': text, 'predictions': results[:10], 'method': 'embedding_cosine', 'confidence': results[0]['score'] if results else 0}

    def _fallback(self, text, seed_labels):
        return {'query': text, 'predictions': [{'label': l, 'score': 1.0, 'isDirectMatch': True} for l in seed_labels], 'method': 'keyword_fallback', 'confidence': 0.5}

    def suggest_next_samples(self, candidates, top_k=5):
        if not self.trained:
            return candidates[:top_k]
        scored = []
        for mol in candidates:
            unc = self.estimate_uncertainty(mol.get('smiles', ''))
            scored.append({**mol, 'uncertainty': unc['totalUncertainty'] if unc else 0})
        scored.sort(key=lambda x: x['uncertainty'], reverse=True)
        return scored[:top_k]
