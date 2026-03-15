# neural_net.py — Level 2: BatchNorm + GPU 학습
import torch
import torch.nn as nn
import numpy as np


class PerfumeNet(nn.Module):
    def __init__(self, input_dim=24, output_dim=80):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


class NeuralNet:
    MOODS = ['romantic','fresh','elegant','sexy','mysterious','cheerful','calm','luxurious','natural','cozy']
    SEASONS = ['spring','summer','fall','winter']
    CATS = ['floral','citrus','woody','spicy','fruity','gourmand','aquatic','amber','musk','aromatic']

    def __init__(self, db, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.db = db
        self.model = None
        self.ids = []
        self.trained = False

    def build_model(self):
        self.ids = [i['id'] for i in self.db]
        self.model = PerfumeNet(24, len(self.ids)).to(self.device)

    def encode(self, mood, season, prefs):
        return ([1 if m==mood else 0 for m in self.MOODS]
               +[1 if s==season else 0 for s in self.SEASONS]
               +[1 if c in prefs else 0 for c in self.CATS])

    def train(self, formulator_fn, epochs=30, on_progress=None):
        if not self.model: self.build_model()
        ins, outs = [], []
        for _ in range(800):
            m = np.random.choice(self.MOODS)
            s = np.random.choice(self.SEASONS)
            p = [c for c in self.CATS if np.random.random()>0.7]
            f = formulator_fn(m, s, p)
            ins.append(self.encode(m, s, p))
            v = [0.0]*len(self.ids)
            for it in f:
                idx = self.ids.index(it['id']) if it['id'] in self.ids else -1
                if idx>=0: v[idx] = it['percentage']/100
            outs.append(v)
        X = torch.tensor(ins, dtype=torch.float32).to(self.device)
        Y = torch.tensor(outs, dtype=torch.float32).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        crit = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(X, Y)
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
        self.model.train()
        for ep in range(epochs):
            tl = 0
            for bx,by in dl:
                opt.zero_grad(); out=self.model(bx); loss=crit(out,by); loss.backward(); opt.step(); tl+=loss.item()
            if on_progress: on_progress(ep+1, epochs, tl/len(dl))
        self.trained = True

    @torch.no_grad()
    def predict(self, mood, season, prefs):
        if not self.trained: return None
        self.model.eval()
        X = torch.tensor([self.encode(mood,season,prefs)], dtype=torch.float32).to(self.device)
        out = self.model(X).squeeze().cpu().numpy()
        items = [(self.ids[i],float(v)) for i,v in enumerate(out) if v>0.02]
        items.sort(key=lambda x:x[1], reverse=True)
        sel = items[:10]
        tot = sum(v for _,v in sel) or 1
        fm = [{'id':id,'percentage':round(v/tot*100,1)} for id,v in sel]
        adj = 100-sum(f['percentage'] for f in fm)
        if fm: fm[0]['percentage'] = round(fm[0]['percentage']+adj, 1)
        return fm
