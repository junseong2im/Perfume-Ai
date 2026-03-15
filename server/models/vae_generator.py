# vae_generator.py — Level 3: β-VAE + KL Divergence (PyTorch CUDA)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim=80, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
                                 nn.Linear(64, 32), nn.ReLU())
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        # Decoder
        self.dec = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.BatchNorm1d(32),
                                 nn.Linear(32, 64), nn.ReLU(),
                                 nn.Linear(64, input_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta=0.5):
    """Reconstruction + β * KL Divergence"""
    recon_loss = F.binary_cross_entropy(recon, x, reduction='sum')
    # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


class VAEGenerator:
    def __init__(self, db, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.db = db
        self.ids = []
        self.vae = None
        self.beta = 0.5
        self.latent_dim = 8
        self.trained = False
        self.last_kl = 0
        self.last_recon = 0

    def build_model(self):
        self.ids = [i['id'] for i in self.db]
        self.vae = VAE(len(self.ids), self.latent_dim).to(self.device)

    def train(self, formulator_fn, epochs=40, on_progress=None):
        if not self.vae: self.build_model()
        data = []
        moods = ['romantic','fresh','elegant','sexy','mysterious','cheerful','calm','luxurious','natural','cozy']
        seasons = ['spring','summer','fall','winter']
        cats = ['floral','citrus','woody','spicy','fruity','gourmand','aquatic','amber','musk','aromatic']
        for _ in range(600):
            m = np.random.choice(moods); s = np.random.choice(seasons)
            p = [c for c in cats if np.random.random()>0.7]
            f = formulator_fn(m, s, p)
            v = [0.0]*len(self.ids)
            for it in f:
                idx = self.ids.index(it['id']) if it['id'] in self.ids else -1
                if idx>=0: v[idx] = it['percentage']/100
            data.append(v)
        X = torch.tensor(data, dtype=torch.float32).to(self.device)
        opt = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        ds = torch.utils.data.TensorDataset(X)
        dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
        self.vae.train()
        for ep in range(epochs):
            tl, tr, tk = 0, 0, 0
            for (bx,) in dl:
                opt.zero_grad()
                recon, mu, logvar = self.vae(bx)
                loss, rl, kl = vae_loss(recon, bx, mu, logvar, self.beta)
                loss.backward(); opt.step()
                tl += loss.item(); tr += rl.item(); tk += kl.item()
            n = len(dl)
            self.last_recon = tr/n; self.last_kl = tk/n
            if on_progress: on_progress(ep+1, epochs, tl/n)
        self.trained = True

    @torch.no_grad()
    def generate_random(self):
        if not self.trained: return None
        self.vae.eval()
        z = torch.randn(1, self.latent_dim).to(self.device)
        out = self.vae.decode(z).squeeze().cpu().numpy()
        return self._decode(out)

    @torch.no_grad()
    def generate_from_latent(self, vec):
        if not self.trained: return None
        self.vae.eval()
        full = [0.0]*self.latent_dim
        for i,v in enumerate(vec[:self.latent_dim]): full[i]=v
        z = torch.tensor([full], dtype=torch.float32).to(self.device)
        out = self.vae.decode(z).squeeze().cpu().numpy()
        return self._decode(out)

    @torch.no_grad()
    def interpolate(self, formula_a, formula_b, steps=5):
        if not self.trained: return []
        self.vae.eval()
        va = self._encode_formula(formula_a)
        vb = self._encode_formula(formula_b)
        xa = torch.tensor([va], dtype=torch.float32).to(self.device)
        xb = torch.tensor([vb], dtype=torch.float32).to(self.device)
        za, _ = self.vae.encode(xa); zb, _ = self.vae.encode(xb)
        results = []
        for i in range(steps+1):
            t = i/steps
            zi = za*(1-t) + zb*t
            out = self.vae.decode(zi).squeeze().cpu().numpy()
            results.append(self._decode(out))
        return results

    @torch.no_grad()
    def encode(self, formula):
        if not self.trained: return None
        self.vae.eval()
        v = self._encode_formula(formula)
        x = torch.tensor([v], dtype=torch.float32).to(self.device)
        mu, _ = self.vae.encode(x)
        return mu.squeeze().cpu().tolist()

    def _encode_formula(self, formula):
        v = [0.0]*len(self.ids)
        for it in formula:
            idx = self.ids.index(it['id']) if it['id'] in self.ids else -1
            if idx>=0: v[idx] = it['percentage']/100
        return v

    def _decode(self, vec):
        items = [(self.ids[i],float(v)) for i,v in enumerate(vec) if v>0.015]
        items.sort(key=lambda x:x[1], reverse=True)
        sel = items[:10]
        tot = sum(v for _,v in sel) or 1
        fm = [{'id':id,'percentage':round(v/tot*100,1)} for id,v in sel]
        adj = 100-sum(f['percentage'] for f in fm)
        if fm: fm[0]['percentage']=round(fm[0]['percentage']+adj,1)
        return fm
