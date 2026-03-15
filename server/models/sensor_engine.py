# sensor_engine.py — Pillar 3: nn.Conv1d 초해상도 (PyTorch CUDA)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SuperResNet(nn.Module):
    """Conv1D 초해상도: [B, 1, 20] → [B, 1, 200]"""
    def __init__(self):
        super().__init__()
        # 인코더
        self.enc1 = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU())

        # 업샘플링 경로
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)  # 20→40
        self.conv_up1 = nn.Sequential(nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU())
        self.up2 = nn.Upsample(scale_factor=5, mode='linear', align_corners=True)  # 40→200
        self.conv_up2 = nn.Sequential(nn.Conv1d(32, 16, 5, padding=2), nn.ReLU())

        # Skip connection
        self.skip_up = nn.Upsample(scale_factor=10, mode='linear', align_corners=True)
        self.skip_conv = nn.Conv1d(1, 16, 1)

        # 출력
        self.output = nn.Sequential(nn.Conv1d(16, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        # x: [B, 1, 20]
        skip = self.skip_conv(self.skip_up(x))  # [B, 16, 200]
        x = self.enc1(x)   # [B, 16, 20]
        x = self.enc2(x)   # [B, 32, 20]
        x = self.up1(x)    # [B, 32, 40]
        x = self.conv_up1(x)
        x = self.up2(x)    # [B, 32, 200]
        x = self.conv_up2(x)  # [B, 16, 200]
        x = F.relu(x + skip)
        return self.output(x)  # [B, 1, 200]


class SensorEngine:
    """4채널 바이오센서 + Conv1D 초해상도"""

    SENSOR_PROFILES = [
        {'name': 'MOX-1', 'type': '금속산화물', 'sensitivity': ['citrus', 'fresh', 'green'], 'peakTime': 5, 'decayRate': 0.3},
        {'name': 'SAW-2', 'type': 'SAW', 'sensitivity': ['floral', 'rose', 'jasmine'], 'peakTime': 8, 'decayRate': 0.15},
        {'name': 'QCM-3', 'type': 'QCM 크리스탈', 'sensitivity': ['woody', 'amber', 'musk'], 'peakTime': 12, 'decayRate': 0.08},
        {'name': 'BIO-4', 'type': '바이오센서', 'sensitivity': ['sweet', 'vanilla', 'fruity'], 'peakTime': 6, 'decayRate': 0.2},
    ]

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trained = False

    def build_model(self):
        self.model = SuperResNet().to(self.device)
        print(f"[SensorEngine] Conv1D SR model on {self.device}")

    def simulate_response(self, molecule, channel=0):
        profile = self.SENSOR_PROFILES[channel % 4]
        labels = molecule.get('odor_labels', [])
        match = len(set(profile['sensitivity']) & set(labels))
        amp = 0.3 + (match / max(len(profile['sensitivity']), 1)) * 0.7

        high_res = []
        for i in range(200):
            t = i / 200 * 30
            rise = 1 - np.exp(-t / profile['peakTime'])
            decay = np.exp(-profile['decayRate'] * max(0, t - profile['peakTime'] * 2))
            noise = np.random.randn() * 0.02
            high_res.append(max(0, min(1, amp * rise * decay + noise)))

        low_res = [np.mean(high_res[i * 10:(i + 1) * 10]) for i in range(20)]
        return {'highRes': high_res, 'lowRes': low_res, 'sensor': profile, 'amplitude': amp}

    def train(self, molecular_engine, epochs=40, on_progress=None):
        if self.model is None:
            self.build_model()

        molecules = molecular_engine.get_all() if hasattr(molecular_engine, 'get_all') else molecular_engine
        low_data, high_data = [], []

        for mol in molecules:
            for ch in range(4):
                resp = self.simulate_response(mol, ch)
                low_data.append(resp['lowRes'])
                high_data.append(resp['highRes'])
                # 노이즈 증강
                for _ in range(2):
                    noisy = [max(0, min(1, v + np.random.randn() * 0.08)) for v in resp['lowRes']]
                    low_data.append(noisy)
                    high_data.append(resp['highRes'])

        X = torch.tensor(low_data, dtype=torch.float32).unsqueeze(1).to(self.device)   # [N, 1, 20]
        Y = torch.tensor(high_data, dtype=torch.float32).unsqueeze(1).to(self.device)  # [N, 1, 200]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if on_progress:
                on_progress(epoch + 1, epochs, total_loss / len(loader))

        self.trained = True

    @torch.no_grad()
    def super_resolve(self, low_res_signal):
        if not self.trained:
            return None
        self.model.eval()
        X = torch.tensor([low_res_signal], dtype=torch.float32).unsqueeze(1).to(self.device)
        out = self.model(X).squeeze().cpu().numpy()
        return out.tolist()

    def get_full_profile(self, molecule):
        channels = []
        for ch in range(4):
            resp = self.simulate_response(molecule, ch)
            sr = self.super_resolve(resp['lowRes']) if self.trained else None
            psnr = self._psnr(resp['highRes'], sr) if sr else None
            channels.append({
                'sensor': resp['sensor'],
                'lowRes': resp['lowRes'],
                'highRes': resp['highRes'],
                'superResolved': sr,
                'psnr': psnr,
                'amplitude': resp['amplitude']
            })
        return channels

    def _psnr(self, orig, recon):
        mse = np.mean([(o - r) ** 2 for o, r in zip(orig, recon)])
        return 10 * np.log10(1.0 / max(mse, 1e-10))
