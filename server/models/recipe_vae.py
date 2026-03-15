"""RecipeVAE — Conditional Variational Autoencoder for Recipe Generation
======================================================================
설계안 v3 Model 4 구현

잠재 공간에서 레시피(원료+비율) 생성/보간.
조건부 디코더: z(16d) + mood(16d) + season(4d)

기능:
  - generate_random(mood, season) → 새 레시피
  - interpolate(recipe_A, recipe_B, steps) → 중간 레시피
  - encode(recipe) → latent vector
  - decode(z, mood, season) → recipe

~100K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecipeVAE(nn.Module):
    """Conditional β-VAE for formula generation

    Input: formula vector [n_ingredients] — percentage per ingredient
    Latent: 16d
    Condition: mood(16d) + season(4d)
    """
    def __init__(self, n_ingredients=200, latent_dim=16, n_moods=16, n_seasons=4, beta=0.5):
        super().__init__()
        self.n_ingredients = n_ingredients
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder: formula → latent
        self.encoder = nn.Sequential(
            nn.Linear(n_ingredients, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Conditional decoder: z + condition → formula
        cond_dim = n_moods + n_seasons  # 20d
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.1),
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, n_ingredients),
            # ★ Removed nn.Softmax — use log_softmax in loss for numerical stability
        )

        # Ingredient name → index mapping (runtime에서 설정)
        self._ingredient_to_idx = {}
        self._idx_to_ingredient = {}

    def set_ingredient_mapping(self, names):
        """원료명 리스트 → 인덱스 매핑 설정"""
        self._ingredient_to_idx = {name: i for i, name in enumerate(names)}
        self._idx_to_ingredient = {i: name for i, name in enumerate(names)}

    def encode(self, x):
        """[B, n_ingredients] → mu, logvar
        NOTE: BatchNorm1d requires B>1 during training, so we
        temporarily switch to eval if B==1"""
        if x.size(0) == 1 and self.training:
            self.eval()
            h = self.encoder(x)
            self.train()
        else:
            h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        """[B, latent_dim] + [B, 20] → [B, n_ingredients] (logits)
        NOTE: BatchNorm1d requires B>1 during training"""
        inp = torch.cat([z, condition], dim=-1)
        if inp.size(0) == 1 and self.training:
            self.eval()
            out = self.decoder(inp)
            self.train()
        else:
            out = self.decoder(inp)
        return out  # raw logits, apply softmax externally for probabilities

    def forward(self, x, condition):
        """Full forward: encode → reparameterize → decode"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        return recon, mu, logvar

    def loss(self, recon_logits, x, mu, logvar):
        """β-VAE loss: Reconstruction (CE on simplex) + β * KL divergence
        
        Uses cross-entropy (log_softmax + target) instead of MSE(softmax, target)
        to avoid the vanishing-gradient anti-pattern of Softmax+MSE.
        """
        # Reconstruction: cross-entropy on simplex (target is a probability distribution)
        log_probs = F.log_softmax(recon_logits, dim=-1)
        # Ensure target sums to 1 (it should already, but be safe)
        x_normalized = x / x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        recon_loss = -(x_normalized * log_probs).sum(dim=-1).mean()

        # KL divergence: D_KL(q(z|x) || p(z))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        return recon_loss + self.beta * kl_loss, {'recon': recon_loss, 'kl': kl_loss}

    # === Generation utilities ===

    @torch.no_grad()
    def generate_random(self, n=1, mood_idx=None, season_idx=None, device='cpu',
                        top_k=10, temperature=1.0):
        """랜덤 레시피 생성

        Args:
            n: 생성할 레시피 수
            mood_idx: int (0-15)
            season_idx: int (0-3)
            top_k: 상위 K개 재료만 선택
            temperature: 다양성 제어 (>1=다양, <1=보수적)
        """
        self.eval()

        # Random latent
        z = torch.randn(n, self.latent_dim, device=device) * temperature

        # Condition
        condition = torch.zeros(n, 20, device=device)
        if mood_idx is not None:
            condition[:, mood_idx] = 1.0
        if season_idx is not None:
            condition[:, 16 + season_idx] = 1.0

        # Decode (returns logits now)
        logits = self.decode(z, condition)
        formulas = F.softmax(logits, dim=-1)  # Convert to probabilities for selection

        # Top-K selection
        results = []
        for formula in formulas:
            # Select top K ingredients
            topk_vals, topk_idx = torch.topk(formula, min(top_k, self.n_ingredients))
            # Renormalize
            topk_vals = topk_vals / topk_vals.sum()

            recipe = []
            for val, idx in zip(topk_vals, topk_idx):
                if val.item() > 0.01:  # 1% 이상만
                    name = self._idx_to_ingredient.get(idx.item(), f'ingredient_{idx.item()}')
                    recipe.append({
                        'ingredient': name,
                        'ratio': round(val.item() * 100, 2),
                    })
            results.append(recipe)

        return results

    @torch.no_grad()
    def interpolate(self, recipe_a, recipe_b, steps=5, condition=None, device='cpu'):
        """두 레시피 사이 보간

        Args:
            recipe_a, recipe_b: [n_ingredients] formula vectors
            steps: 보간 스텝 수
            condition: [20] condition vector
        """
        self.eval()
        mu_a, _ = self.encode(recipe_a.unsqueeze(0).to(device))
        mu_b, _ = self.encode(recipe_b.unsqueeze(0).to(device))

        if condition is None:
            condition = torch.zeros(1, 20, device=device)
        else:
            condition = condition.unsqueeze(0).to(device)

        interpolated = []
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * mu_a + alpha * mu_b
            formula = self.decode(z, condition).squeeze(0)
            interpolated.append(formula.cpu())

        return interpolated

    @torch.no_grad()
    def encode_recipe(self, ingredients, ratios, device='cpu'):
        """원료명+비율 → latent vector"""
        formula = torch.zeros(self.n_ingredients, device=device)
        for name, ratio in zip(ingredients, ratios):
            idx = self._ingredient_to_idx.get(name)
            if idx is not None:
                formula[idx] = ratio / 100.0  # Normalize to 0-1
        mu, _ = self.encode(formula.unsqueeze(0))
        return mu.squeeze(0)


if __name__ == '__main__':
    print("=== RecipeVAE Architecture Test ===")
    model = RecipeVAE(n_ingredients=200)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    B = 4
    x = torch.rand(B, 200)
    x = x / x.sum(dim=-1, keepdim=True)  # Normalize
    condition = torch.zeros(B, 20)
    condition[:, 0] = 1  # romantic mood
    condition[:, 16] = 1  # spring

    recon, mu, logvar = model(x, condition)
    loss, details = model.loss(recon, x, mu, logvar)
    print(f"  Recon: {recon.shape}, Loss: {loss.item():.4f} (recon={details['recon'].item():.4f}, kl={details['kl'].item():.4f})")

    # Test generation
    model.set_ingredient_mapping([f'ing_{i}' for i in range(200)])
    recipes = model.generate_random(n=2, mood_idx=0, season_idx=1)
    for i, r in enumerate(recipes):
        print(f"  Recipe {i}: {len(r)} ingredients")
        for item in r[:3]:
            print(f"    {item['ingredient']}: {item['ratio']}%")

    print("\n✅ RecipeVAE test passed!")
