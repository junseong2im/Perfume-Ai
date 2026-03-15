"""RecipeVAE - Conditional Variational Autoencoder for Recipe Generation
======================================================================
Latent space recipe generation and interpolation.
Conditional decoder: z(16d) + mood(16d) + season(4d)
~100K parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecipeVAE(nn.Module):
    def __init__(self, n_ingredients=200, latent_dim=16, n_moods=16, n_seasons=4, beta=0.5):
        super().__init__()
        self.n_ingredients = n_ingredients
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(n_ingredients, 128), nn.ReLU(), nn.GroupNorm(1, 128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.GroupNorm(1, 64),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        cond_dim = n_moods + n_seasons
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64), nn.ReLU(), nn.GroupNorm(1, 64), nn.Dropout(0.1),
            nn.Linear(64, 128), nn.ReLU(), nn.GroupNorm(1, 128),
            nn.Linear(128, n_ingredients), nn.Softmax(dim=-1),
        )
        self._ingredient_to_idx = {}
        self._idx_to_ingredient = {}

    def set_ingredient_mapping(self, names):
        self._ingredient_to_idx = {name: i for i, name in enumerate(names)}
        self._idx_to_ingredient = {i: name for i, name in enumerate(names)}

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, condition):
        return self.decoder(torch.cat([z, condition], dim=-1))

    def forward(self, x, condition):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, condition)
        return recon, mu, logvar

    def loss(self, recon, x, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + self.beta * kl_loss, {'recon': recon_loss, 'kl': kl_loss}

    @torch.no_grad()
    def generate_random(self, n=1, mood_idx=None, season_idx=None, device='cpu',
                        top_k=10, temperature=1.0):
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device) * temperature
        condition = torch.zeros(n, 20, device=device)
        if mood_idx is not None:
            condition[:, mood_idx] = 1.0
        if season_idx is not None:
            condition[:, 16 + season_idx] = 1.0
        formulas = self.decode(z, condition)
        results = []
        for formula in formulas:
            topk_vals, topk_idx = torch.topk(formula, min(top_k, self.n_ingredients))
            topk_vals = topk_vals / topk_vals.sum()
            recipe = []
            for val, idx in zip(topk_vals, topk_idx):
                if val.item() > 0.01:
                    name = self._idx_to_ingredient.get(idx.item(), f'ingredient_{idx.item()}')
                    recipe.append({'ingredient': name, 'ratio': round(val.item() * 100, 2)})
            results.append(recipe)
        return results

    @torch.no_grad()
    def interpolate(self, recipe_a, recipe_b, steps=5, condition=None, device='cpu'):
        self.eval()
        mu_a, _ = self.encode(recipe_a.unsqueeze(0).to(device))
        mu_b, _ = self.encode(recipe_b.unsqueeze(0).to(device))
        if condition is None:
            condition = torch.zeros(1, 20, device=device)
        else:
            condition = condition.unsqueeze(0).to(device)
        return [(1 - a) * mu_a + a * mu_b
                for a in np.linspace(0, 1, steps)]
