"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class SVDLoss(torch.nn.Module):
    """Deep-SVDD loss."""
    def __init__(self, radius, device, center=None, nu_val=.1, objective='soft'):
        super(SVDLoss, self).__init__()
        self.device = device
        if isinstance(radius, float):
            self.radius = torch.tensor(radius).to(self.device)
        else:
            self.radius = radius.to(self.device)
        if center is not None:
            self.center = center.to(self.device)
        else:
            self.center = center
        self.objective = objective
        self.nu_val = nu_val
        self.cosine_fn = nn.CosineSimilarity(dim=-1)

    def forward(self, z_embed, update=False, test=False, cosine=False):
        dist = torch.sum((z_embed - self.center)**2, dim=-1)

        if self.objective == 'soft':
            scores = dist - self.radius ** 2
            loss = self.radius**2 + (1/self.nu_val) \
                * torch.mean(torch.max(torch.zeros_like(scores), scores))

            if update:
                self._update_radius(dist)
        else:
            scores = dist
            loss = torch.mean(dist)

        if test:
            if cosine:
                score = self.cosine_fn(F.normalize(z_embed, dim=-1),
                                        F.normalize(self.center.unsqueeze(0), dim=-1))
                scores = 1 - score
            return scores
        else:
            return loss

    def init_center(self, embeddings: torch.Tensor, eps: float = 0.1):
        """Initialize center from a tensor of embeddings (N, dim)."""
        center = embeddings.mean(dim=0)
        center[(center.abs() < eps) & (center < 0)] = -eps
        center[(center.abs() < eps) & (center > 0)] = eps
        self.center = center.to(self.device)
        return self.center

    def _update_radius(self, dist: torch.Tensor):
        self.radius = torch.tensor(np.quantile(np.sqrt(dist.clone().data.cpu().numpy()),
                                                1-self.nu_val)).to(self.device)
