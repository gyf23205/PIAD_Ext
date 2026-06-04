"""
CATS nn.Module extracted from cats.py.

Copyright (c) 2024 Orange - All rights reserved
Author:  Joël Roman Ky
MIT License (https://opensource.org/licenses/MIT)
"""

import torch
from torch import nn

from baselines.cats.encoders.ts2vec import TSEncoder
from baselines.cats.encoders.mlp_encoder import TimeSeriesEncoder


class CATSModel(nn.Module):
    """CATS encoder + projection head.

    Supported encoder_type: 'ts2vec' (default), 'mlp'.
    Forward returns (h_i, v_i):
      h_i — temporal embedding  (batch, win_size, output_size)
      v_i — projection vector   (batch, proj_size)
    """
    def __init__(self, input_size: int, proj_size: int, win_size: int,
                output_size: int, encoder_type: str = 'ts2vec'):
        super(CATSModel, self).__init__()
        self.encoder_type = encoder_type
        self.win_size = win_size

        if encoder_type == 'ts2vec':
            self.base = TSEncoder(input_dims=input_size, output_dims=output_size)
        elif encoder_type == 'mlp':
            self.base = TimeSeriesEncoder(input_size=input_size, embedding_size=output_size)
        else:
            raise ValueError(f"Unsupported encoder_type '{encoder_type}'. Choose 'ts2vec' or 'mlp'.")

        feat_size = output_size * win_size
        self.projection = nn.Sequential(
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size // 2),
            nn.ReLU(),
            nn.Linear(feat_size // 2, proj_size),
        )
        self.proj_size = proj_size

    def forward(self, x, mask=False):
        """Forward pass.

        Args:
            x: (batch, win_size, input_size)
            mask: apply binomial mask (ts2vec only)

        Returns:
            h_i: (batch, win_size, output_size)
            v_i: (batch, proj_size)
        """
        if self.encoder_type == 'ts2vec':
            h_i = self.base(x, mask)
        else:  # mlp — no mask support
            h_i = self.base(x)

        out = h_i.reshape(h_i.size(0), -1)
        v_i = self.projection(out)
        return h_i, v_i
