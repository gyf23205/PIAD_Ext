"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(TimeSeriesEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, input_size//4),
            nn.ReLU(),
            nn.Linear(input_size//4, embedding_size),
        )

    def forward(self, x):
        return self.encoder(x)
