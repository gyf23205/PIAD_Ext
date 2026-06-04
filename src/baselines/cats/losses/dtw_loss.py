"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import torch
import torch.nn as nn

from baselines.cats.losses.soft_dtw_cuda import SoftDTW


class DTWLoss(nn.Module):
    """Soft-DTW divergence loss function.
    https://arxiv.org/pdf/2010.08354.pdf
    """
    def __init__(self, device, use_soft_dtw=True, use_cuda=False, gamma=.1):
        super(DTWLoss, self).__init__()
        self.device = device
        self.use_soft_dtw = use_soft_dtw
        self.soft_dtw = SoftDTW(use_cuda=use_cuda, gamma=gamma)

    def forward(self, vector_x, vector_y):
        if self.use_soft_dtw:
            loss = self.soft_dtw(vector_x, vector_y) \
                - .5 * (self.soft_dtw(vector_x, vector_x) + self.soft_dtw(vector_y, vector_y))
        else:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            dist_list = [fastdtw(vector_x[i].detach().numpy(),
                                vector_y[i].detach().numpy(),
                                dist=euclidean)
                            for i in range(vector_x.size(0))
                            ]
            dist_list = [dist_tup[0] for dist_tup in dist_list]
            dist_array = np.array(dist_list)
            loss = torch.from_numpy(dist_array)
        return loss.to(self.device)
