"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import numpy as np
import torch
import torch.nn as nn


class TCLoss(nn.Module):
    """Temporal Contrastive Loss."""
    def __init__(self, loss_fn, device, crop_size_min=5, crop_size_max=10,
                if_use_dtw=False, max_margin=5, min_margin=1, num_clusters=2,
                temperature=.1, margin=5):
        super(TCLoss, self).__init__()
        self.sim_loss = loss_fn
        self.crop_size_min = crop_size_min
        self.crop_size_max = crop_size_max
        self.use_dtw = if_use_dtw
        self.max_margin = max_margin
        self.min_margin = min_margin
        self.margin = margin
        self.device = device
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.clusters_centers = None
        self.num_clusters = num_clusters

    def _update_margin(self, new_margin):
        self.margin = new_margin

    def sdtw_similarity(self, vect_x, vect_y):
        b_size, win_size, n_feats = vect_x.shape
        vect_x_row = vect_x.unsqueeze(0).expand(b_size, b_size,
                                                win_size, n_feats).reshape(-1, win_size, n_feats)
        vect_y_col = vect_y.unsqueeze(1).expand(b_size, b_size,
                                                win_size, n_feats).reshape(-1, win_size, n_feats)
        sim_matrix = self.sim_loss(vect_x_row, vect_y_col).reshape(b_size, b_size)
        return sim_matrix

    def random_crop(self, data1, data3=None):
        crop_size = np.random.randint(self.crop_size_min, self.crop_size_max)
        max_start_index = data1.size(1) - crop_size + 1
        start_index = np.random.randint(0, max_start_index)
        crop_data_1 = data1[:, start_index : start_index + crop_size, :]

        start_index = np.random.randint(0, max_start_index)
        crop_data_2 = data1[:, start_index : start_index + crop_size, :]

        if data3 is not None:
            crop_data_3 = []
            for data in data3:
                start_index = np.random.randint(0, max_start_index)
                crop = data[:, start_index : start_index + crop_size, :]
                crop_data_3.append(crop)
        else:
            crop_data_3 = None
        return crop_data_1, crop_data_2, crop_data_3

    def temporal_triplet_loss(self, crop_z1, crop_z2, crop_z3=None, update=False):
        if self.use_dtw:
            loss = self.sim_loss(crop_z1, crop_z2)  # positive distance

            if crop_z3 is not None:
                loss_neg_list = []
                for crop_z3_i in crop_z3:
                    loss_neg_list.append(self.sim_loss(crop_z1, crop_z3_i))
                loss_neg = torch.mean(torch.stack(loss_neg_list), dim=0)

                if self.margin is None:
                    self.margin = self.max_margin

                dist = loss - loss_neg + self.margin
                loss = torch.clamp(dist, min=0.0)

            loss = torch.mean(loss)
            return loss
        else:
            loss = self.sim_loss(crop_z1, crop_z2, crop_z3)

    def forward(self, z1_batch, z2_batch, z3_batch=None, update=False,
                crop=True, cluster=False):
        if self.use_dtw:
            if cluster:
                raise NotImplementedError(
                    "TCLoss cluster mode requires pytorch_kmeans which is not bundled.")

            if crop:
                crop_z1, crop_z2, crop_z3 = self.random_crop(z1_batch, z3_batch)
                loss1 = self.temporal_triplet_loss(crop_z1, crop_z2, crop_z3, update)
                crop_z1, crop_z2, crop_z3 = self.random_crop(z2_batch, z3_batch)
                loss2 = self.temporal_triplet_loss(crop_z1, crop_z2, crop_z3, update)
                loss = (loss1 + loss2) / 2
            else:
                loss = self.temporal_triplet_loss(z1_batch, z2_batch, z3_batch)
        else:
            loss = 0
            for i in range(z1_batch.size(0)):
                z3_i = z3_batch[i] if z3_batch is not None else None
                loss += self.temporal_triplet_loss(z1_batch[i], z2_batch[i], z3_i, update)
            loss /= z1_batch.size(0)
        return loss
