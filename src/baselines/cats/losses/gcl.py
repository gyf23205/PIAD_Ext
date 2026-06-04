"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import math

import numpy as np

import torch


class GCLoss(torch.nn.Module):
    """Global Contrastive Learning loss.
    Based on this implementation
    https://github.com/emadeldeen24/TS-TCC/blob/main/models/loss.py
    """
    def __init__(self, device, temperature, class_num=2):
        super(GCLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.class_num = class_num
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_nt_xent_loss(self, z_pos_1, z_pos_2, z_neg=None):
        representations = torch.cat([z_pos_1, z_pos_2], dim=0)
        batch_size = z_pos_1.size(0)
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        if z_neg is not None:
            sim_matrix_neg = []
            for z_neg_i in z_neg:
                sim_matrix_pos_neg_1 = self._cosine_similarity(z_pos_1.unsqueeze(0),
                                                                z_neg_i.unsqueeze(0))
                sim_matrix_pos_neg_2 = self._cosine_similarity(z_pos_2.unsqueeze(0),
                                                                z_neg_i.unsqueeze(0))
                sim_matrix_neg.append(sim_matrix_pos_neg_1)
                sim_matrix_neg.append(sim_matrix_pos_neg_2)

            non_neg_values = torch.cat(sim_matrix_neg).view(2 * batch_size, -1)

        similarity_matrix = self._cosine_similarity(representations.unsqueeze(1),
                                                    representations.unsqueeze(0))

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        if z_neg is not None:
            negatives = torch.cat([non_neg_values, negatives], dim=1).view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0]).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (logits.shape[0])

    def _get_cluster_entropy(self, c_pos_1, c_pos_2, c_neg=None):
        p_pos_1 = c_pos_1.sum(0).view(-1)
        p_pos_1 /= p_pos_1.sum()
        p_pos_1 = torch.clamp(p_pos_1, min=1e-8)
        ne_i = math.log(p_pos_1.size(0)) + (p_pos_1 * torch.log(p_pos_1)).sum()

        p_pos_2 = c_pos_2.sum(0).view(-1)
        p_pos_2 /= p_pos_2.sum()
        p_pos_2 = torch.clamp(p_pos_2, min=1e-8)
        ne_j = math.log(p_pos_2.size(0)) + (p_pos_2 * torch.log(p_pos_2)).sum()
        ne_loss = ne_i + ne_j

        if c_neg is not None:
            ne_neg = 0
            for c_neg_i in c_neg:
                c_neg_i = c_neg_i.sum(0).view(-1)
                c_neg_i /= c_neg_i.sum()
                c_neg_i = torch.clamp(c_neg_i, min=1e-8)
                ne_neg += math.log(c_neg_i.size(0)) + (c_neg_i * torch.log(c_neg_i)).sum()
            ne_loss += ne_neg
        return ne_loss

    def _get_cluster_loss(self, c_pos_1, c_pos_2, c_neg=None):
        ne_loss = self._get_cluster_entropy(c_pos_1, c_pos_2, c_neg)

        c_pos_1 = c_pos_1.t()
        c_pos_2 = c_pos_2.t()
        c = torch.cat((c_pos_1, c_pos_2), dim=0)

        labels = torch.cat([torch.arange(self.class_num) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        if c_neg is not None:
            sim_matrix_neg = []
            for c_neg_i in c_neg:
                c_neg_i = c_neg_i.t()
                sim_matrix_pos_neg_1 = self._cosine_similarity(c_pos_1.unsqueeze(0),
                                                                c_neg_i.unsqueeze(0))
                sim_matrix_pos_neg_2 = self._cosine_similarity(c_pos_2.unsqueeze(0),
                                                                c_neg_i.unsqueeze(0))
                sim_matrix_neg.append(sim_matrix_pos_neg_1)
                sim_matrix_neg.append(sim_matrix_pos_neg_2)
            non_neg_values = torch.cat(sim_matrix_neg).view(2 * self.class_num, -1)

        similarity_matrix = self._cosine_similarity(c.unsqueeze(1), c.unsqueeze(0))

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        non_neg_values_3 = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        if c_neg is not None:
            negatives = torch.cat([non_neg_values, non_neg_values_3],
                                  dim=1).view(2 * self.class_num, -1)
        else:
            negatives = non_neg_values_3.detach().clone()

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0]).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss + ne_loss

    def forward(self, z_pos_1, z_pos_2, z_neg=None, cluster=False):
        if cluster:
            loss = self._get_cluster_loss(z_pos_1, z_pos_2, z_neg)
            return loss
        loss = self._get_nt_xent_loss(z_pos_1, z_pos_2, z_neg=z_neg)
        return loss
