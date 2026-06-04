"""
Copyright (c) 2024 Orange - All rights reserved

Author:  Joël Roman Ky
This code is distributed under the terms and conditions
of the MIT License (https://opensource.org/licenses/MIT)
"""

import random

import numpy as np
import torch


class TimeSeriesAugmentation:
    """Time Series Data Augmentation."""
    def __init__(self, spike_factor=3, jitter_factor=0.05, reduce_ratio=.9,
                window_ratio=.1, scales=None, scales_ratio=None, drift_slope=0.05,
                mask_ratio=None, seed=42, min_dim=.2, max_dim=.9):
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.spike_factor = spike_factor
        self.jitter_factor = jitter_factor
        self.reduce_ratio = reduce_ratio
        self.drift_slope = drift_slope
        self.mask_ratio = mask_ratio if mask_ratio is not None else [.1, .5]
        self.window_ratio = window_ratio
        self.scales = scales if scales is not None else [.5, 2.]
        self.scales_ratio = scales_ratio if scales_ratio is not None else [.1, .5]
        self.positives = {
            'jitter': self.jitter,
            'window_slicing': self.window_slicing,
            'window_warping': self.window_warping,
            'scaling': self.scaling,
            'none': self.none_augment,
        }
        self.negatives = {
            'spike': self.extreme_spike,
            'trend': self.trend,
            'shuffle': self.shuffle,
            'scaling': self.scaling,
            'masking': self.random_mask,
        }
        self.seed = seed

    def extreme_spike(self, time_series, spike_ratio=.1):
        num_spikes = int(spike_ratio * time_series.shape[0])
        np.random.seed(self.seed)
        shuffled_indices = np.random.permutation(time_series.shape[0])
        indices = shuffled_indices[:num_spikes]
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        for i in range(time_series.shape[-1]):
            if i in selected_indices:
                max_feat = time_series[:, i].max()
                spike_value = random.uniform(float(max_feat), float(max_feat) * self.spike_factor)
                time_series[indices, i] = spike_value
        return time_series

    def trend(self, time_series):
        sequence_length, n_features = time_series.size()
        time = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        drift_factor = random.uniform(0.01, self.drift_slope)
        drift = time * drift_factor
        drift = drift.expand(sequence_length, n_features)
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        time_series[:, selected_indices] += drift[:, selected_indices]
        return time_series

    def shuffle(self, time_series, max_segments=5, seg_mode="equal"):
        orig_steps = np.arange(time_series.shape[0])
        num_segs = np.random.randint(3, max_segments)
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        if num_segs > 1:
            if seg_mode == "random":
                split_points = np.random.choice(time_series.shape[0]-2, num_segs-1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs)
            random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret = time_series.clone().detach()
            for i in selected_indices:
                ret[:, i] = time_series[warp, i].clone().detach()
        else:
            ret = time_series.clone().detach()
        return ret

    def jitter(self, time_series):
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        jitter = torch.randn(time_series[:, selected_indices].shape) * self.jitter_factor
        time_series[:, selected_indices] = time_series[:, selected_indices] + jitter
        return time_series

    def scaling(self, time_series):
        scaling_factor = random.uniform(self.scales[0], self.scales[1])
        scale_ratio = random.uniform(self.scales_ratio[0], self.scales_ratio[1])
        scale_size = int(scale_ratio * time_series.size(0))
        scale_size = max(scale_size, 1)
        max_start_index = time_series.size(0) - scale_size + 1
        start_index = np.random.randint(0, max_start_index)
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        time_series[start_index : start_index + scale_size, selected_indices] *= scaling_factor
        return time_series

    def none_augment(self, time_series):
        return time_series

    def random_mask(self, time_series):
        n_samples = time_series.size(0)
        mask_factor = random.uniform(self.mask_ratio[0], self.mask_ratio[1])
        n_masked = int(n_samples * mask_factor)
        mask = torch.zeros(n_samples)
        mask[:n_masked] = 1
        mask_idx = torch.randperm(n_samples)
        mask = mask[mask_idx] == 1
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        for i in selected_indices:
            time_series[mask, i] = 0
        return time_series

    def window_slicing(self, time_series):
        target_len = int(np.ceil(self.reduce_ratio * time_series.shape[0]))
        if target_len >= time_series.shape[0]:
            return time_series
        high = time_series.shape[0] - target_len
        if high <= 0:
            return time_series
        starts = np.random.randint(low=0, high=high)
        ends = target_len + starts
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        for dim in selected_indices:
            time_series[:, dim] = torch.tensor(np.interp(
                np.linspace(0, target_len, num=time_series.shape[0]),
                np.arange(target_len),
                time_series[starts:ends, dim].numpy()
            ).T, dtype=torch.float)
        return time_series

    def window_warping(self, time_series):
        warp_scale = np.random.choice(self.scales)
        warp_size = int(np.ceil(self.window_ratio * time_series.shape[0]))
        warp_size = max(warp_size, 1)
        window_steps = np.arange(warp_size)
        high = time_series.shape[0] - warp_size - 1
        if high <= 1:
            return time_series
        window_starts = np.random.randint(low=1, high=high)
        window_ends = int(window_starts + warp_size)
        dim_ratio = random.uniform(self.min_dim, self.max_dim)
        select_dim = int(dim_ratio * time_series.shape[-1])
        selected_indices = np.random.choice(time_series.shape[-1], select_dim, replace=False)
        for dim in selected_indices:
            start_seg = time_series[:window_starts, dim].numpy()
            window_seg = np.interp(
                np.linspace(0, warp_size-1, num=int(warp_size*warp_scale)),
                window_steps,
                time_series[window_starts:window_ends, dim].numpy()
            )
            end_seg = time_series[window_ends:, dim].numpy()
            warped = np.concatenate((start_seg, window_seg, end_seg))
            time_series[:, dim] = torch.tensor(np.interp(
                np.arange(time_series.shape[0]),
                np.linspace(0, time_series.shape[0]-1., num=warped.size),
                warped
            ).T, dtype=torch.float)
        return time_series

    def augment(self, time_series, aug_funct_names=None, positive=True):
        augment_dict = self.positives if positive else self.negatives
        if not all(aug_func in augment_dict for aug_func in aug_funct_names):
            raise ValueError(f"Augmentation functions must be in {list(augment_dict.keys())}")
        aug_func_list = [augment_dict[name] for name in aug_funct_names]
        view = time_series.clone().detach()
        for aug_func in aug_func_list:
            view = aug_func(view)
        return view

    def random_augment(self, time_series, aug_funct_names=None):
        augment_dict = {**self.positives, **self.negatives}
        if not all(aug_func in augment_dict for aug_func in aug_funct_names):
            raise ValueError(f"Augmentation functions must be in {list(augment_dict.keys())}")
        aug_func = random.choice(aug_funct_names)
        return augment_dict[aug_func](time_series.clone().detach())
