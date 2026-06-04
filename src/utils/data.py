"""Shared data-loading utilities for all PIAD_Ext baselines."""

import numpy as np
from torch.utils.data import DataLoader


def extract_numpy(torch_dataset, batch_size: int = 512):
    """Extract (X, y_multihot, semi_y_multihot) from a MySpoofingPhysical / Subset.

    Returns: X (n, feat), y (n, n_ac), semi_y (n, n_ac)
    """
    loader = DataLoader(torch_dataset, batch_size=batch_size,
                        shuffle=False, drop_last=False)
    Xs, ys, semis = [], [], []
    for batch in loader:
        sample, target, semi_target, *_ = batch
        Xs.append(sample.numpy())
        ys.append(target.numpy())
        semis.append(semi_target.numpy())
    return (np.concatenate(Xs),
            np.concatenate(ys),
            np.concatenate(semis))
