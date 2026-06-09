from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
from base.spoofing_dataset_next import MySpoofingPhysical
from base.spoofing_dataset import MySpoofing
from .preprocessing import create_semisupervised_setting
import torch
import os
import logging
import numpy as np
from sklearn.model_selection import train_test_split

from pathlib import Path



class SpoofingWindPhysical(BaseADDataset):
    def __init__(self, root: str, dataset_name: str, n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None):
        super().__init__(root)

        # Define normal and outlier classes

        self.n_classes = 3 # 0: normal, 1: linear attack, 2: sinusoid attack
        self.normal_classes = (0,)
        self.outlier_classes = (1, 2)
        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = self.outlier_classes[:n_known_outlier_classes]
        self.unknown_outlier_classes = tuple(set(self.outlier_classes) - set(self.known_outlier_classes))
        self.n_anomaly_classes = len(self.outlier_classes)
        # Get logger
        logger = logging.getLogger()

        # Load data
        test_ratio = 0.2
        # test_ratio = 0.0001
        path = os.path.join(root,'multi_anomalies')
        signals = np.load(os.path.join(path,'data_real_spoofing_wind.npy'))
        signals_next = np.load(os.path.join(path, 'next_real_spoofing_wind.npy'))
        flags = np.load(os.path.join(path,'labels_real_spoofing_wind.npy'))
        idx_norm = flags == 0
        idx_out = np.isin(flags, self.outlier_classes)

        # Convert scalar flags → 2-D multi-hot: (n_samples, n_anomaly_classes)
        n_ac = self.n_anomaly_classes
        flags_mh = np.zeros((len(flags), n_ac), dtype=np.float32)
        for col, cls in enumerate(self.outlier_classes):
            flags_mh[flags == cls, col] = 1.0

        # Split normal samples
        (X_train_norm, X_test_norm,
         fmh_train_norm, fmh_test_norm,
         next_train_norm, next_test_norm) = train_test_split(
            signals[idx_norm], flags_mh[idx_norm], signals_next[idx_norm],
            test_size=test_ratio, random_state=random_state)

        # Split outlier samples
        (X_train_out, X_test_out,
         fmh_train_out, fmh_test_out,
         next_train_out, next_test_out) = train_test_split(
            signals[idx_out], flags_mh[idx_out], signals_next[idx_out],
            test_size=test_ratio, random_state=random_state)

        X_train    = np.concatenate([X_train_norm, X_train_out])
        X_test     = np.concatenate([X_test_norm,  X_test_out])
        y_train    = np.concatenate([fmh_train_norm, fmh_train_out])   # multi-hot
        y_test     = np.concatenate([fmh_test_norm,  fmh_test_out])    # multi-hot
        next_train = np.concatenate([next_train_norm, next_train_out])
        next_test  = np.concatenate([next_test_norm,  next_test_out])
        logger.info(f'n sample in train: Normal: {len(fmh_train_norm)}, '
                    f'anomaly columns sum: {fmh_train_out.sum(axis=0).tolist()}')
        logger.info(f'n sample in test:  Normal: {len(fmh_test_norm)}, '
                    f'anomaly columns sum: {fmh_test_out.sum(axis=0).tolist()}')

        # Construct validation set
        val_ratio = 0.5
        idx_val = np.random.choice(len(y_test), size=int(val_ratio * len(y_test)), replace=False)
        mask = np.ones(len(y_test), dtype=bool)
        mask[idx_val] = False
        X_val  = X_test[~mask]
        y_val  = y_test[~mask]
        X_test = X_test[mask]
        y_test = y_test[mask]

        next_val  = next_test[~mask]
        next_test = next_test[mask]

        # Get training set
        train_set = MySpoofingPhysical(X_train, y_train, next_train)

        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(
            train_set.targets.cpu().numpy(),
            self.known_outlier_classes,
            self.outlier_classes,
            ratio_known_normal, ratio_known_outlier, ratio_pollution
        )
        train_set.semi_targets[idx] = torch.tensor(semi_targets, dtype=torch.float32)

        self.X_train, self.y_train, self.semi_y, self.X_test, self.y_test, self.X_val, self.y_val = X_train, y_train, np.array(semi_targets), X_test, y_test, X_val, y_val
        # Subset train_+set to semi_supervised setup
        self.train_set = Subset(train_set, idx)
        self.val_set = MySpoofingPhysical(X_val, y_val, next_val)

        #Get test set
        self.test_set = MySpoofingPhysical(X_test, y_test, next_test)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size,shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=False)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, val_loader, test_loader

    def data_direct(self):
        return self.X_train, self.y_train, self.semi_y, self.X_test, self.y_test, self.X_val, self.y_val
