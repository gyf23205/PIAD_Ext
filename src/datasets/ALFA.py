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



class ALFA(BaseADDataset):
    def __init__(self, root: str, known_outlier_class: tuple = tuple(), n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0,
                 ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0, random_state=None):
        super().__init__(root)

        # Define normal and outlier classes
        # if subclasses:
        # Contain all 8 fine-grained types
        self.n_classes = 8 # 0: normal, 1: engine failure, 2, 3: aileron failure (right, left), 4: elevator stuck at zero, 5, 6, 7: rudder failure (left, right, zero);
                            # Multi-failure cases: 8: both ailerons fail, 9: rudder zero and left aileron failure
        
        self.normal_classes = (0,)
        self.outlier_classes = (1, 2, 3, 4, 5, 6, 7)
        # else:
        #     # Contain 1 case for each anomaly type, other cases are used as unseen anomalies during testing
        #     self.n_classes = 5 # 0: normal, 1: engine failure, 2: aileron failure, 3: elevator failure, 4: rudder failure
        #     self.normal_classes = (0,)
        #     self.outlier_classes = (1, 2, 3, 4)
        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        else:
            self.known_outlier_classes = known_outlier_class
        self.unknown_outlier_classes = tuple(set(self.outlier_classes) - set(self.known_outlier_classes))
        self.n_anomaly_classes = len(self.outlier_classes)
        # Get logger
        logger = logging.getLogger()

        # Load data
        test_ratio = 0.2
        path = os.path.join(root,'ALFA')
        # signals      = np.load(os.path.join(path,'X_median-resampling_single_anomalies.npy'))
        # signals_next = np.load(os.path.join(path, 'next_median-resampling_single_anomalies.npy'))
        # flags        = np.load(os.path.join(path,'y_median-resampling_single_anomalies.npy'))

        signals      = np.load(os.path.join(path,'X_all.npy'))
        signals_next = np.load(os.path.join(path, 'next_all.npy'))
        flags        = np.load(os.path.join(path,'y_all.npy'))

        # # Correct labels (map scalar class indices when not using subclasses)
        # if not subclasses:
        #     flags[flags==2] = 2
        #     flags[flags==3] = 2
        #     flags[flags==4] = 3
        #     flags[flags==5] = 4
        #     flags[flags==6] = 4

        # ------------------------------------------------------------------
        # Convert scalar flags → 2-D multi-hot: (n_samples, n_anomaly_classes)
        # Column k corresponds to outlier_classes[k].
        # Normal samples (flag == 0) get all-zero rows.
        # ------------------------------------------------------------------
        n_ac = self.n_anomaly_classes
        flags_mh = np.zeros((len(flags), n_ac), dtype=np.float32)
        for col, cls in enumerate(self.outlier_classes + (8, 9)):
            if cls == 8:
                flags_mh[flags == cls, 1:3] = 1.0
            elif cls == 9:
                flags_mh[flags == cls, 2] = 1.0
                flags_mh[flags == cls, 6] = 1.0
            else:
                flags_mh[flags == cls, col] = 1.0

        idx_norm = (flags == 0)
        idx_out = np.isin(flags, self.outlier_classes)

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
        # Use the passed random_state for the val/test split so that the split
        # is deterministic (not dependent on the global numpy random state).
        val_ratio = 0.5
        _rng = random_state if random_state is not None else np.random
        idx_val = _rng.choice(len(y_test), size=int(val_ratio*len(y_test)), replace=False)
        mask = np.ones(len(y_test), dtype=bool)
        mask[idx_val] = False
        X_val   = X_test[~mask]
        y_val   = y_test[~mask]
        X_test  = X_test[mask]
        y_test  = y_test[mask]

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
        train_set.semi_targets[idx] = torch.tensor(semi_targets)

        self.X_train  = X_train
        self.y_train  = y_train          # multi-hot (n, n_ac)
        self.semi_y   = semi_targets     # multi-hot (len(idx), n_ac)
        self.X_test   = X_test
        self.y_test   = y_test           # multi-hot
        self.X_val    = X_val
        self.y_val    = y_val            # multi-hot

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)
        self.val_set   = MySpoofingPhysical(X_val, y_val, next_val)

        # Get test set
        self.test_set  = MySpoofingPhysical(X_test, y_test, next_test)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        val_loader   = DataLoader(dataset=self.val_set,   batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=False)
        test_loader  = DataLoader(dataset=self.test_set,  batch_size=batch_size, shuffle=shuffle_test,
                                  num_workers=num_workers, drop_last=False)
        return train_loader, val_loader, test_loader

    def data_direct(self):
        return self.X_train, self.y_train, self.semi_y, self.X_test, self.y_test, self.X_val, self.y_val
