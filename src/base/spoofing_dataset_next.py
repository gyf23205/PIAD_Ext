import torch
from torch.utils.data import Dataset

class MySpoofingPhysical(Dataset):
    def __init__(self, data, targets, data_next) -> None:
        super().__init__()
        self.classes = [0, 1]

        self.data = torch.tensor(data, dtype=torch.float32)
        # targets is 2D multi-hot: (n_samples, n_anomaly_classes)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.data_next = torch.tensor(data_next, dtype=torch.float32)

        # Initialize all semi_targets to -1 (unlabeled).
        # Shape matches targets: (n_samples, n_anomaly_classes).
        # Semantics:  all -1  → unlabeled
        #             all  0  → labeled normal
        #             some 1s → labeled anomaly (multi-hot for known classes)
        self.semi_targets = -torch.ones_like(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index, data_next)
              target and semi_target are 1-D float tensors of length n_anomaly_classes.
        """
        sample     = self.data[index]
        target     = self.targets[index]        # (n_anomaly_classes,)
        semi_target = self.semi_targets[index]  # (n_anomaly_classes,)
        data_next  = self.data_next[index]

        return sample, target, semi_target, index, data_next

    def __len__(self):
        return len(self.targets)
