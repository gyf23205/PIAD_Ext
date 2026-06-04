"""
NNG-Mix baseline for PIAD_Ext.

Transfers the NNG-Mix pseudo-anomaly generation algorithm (from C:/Code/NNG-Mix)
into the PIAD_Ext framework.  Training uses standard cross-entropy on labeled
data augmented with NNG-Mix pseudo-anomalies — no EMA, no queue.

Reference:
  "NNG-Mix: Improving Semi-supervised Anomaly Detection with Pseudo-anomaly Generation"
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import (
    compute_anomaly_metrics,
    multihot_to_single,
    truth_multihot_to_single,
)

__all__ = [
    "generate_nng_mix",
    "NNGMixNet",
    "NNGMixTrainer",
    "compute_anomaly_metrics",
    "multihot_to_single",
    "truth_multihot_to_single",
]


# ---------------------------------------------------------------------------
# NNG-Mix data augmentation
# ---------------------------------------------------------------------------

def generate_nng_mix(
    X_anomaly: np.ndarray,
    y_anomaly: np.ndarray,
    X_normal_pool: np.ndarray,
    n_pseudo: int,
    nn_k: int = 10,
    nn_k_anomaly: int = 10,
    mixup_alpha: float = 0.2,
    mixup_beta: float = 0.2,
    nn_mix_gaussian: bool = False,
    nn_mix_gaussian_std: float = 0.01,
    use_uniform: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pseudo-anomalies using the NNG-Mix algorithm.

    For each pseudo-anomaly, with 50% probability:
      - Mix a labeled anomaly with a nearest-normal-pool sample.
      - Mix a labeled anomaly with a nearest-anomaly sample.

    Class label of the pseudo-anomaly inherits from the anchor anomaly (index1),
    which preserves the multi-class structure of PIAD_Ext datasets.

    Args:
        X_anomaly:      (n_a, d) labeled anomaly samples
        y_anomaly:      (n_a,)   integer class indices 1..K
        X_normal_pool:  (n_u, d) unlabeled + labeled-normal samples
        n_pseudo:       number of pseudo-anomalies to generate
        nn_k:           K nearest neighbours queried from the normal pool
        nn_k_anomaly:   K nearest neighbours queried from anomaly set
        mixup_alpha/beta: Beta distribution parameters for mixing coefficient
        nn_mix_gaussian: if True, add Gaussian noise before mixing
        nn_mix_gaussian_std: std-dev of the Gaussian noise
        use_uniform:    if True, sample λ from Uniform(0,1) instead of Beta

    Returns:
        X_pseudo: (n_pseudo, d)
        y_pseudo: (n_pseudo,)  int64 class indices 1..K
    """
    if n_pseudo == 0 or len(X_anomaly) == 0:
        return np.empty((0, X_anomaly.shape[1])), np.empty(0, dtype=np.int64)

    n_a = len(X_anomaly)
    n_u = len(X_normal_pool)
    d = X_anomaly.shape[1]

    tree_normal = spatial.KDTree(X_normal_pool) if n_u > 0 else None
    tree_anomaly = spatial.KDTree(X_anomaly) if n_a > 1 else None

    k_normal = max(1, min(nn_k, n_u))
    k_anom = max(1, min(nn_k_anomaly, n_a))

    X_list, y_list = [], []

    for _ in range(n_pseudo):
        use_normal_branch = (np.random.uniform() > 0.5) and (tree_normal is not None)

        i1 = np.random.randint(n_a)

        if use_normal_branch:
            _, ind = tree_normal.query(X_anomaly[i1:i1 + 1], k=k_normal)
            i2 = int(np.random.choice(ind[0]))

            lam = np.random.uniform() if use_uniform else np.random.beta(mixup_alpha, mixup_beta)

            if nn_mix_gaussian:
                n1 = np.random.normal(0, nn_mix_gaussian_std, d)
                n2 = np.random.normal(0, nn_mix_gaussian_std, d)
                sample = lam * (n1 + X_anomaly[i1]) + (1 - lam) * (n2 + X_normal_pool[i2])
            else:
                sample = lam * X_anomaly[i1] + (1 - lam) * X_normal_pool[i2]
        else:
            if tree_anomaly is not None:
                _, ind = tree_anomaly.query(X_anomaly[i1:i1 + 1], k=k_anom)
                ind = ind.reshape(-1)
                candidates = ind[ind != i1]
                i2 = int(np.random.choice(candidates)) if len(candidates) > 0 else i1
            else:
                i2 = i1

            lam = np.random.uniform() if use_uniform else np.random.beta(mixup_alpha, mixup_beta)

            if nn_mix_gaussian:
                n1 = np.random.normal(0, nn_mix_gaussian_std, d)
                n2 = np.random.normal(0, nn_mix_gaussian_std, d)
                sample = lam * (n1 + X_anomaly[i1]) + (1 - lam) * (n2 + X_anomaly[i2])
            else:
                sample = lam * X_anomaly[i1] + (1 - lam) * X_anomaly[i2]

        X_list.append(sample)
        y_list.append(y_anomaly[i1])

    return np.vstack(X_list), np.array(y_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class NNGMixNet(nn.Module):
    """MLP encoder + linear classifier for NNG-Mix.

    Simpler than DASONet (no projection head, no EMA) — all we need for
    standard supervised cross-entropy training on augmented data.

    h_dims / rep_dim map onto the same layer sizes as setting.hd1/hd2/rep so
    that the backbone capacity matches the proposed method when configured from
    main_NNGMix.py.
    """

    def __init__(self, input_dim: int, h_dims: list, rep_dim: int, n_classes: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in h_dims:
            layers += [
                nn.Linear(in_dim, h, bias=False),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(inplace=True),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, rep_dim, bias=False))
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(rep_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.classifier(self.encoder(x))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class NNGMixTrainer:
    """NNG-Mix semi-supervised trainer.

    Interface mirrors DASOTrainer (fit / predict / predict_labels /
    save_checkpoint / load_checkpoint) for drop-in comparison.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        # Training
        self.lr = cfg.get("lr", 0.001)
        self.n_epochs = cfg.get("n_epochs", 200)
        self.batch_size = cfg.get("batch_size", 128)
        self.h_dims = cfg.get("h_dims", [256, 512])
        self.rep_dim = cfg.get("rep_dim", 64)
        self.eval_period = cfg.get("eval_period", 10)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        # NNG-Mix augmentation
        self.num_times = cfg.get("num_times", 10)
        self.nn_k = cfg.get("nn_k", 10)
        self.nn_k_anomaly = cfg.get("nn_k_anomaly", 10)
        self.mixup_alpha = cfg.get("mixup_alpha", 0.2)
        self.mixup_beta = cfg.get("mixup_beta", 0.2)
        self.nn_mix_gaussian = cfg.get("nn_mix_gaussian", True)
        self.nn_mix_gaussian_std = cfg.get("nn_mix_gaussian_std", 0.01)
        self.use_uniform = cfg.get("use_uniform", False)

        self.net: NNGMixNet | None = None
        self.n_classes: int | None = None
        self.input_dim: int | None = None
        self._best_auc: float = 0.0
        self._best_net_state: dict | None = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        semi_y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Train on semi-supervised data augmented with NNG-Mix pseudo-anomalies.

        X_train:  (n_sub, feat_dim)  — training features (Subset of full train)
        semi_y:   (n_sub, n_ac)      — semi-supervised multi-hot labels
                  all -1 = unlabeled, all 0 = labeled normal, 1s = labeled anomaly
        X_val:    (n_val, feat_dim)
        y_val:    (n_val, n_ac)      — ground-truth multi-hot labels (no -1s)

        Returns best validation AUC.
        """
        assert X_train.shape[0] == semi_y.shape[0], (
            f"X_train rows ({X_train.shape[0]}) != semi_y rows ({semi_y.shape[0]}). "
            "Extract via dataset.train_set (the Subset), not data_direct()."
        )

        X_train = X_train.reshape(len(X_train), -1)
        self.input_dim = X_train.shape[1]
        self.n_classes = semi_y.shape[1] + 1  # anomaly columns + normal class 0
        device = torch.device(self.device)

        single_y = multihot_to_single(semi_y)
        val_y_single = truth_multihot_to_single(y_val)

        # Partition training data
        X_unlabeled = X_train[single_y < 0]
        X_labeled_normal = X_train[single_y == 0]
        X_anomaly = X_train[single_y > 0]
        y_anomaly = single_y[single_y > 0]  # class indices 1..K

        # NNG-Mix: normal pool = unlabeled + labeled normal (mirrors original paper)
        parts = [X_unlabeled]
        if len(X_labeled_normal) > 0:
            parts.append(X_labeled_normal)
        X_normal_pool = np.vstack(parts)

        print(
            f"  NNGMix | n_classes={self.n_classes}  "
            f"anomaly={len(X_anomaly)}  normal_pool={len(X_normal_pool)}"
        )

        # Generate pseudo-anomalies
        n_pseudo = len(X_anomaly) * self.num_times
        X_pseudo, y_pseudo = generate_nng_mix(
            X_anomaly, y_anomaly, X_normal_pool, n_pseudo,
            nn_k=self.nn_k,
            nn_k_anomaly=self.nn_k_anomaly,
            mixup_alpha=self.mixup_alpha,
            mixup_beta=self.mixup_beta,
            nn_mix_gaussian=self.nn_mix_gaussian,
            nn_mix_gaussian_std=self.nn_mix_gaussian_std,
            use_uniform=self.use_uniform,
        )
        print(f"  Generated {len(X_pseudo)} pseudo-anomalies")

        # Build augmented training set.
        # Unlabeled samples are treated as class 0 (normal), matching the original
        # NNG-Mix paper where unlabeled_data gets y=0.
        X_combined = np.vstack([X_normal_pool, X_anomaly] +
                               ([X_pseudo] if len(X_pseudo) > 0 else []))
        y_combined = np.concatenate([
            np.zeros(len(X_normal_pool), dtype=np.int64),
            y_anomaly,
            y_pseudo,
        ])

        # Build network and optimizer
        self.net = NNGMixNet(
            self.input_dim, self.h_dims, self.rep_dim, self.n_classes
        ).to(device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        milestones = [int(self.n_epochs * f) for f in (0.5, 0.75, 0.9)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1
        )

        X_t = torch.tensor(X_combined, dtype=torch.float32)
        y_t = torch.tensor(y_combined, dtype=torch.long)
        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self._best_auc = 0.0
        self._best_net_state = None

        for epoch in range(1, self.n_epochs + 1):
            self.net.train()
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                loss = F.cross_entropy(self.net(X_b), y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            if epoch % self.eval_period == 0 or epoch == self.n_epochs:
                auc = self._eval_auc(X_val, val_y_single, device)
                print(
                    f"  Epoch {epoch:4d}/{self.n_epochs} | "
                    f"AUC={auc:.4f}  best={self._best_auc:.4f}"
                )
                if auc > self._best_auc:
                    self._best_auc = auc
                    self._best_net_state = copy.deepcopy(self.net.state_dict())

        if self._best_net_state is not None:
            self.net.load_state_dict(self._best_net_state)

        return self._best_auc

    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Anomaly score = 1 - P(normal | x).  Higher means more anomalous."""
        device = torch.device(self.device)
        self.net.eval()
        scores = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
            batch_size=512, shuffle=False,
        )
        with torch.no_grad():
            for (x,) in loader:
                logits = self.net(x.to(device))
                probs = F.softmax(logits, dim=1)
                scores.append((1.0 - probs[:, 0]).cpu().numpy())
        return np.concatenate(scores)

    def predict_labels(self, X_test: np.ndarray) -> np.ndarray:
        """Return predicted class indices (argmax of logits)."""
        device = torch.device(self.device)
        self.net.eval()
        preds = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
            batch_size=512, shuffle=False,
        )
        with torch.no_grad():
            for (x,) in loader:
                logits = self.net(x.to(device))
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save model weights and meta-data."""
        state = {
            "model": self.net.state_dict(),
            "meta": {
                "input_dim": self.input_dim,
                "h_dims": self.h_dims,
                "rep_dim": self.rep_dim,
                "n_classes": self.n_classes,
                "best_auc": self._best_auc,
            },
            "cfg": {
                "num_times": self.num_times,
                "nn_k": self.nn_k,
                "nn_k_anomaly": self.nn_k_anomaly,
                "mixup_alpha": self.mixup_alpha,
                "mixup_beta": self.mixup_beta,
                "nn_mix_gaussian": self.nn_mix_gaussian,
                "nn_mix_gaussian_std": self.nn_mix_gaussian_std,
            },
        }
        torch.save(state, path)
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Restore a checkpoint saved by save_checkpoint()."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        meta = state["meta"]
        self.input_dim = meta["input_dim"]
        self.h_dims = meta["h_dims"]
        self.rep_dim = meta["rep_dim"]
        self.n_classes = meta["n_classes"]
        self._best_auc = meta.get("best_auc", 0.0)
        device = torch.device(self.device)
        self.net = NNGMixNet(
            self.input_dim, self.h_dims, self.rep_dim, self.n_classes
        ).to(device)
        self.net.load_state_dict(state["model"])

    # ------------------------------------------------------------------
    def _eval_auc(
        self, X_val: np.ndarray, y_val_single: np.ndarray, device: torch.device
    ) -> float:
        scores = self.predict(X_val)
        y_bin = (y_val_single > 0).astype(int)
        try:
            return float(roc_auc_score(y_bin, scores))
        except ValueError:
            return 0.0
