"""
CATS baseline adapter for PIAD_Ext.

Wraps the CATS contrastive learning algorithm (IEEE BigData 2024) in the same
Trainer interface used by NNGMix and DASO, so it can be trained and evaluated
on ALFA / Pegasus / spoofing datasets with a single call to fit() / predict().

Key differences from the original CATS paper:
  * Data arrives as flat vectors (N, feat_dim); we reshape to (N, win_size, n_features)
    before passing to the encoder.
  * Contrastive training is label-free (GCL + TCL losses), matching the paper.
  * After training, semi_y is used only to seed the SVDD center (labeled normals)
    and compute per-class centroids (for multi-class predictions).
  * Anomaly score = squared distance from mean-pooled embedding to SVDD center.
  * Class prediction = nearest centroid in the embedding space.
"""

import copy
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from baselines.cats.cats_model import CATSModel
from baselines.cats.losses.gcl import GCLoss
from baselines.cats.losses.dtw_loss import DTWLoss
from baselines.cats.losses.tcl import TCLoss
from baselines.cats.utils.augmentation import TimeSeriesAugmentation
from utils.metrics import (
    multihot_to_single,
    truth_multihot_to_single,
    compute_anomaly_metrics,
)

logger = logging.getLogger(__name__)


class CATSTrainer:
    """Semi-supervised wrapper around CATS for PIAD_Ext datasets.

    Config keys
    -----------
    win_size        : int   — temporal window length (e.g. 20 for Pegasus)
    n_features      : int   — features per timestep  (e.g. 44 for Pegasus)
    output_size     : int   — encoder embedding dim  (default: setting.rep = 64)
    proj_size       : int   — projection head dim    (default: output_size // 2)
    encoder_type    : str   — 'ts2vec' | 'mlp'       (default: 'ts2vec')
    lr              : float — Adam learning rate      (default: 0.001)
    n_epochs        : int   — max training epochs     (default: 100)
    batch_size      : int   — mini-batch size         (default: 512)
    weight_decay    : float — Adam weight decay       (default: 1e-5)
    patience        : int   — early-stopping patience (default: 50)
    coef_gcl        : float — GCL loss weight         (default: 0.5)
    coef_tcl        : float — TCL loss weight         (default: 0.5)
    temperature     : float — GCL temperature         (default: 0.1)
    gamma           : float — Soft-DTW gamma          (default: 0.1)
    margin          : float — TCL triplet margin      (default: 5.0)
    device          : str   — 'cuda' or 'cpu'
    """

    def __init__(self, config: dict):
        self.win_size     = config['win_size']
        self.n_features   = config['n_features']
        self.output_size  = config.get('output_size', 64)
        self.proj_size    = config.get('proj_size', self.output_size // 2)
        self.encoder_type = config.get('encoder_type', 'ts2vec')
        self.lr           = config.get('lr', 0.001)
        self.n_epochs     = config.get('n_epochs', 100)
        self.batch_size   = config.get('batch_size', 512)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.patience     = config.get('patience', 50)
        self.coef_gcl     = config.get('coef_gcl', 0.5)
        self.coef_tcl     = config.get('coef_tcl', 0.5)
        self.temperature  = config.get('temperature', 0.1)
        self.gamma        = config.get('gamma', 0.1)
        self.margin       = config.get('margin', 5.0)
        self.device       = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model: CATSModel | None = None
        self.svdd_center: torch.Tensor | None = None   # (output_size,)
        self.centroids: dict[int, torch.Tensor] = {}   # class_id -> (output_size,)
        self.best_val_auc: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """(N, feat_dim) → (N, win_size, n_features)."""
        return X.reshape(-1, self.win_size, self.n_features)

    def _get_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Return mean-pooled encoder embeddings (N, output_size)."""
        self.model.eval()
        all_emb = []
        loader = DataLoader(
            TensorDataset(torch.tensor(self._reshape(X), dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=False,
        )
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                h_i, _ = self.model(batch, mask=False)
                all_emb.append(h_i.mean(dim=1).cpu().numpy())  # mean over time
        return np.concatenate(all_emb, axis=0)

    def _augment_batch(self, batch_np: np.ndarray, augmentor: TimeSeriesAugmentation):
        """Apply augmentation per sample on CPU; returns three tensors."""
        n = len(batch_np)
        pos1_list, pos2_list, neg_list = [], [], []
        for j in range(n):
            sample = torch.tensor(batch_np[j], dtype=torch.float32)
            pos1_list.append(augmentor.augment(sample.clone(), ['jitter'], positive=True))
            pos2_list.append(augmentor.augment(sample.clone(), ['scaling'], positive=True))
            neg_list.append(augmentor.augment(sample.clone(), ['trend', 'spike'], positive=False))
        pos1 = torch.stack(pos1_list).to(self.device)
        pos2 = torch.stack(pos2_list).to(self.device)
        neg  = [torch.stack(neg_list).to(self.device)]
        return pos1, pos2, neg

    def _build_model(self) -> CATSModel:
        return CATSModel(
            input_size=self.n_features,
            proj_size=self.proj_size,
            win_size=self.win_size,
            output_size=self.output_size,
            encoder_type=self.encoder_type,
        ).to(self.device)

    def _build_losses(self):
        gcl = GCLoss(device=self.device, temperature=self.temperature)
        dtw = DTWLoss(device=self.device, use_soft_dtw=True,
                      use_cuda=False, gamma=self.gamma)
        tcl = TCLoss(
            loss_fn=dtw, device=self.device,
            crop_size_min=max(1, int(0.9 * self.win_size)),
            crop_size_max=max(2, self.win_size + 1),
            if_use_dtw=True, margin=self.margin,
        )
        return gcl, dtw, tcl

    def _compute_val_loss(self, X_val: np.ndarray, gcl: GCLoss, tcl: TCLoss,
                          augmentor: TimeSeriesAugmentation) -> float:
        self.model.eval()
        X_r = self._reshape(X_val)
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for i in range(0, len(X_r), self.batch_size):
                batch_np = X_r[i : i + self.batch_size]
                pos1, pos2, neg = self._augment_batch(batch_np, augmentor)
                pf1, pe1 = self.model(pos1)
                pf2, pe2 = self.model(pos2)
                neg_outs = [self.model(n) for n in neg]
                neg_emb  = [o[1] for o in neg_outs]
                neg_feat = [o[0] for o in neg_outs]
                loss = (self.coef_gcl * gcl(pe1, pe2, neg_emb)
                        + self.coef_tcl * tcl(pf1, pf2, neg_feat, crop=True))
                total_loss += loss.item()
                n_batches  += 1
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, semi_y: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train CATS on X_train, validate on (X_val, y_val).

        Returns best validation AUC.
        """
        self.model = self._build_model()
        gcl, dtw, tcl = self._build_losses()
        augmentor = TimeSeriesAugmentation()

        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

        X_r = self._reshape(X_train)  # (N, win_size, n_features)
        best_val_loss = float('inf')
        patience_ctr = 0
        best_state = None

        for epoch in range(self.n_epochs):
            self.model.train()
            perm = np.random.permutation(len(X_r))
            X_shuf = X_r[perm]
            epoch_loss = 0.0
            n_batches  = 0

            for i in range(0, len(X_shuf), self.batch_size):
                batch_np = X_shuf[i : i + self.batch_size]
                pos1, pos2, neg = self._augment_batch(batch_np, augmentor)

                optimizer.zero_grad()
                pf1, pe1 = self.model(pos1)
                pf2, pe2 = self.model(pos2)
                neg_outs = [self.model(n) for n in neg]
                neg_emb  = [o[1] for o in neg_outs]
                neg_feat = [o[0] for o in neg_outs]

                loss = (self.coef_gcl * gcl(pe1, pe2, neg_emb)
                        + self.coef_tcl * tcl(pf1, pf2, neg_feat, crop=True))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            scheduler.step(epoch)

            val_loss = self._compute_val_loss(X_val, gcl, tcl, augmentor)
            if epoch % 10 == 0:
                logger.info(f'Epoch {epoch:4d}  train_loss={epoch_loss/max(n_batches,1):.4f}'
                            f'  val_loss={val_loss:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr  = 0
                best_state    = copy.deepcopy(self.model.state_dict())
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # SVDD center: mean over all training windows (matches original unsupervised CATS)
        emb_center = self._get_embeddings(X_train)
        single_labels = multihot_to_single(semi_y)
        center = emb_center.mean(axis=0)
        eps = 0.1
        center[(np.abs(center) < eps) & (center < 0)] = -eps
        center[(np.abs(center) < eps) & (center > 0)] =  eps
        self.svdd_center = torch.tensor(center, dtype=torch.float32).to(self.device)

        # Per-class centroids for multi-class prediction
        self.centroids = {}
        for cls_id in np.unique(single_labels[single_labels >= 0]):
            mask = single_labels == cls_id
            if mask.sum() > 0:
                emb_cls = self._get_embeddings(X_train[mask])
                self.centroids[int(cls_id)] = torch.tensor(
                    emb_cls.mean(axis=0), dtype=torch.float32
                ).to(self.device)

        # Compute validation AUC
        val_emb    = self._get_embeddings(X_val)
        val_scores = np.sum((val_emb - self.svdd_center.cpu().numpy()) ** 2, axis=-1)
        y_true_bin = (truth_multihot_to_single(y_val) > 0).astype(int)
        try:
            self.best_val_auc = float(roc_auc_score(y_true_bin, val_scores))
        except ValueError:
            self.best_val_auc = 0.0

        logger.info(f'Training complete — val AUC = {self.best_val_auc:.4f}')
        return self.best_val_auc

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return anomaly scores (N,): squared distance to SVDD center."""
        emb    = self._get_embeddings(X_test)
        center = self.svdd_center.cpu().numpy()
        return np.sum((emb - center) ** 2, axis=-1)

    def predict_labels(self, X_test: np.ndarray) -> np.ndarray:
        """Return predicted class indices (N,) via nearest-centroid."""
        emb = self._get_embeddings(X_test)
        if not self.centroids:
            return np.zeros(len(X_test), dtype=np.int64)
        classes   = sorted(self.centroids.keys())
        c_matrix  = np.stack([self.centroids[c].cpu().numpy() for c in classes])
        # (N, n_classes) squared distances
        dists  = np.sum((emb[:, None, :] - c_matrix[None, :, :]) ** 2, axis=-1)
        nearest = np.argmin(dists, axis=-1)
        return np.array([classes[i] for i in nearest], dtype=np.int64)

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            'model_state': self.model.state_dict(),
            'svdd_center': self.svdd_center.cpu(),
            'centroids':   {k: v.cpu() for k, v in self.centroids.items()},
            'config': {
                'win_size':     self.win_size,
                'n_features':   self.n_features,
                'output_size':  self.output_size,
                'proj_size':    self.proj_size,
                'encoder_type': self.encoder_type,
            },
        }, path)
        logger.info(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt['config']
        self.win_size     = cfg['win_size']
        self.n_features   = cfg['n_features']
        self.output_size  = cfg['output_size']
        self.proj_size    = cfg['proj_size']
        self.encoder_type = cfg['encoder_type']

        self.model = self._build_model()
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        self.svdd_center = ckpt['svdd_center'].to(self.device)
        self.centroids   = {k: v.to(self.device) for k, v in ckpt['centroids'].items()}
        logger.info(f'Checkpoint loaded from {path}')
