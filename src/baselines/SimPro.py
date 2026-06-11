"""
SimPro: A Simple Probabilistic Framework for Realistic Long-Tailed
Semi-Supervised Learning.

Re-implemented for PIAD_Ext from:
  "SimPro: A Simple Probabilistic Framework Towards Realistic Long-Tailed
   Semi-Supervised Learning"  Du et al., ICML 2024.
  https://github.com/LeapLabTHU/SimPro

Key adaptations from the original image-classification implementation:
  * MLP backbone instead of WideResNet (matches PIAD_Ext network style)
  * Multi-hot → single-label conversion for ALFA / Pegasus datasets
  * Gaussian-noise augmentation instead of RandAugment (tabular/time-series)
  * Anomaly score = 1 - P(normal | x) from raw-logit softmax (paper Eq. 11;
    the φ adjustment lives in the training loss only, as in the original repo)
  * α auto-computed from labeled/unlabeled ratio per the paper (Sec. 3.3)
  * φ / π_u smoothed and floored: unlike the original benchmarks, a class can
    have zero labeled samples here (e.g. ratio_known_outlier = 0)
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import (  # noqa: F401 – re-exported for callers that use baselines.SimPro
    multihot_to_single,
    truth_multihot_to_single,
    compute_anomaly_metrics,
)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _weak_aug(x: torch.Tensor) -> torch.Tensor:
    return x + 0.01 * torch.randn_like(x)


def _strong_aug(x: torch.Tensor) -> torch.Tensor:
    noise = 0.05 * torch.randn_like(x)
    scale = torch.empty(x.size(0), *([1] * (x.dim() - 1)), device=x.device).uniform_(0.8, 1.2)
    return x * scale + noise


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class SimProNet(nn.Module):
    """MLP encoder + linear classifier for SimPro.

    Architecture mirrors DASO's _EncoderBackbone + linear head so that
    hyperparameter choices (h_dims, rep_dim) are directly comparable.

    forward(x) -> logits of shape (batch, n_classes)
    No softmax applied; callers add the Bayes logit adjustment (τ·log φ) before
    passing to cross-entropy or softmax.
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
        self.encoder   = nn.Sequential(*layers)
        self.classifier = nn.Linear(rep_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.classifier(self.encoder(x))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SimProTrainer:
    """SimPro semi-supervised trainer adapted for PIAD_Ext anomaly detection.

    Implements the EM-based probabilistic framework from Algorithm 1 of the
    SimPro paper with an MLP backbone and Gaussian-noise augmentation.

    Interface follows the fit/predict pattern of other PIAD_Ext baselines
    (DASO, CATS, NNGMix) for drop-in comparison.

    Important: X_train and semi_y must have the same number of rows.
    Use dataset.train_set (the Subset) rather than data_direct() — see
    main_SimPro.py for the recommended extraction pattern.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.n_epochs   = cfg.get("n_epochs",    200)
        self.lr         = cfg.get("lr",           0.01)
        self.batch_size = cfg.get("batch_size",   64)
        self.h_dims     = cfg.get("h_dims",       [256, 128])
        self.rep_dim    = cfg.get("rep_dim",      64)
        self.tau        = cfg.get("tau",          1.0)     # logit-adjustment temperature
        self.threshold  = cfg.get("threshold",    0.95)    # pseudo-label confidence gate
        self.ema_u      = cfg.get("ema_u",        0.9)     # EMA rate for π_u and φ updates
        self.momentum   = cfg.get("momentum",     0.9)     # SGD momentum
        self.weight_decay = cfg.get("weight_decay", 1e-4)
        self.eval_period  = cfg.get("eval_period",  10)
        self.device     = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.model:     SimProNet | None = None
        self.n_classes: int | None = None
        self.input_dim: int | None = None
        # Distributions stored as plain tensors (moved to CPU for checkpointing)
        self._phi:   torch.Tensor | None = None   # overall class frequency φ
        self._pi_u:  torch.Tensor | None = None   # unlabeled marginal π_u
        self._best_auc:        float = 0.0
        self._best_model_state: dict | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, semi_y: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train SimPro and return best validation AUC.

        X_train:  (n_sub, feat_dim)
        semi_y:   (n_sub, n_ac)  — multi-hot, all -1 = unlabeled, all 0 = normal, 1s = anomaly
        X_val:    (n_val, feat_dim)
        y_val:    (n_val, n_ac)  — ground-truth multi-hot (no -1s)
        """
        assert X_train.shape[0] == semi_y.shape[0], (
            f"X_train has {X_train.shape[0]} rows but semi_y has {semi_y.shape[0]}. "
            "Extract data via dataset.train_set (the Subset), not data_direct()."
        )

        self.input_dim = int(np.prod(X_train.shape[1:]))
        self.n_classes = semi_y.shape[1] + 1   # anomaly columns + normal class (0)
        K = self.n_classes
        device = torch.device(self.device)

        # ---- Convert labels ----
        single_y     = multihot_to_single(semi_y)
        val_y_single = truth_multihot_to_single(y_val)

        labeled_mask = single_y >= 0
        X_l = torch.tensor(X_train[labeled_mask],  dtype=torch.float32)
        y_l = torch.tensor(single_y[labeled_mask],  dtype=torch.long)
        X_u = torch.tensor(X_train[~labeled_mask], dtype=torch.float32)

        N = len(X_l)
        M = len(X_u)
        print(f"  SimPro | n_classes={K}  labeled={N}  unlabeled={M}")

        # ---- α: balance factor (Sec. 3.3) ----
        # Paper derives α = µ·(N/M) where µ = M/N → α = 1.0.
        # Using 1.0 ensures labeled loss is always fully weighted regardless of
        # labeled/unlabeled ratio (critical when N << M).
        alpha = 1.0

        # ---- DataLoaders ----
        labeled_ds   = TensorDataset(X_l, y_l)
        labeled_loader = DataLoader(labeled_ds, batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=len(labeled_ds) >= self.batch_size)

        has_unlabeled = M > 0
        if has_unlabeled:
            unlabeled_ds     = TensorDataset(X_u)
            unlabeled_loader = DataLoader(unlabeled_ds, batch_size=self.batch_size * 2,
                                          shuffle=True,
                                          drop_last=len(unlabeled_ds) >= self.batch_size * 2)

        # ---- Model & optimiser ----
        self.model = SimProNet(self.input_dim, self.h_dims, self.rep_dim, K).to(device)
        optimizer  = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        milestones = [int(self.n_epochs * f) for f in (0.5, 0.75, 0.9)]
        scheduler  = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

        # ---- Initialise distributions (Sec. 3.3) ----
        # φ  → from labeled class frequencies (consistent init per paper).
        # Laplace smoothing keeps classes with zero labeled samples reachable
        # (the original benchmarks always have ≥1 labeled sample per class).
        counts = torch.zeros(K)
        for c in y_l.tolist():
            counts[c] += 1
        phi  = ((counts + 1.0) / (counts.sum() + K)).to(device)
        # π_u → uniform (no assumption on unlabeled distribution per paper)
        pi_u = torch.ones(K, device=device) / K

        self._best_auc        = 0.0
        self._best_model_state = None
        _best_phi  = phi.detach().cpu().clone()
        _best_pi_u = pi_u.detach().cpu().clone()

        # ---- EM training loop ----
        for epoch in range(1, self.n_epochs + 1):
            self.model.train()

            # Accumulators for end-of-epoch distribution updates (Eq. 7 & 9)
            pi_e = torch.zeros(K, device=device)   # Σ psd[mask] over epoch
            N_e  = torch.zeros(K, device=device)   # labeled count per class

            labeled_iter   = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader) if has_unlabeled else None
            # Consume the full unlabeled pool each epoch (the original cycles
            # both loaders over a fixed iteration count); driving steps off the
            # labeled loader alone starves training when N << M.
            n_steps = max(len(labeled_loader),
                          len(unlabeled_loader) if has_unlabeled else 0)

            for _ in range(n_steps):
                # ---- Labeled batch ----
                try:
                    X_lb, y_lb = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    X_lb, y_lb  = next(labeled_iter)
                X_lb, y_lb = X_lb.to(device), y_lb.to(device)

                # Count labeled samples per class for N_e
                for c in y_lb.tolist():
                    N_e[c] += 1

                # ---- Unlabeled batch (if available) ----
                loss_u = torch.zeros(1, device=device).squeeze()

                if has_unlabeled:
                    try:
                        (X_ub,) = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        (X_ub,) = next(unlabeled_iter)
                    X_ub = X_ub.to(device)

                    # E-step: generate soft pseudo-labels using Bayes classifier
                    with torch.no_grad():
                        lgt_w  = self.model(_weak_aug(X_ub))
                    adj_u  = torch.log(pi_u ** self.tau + 1e-12)         # (K,)
                    psd    = F.softmax(lgt_w + adj_u, dim=-1)            # (B_u, K)
                    mask   = psd.max(dim=-1)[0].ge(self.threshold)        # (B_u,)

                    # Accumulate pseudo-label counts (only confident samples, Eq. 7)
                    if mask.any():
                        pi_e += psd[mask].sum(dim=0).detach()

                    # M-step: unlabeled CE loss (Eq. 13 / 15)
                    adj   = torch.log(phi ** self.tau + 1e-12)
                    lgt_s = self.model(_strong_aug(X_ub))
                    if mask.any():
                        loss_u = (
                            F.cross_entropy(lgt_s + adj, psd.detach(),
                                            reduction="none") * mask.float()
                        ).mean()

                # M-step: labeled CE loss (Eq. 13 / 14); Alg. 1 augments the
                # labeled forward as well
                adj    = torch.log(phi ** self.tau + 1e-12)
                lgt_l  = self.model(_weak_aug(X_lb))
                loss_l = F.cross_entropy(lgt_l + adj, y_lb)

                loss = alpha * loss_l + loss_u
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # ---- End-of-epoch distribution updates (Eq. 7 & 9) ----
            # Floor + renormalise so no class collapses to an exact zero
            # (a hard zero makes the class unreachable for all later epochs).
            def _floor(p: torch.Tensor) -> torch.Tensor:
                p = p.clamp(min=1e-4)
                return p / p.sum()

            # Update π_u (unlabeled marginal)
            if pi_e.sum() > 1e-12:
                pi_u_new = pi_e / pi_e.sum()
                pi_u     = _floor(self.ema_u * pi_u + (1.0 - self.ema_u) * pi_u_new)

            # Update φ (overall frequency, combines labeled + pseudo-labeled)
            count = pi_e + N_e
            if count.sum() > 1e-12:
                phi_new = count / count.sum()
                phi     = _floor(self.ema_u * phi + (1.0 - self.ema_u) * phi_new)

            # ---- Validation ----
            if epoch % self.eval_period == 0 or epoch == self.n_epochs:
                auc = self._eval_auc(X_val, val_y_single, device)
                print(f"  Epoch {epoch:4d}/{self.n_epochs} | "
                      f"AUC={auc:.4f}  best={self._best_auc:.4f}  "
                      f"α={alpha:.3f}  pseudo_mass={pi_e.sum():.1f}")
                if auc > self._best_auc:
                    self._best_auc        = auc
                    self._best_model_state = copy.deepcopy(self.model.state_dict())
                    _best_phi  = phi.detach().cpu().clone()
                    _best_pi_u = pi_u.detach().cpu().clone()

        # ---- Restore best ----
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
        # Always sync self._phi / self._pi_u to the checkpoint that matched the
        # best model weights.  _eval_auc must NOT write to self._phi so that
        # _best_phi is never clobbered by a later (non-improving) eval call.
        self._phi  = _best_phi
        self._pi_u = _best_pi_u

        return self._best_auc

    # ------------------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Anomaly score = 1 − P(normal | x).  Higher = more anomalous.

        Raw-logit softmax (Eq. 11 of the paper): the φ adjustment appears in
        the training loss only, so f_θ already yields the balanced posterior —
        matching the original repo's evaluation.
        """
        device = torch.device(self.device)

        self.model.eval()
        scores = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
            batch_size=512, shuffle=False,
        )
        with torch.no_grad():
            for (x,) in loader:
                logits = self.model(x.to(device))
                probs  = F.softmax(logits, dim=1)
                scores.append((1.0 - probs[:, 0]).cpu().numpy())
        return np.concatenate(scores)

    def predict_labels(self, X_test: np.ndarray) -> np.ndarray:
        """Return predicted class indices via raw-logit argmax (Eq. 11)."""
        device = torch.device(self.device)

        self.model.eval()
        preds = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
            batch_size=512, shuffle=False,
        )
        with torch.no_grad():
            for (x,) in loader:
                logits = self.model(x.to(device))
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save model weights, distributions, and meta-data."""
        state = {
            "model": self.model.state_dict(),
            "phi":   self._phi,
            "pi_u":  self._pi_u,
            "meta":  {
                "input_dim": self.input_dim,
                "h_dims":    self.h_dims,
                "rep_dim":   self.rep_dim,
                "n_classes": self.n_classes,
                "tau":       self.tau,
                "best_auc":  self._best_auc,
            },
        }
        torch.save(state, path)
        print(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Restore a checkpoint saved by save_checkpoint()."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        meta  = state["meta"]
        self.input_dim = meta["input_dim"]
        self.h_dims    = meta["h_dims"]
        self.rep_dim   = meta["rep_dim"]
        self.n_classes = meta["n_classes"]
        self.tau       = meta.get("tau", self.tau)
        self._best_auc = meta.get("best_auc", 0.0)
        self._phi      = state["phi"]
        self._pi_u     = state["pi_u"]
        device = torch.device(self.device)
        self.model = SimProNet(
            self.input_dim, self.h_dims, self.rep_dim, self.n_classes
        ).to(device)
        self.model.load_state_dict(state["model"])

    # ------------------------------------------------------------------
    def _eval_auc(self, X_val: np.ndarray, y_val_single: np.ndarray,
                  device: torch.device) -> float:
        """Binary ROC-AUC on validation set via raw-logit softmax (Eq. 11)."""
        self.model.eval()
        scores = []
        loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
            batch_size=512, shuffle=False,
        )
        with torch.no_grad():
            for (x,) in loader:
                logits = self.model(x.to(device))
                probs  = F.softmax(logits, dim=1)
                scores.append((1.0 - probs[:, 0]).cpu().numpy())
        y_bin = (y_val_single > 0).astype(int)
        try:
            return float(roc_auc_score(y_bin, np.concatenate(scores)))
        except ValueError:
            return 0.0
