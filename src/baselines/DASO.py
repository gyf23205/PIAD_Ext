"""
DASO: Distribution-Aware Semantics-Oriented pseudo-label baseline.

Re-implemented from scratch for PIAD_Ext.  The algorithm is adapted from:
  "DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced
   Semi-Supervised Learning" (Qualcomm / KAIST, CVPR 2022)

Key differences from the original image-classification implementation:
  * MLP backbone instead of WideResNet (matches PIAD_Ext network style)
  * Multi-hot → single-label conversion for ALFA / Pegasus datasets
  * Queue populated from both labeled AND unlabeled data (predicted labels)
    to handle the case where most anomaly classes have no labeled samples.
  * Anomaly score = 1 - P(normal | x), mirroring test_daso.py in daso_timeseries.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import (  # noqa: F401 – re-exported for callers that use baselines.DASO
    multihot_to_single,
    truth_multihot_to_single,
)


# ---------------------------------------------------------------------------
# Feature Queue
# ---------------------------------------------------------------------------

class FeatureQueue:
    """Per-class prototype memory bank.

    Stores projection-head features for each class and computes running-mean
    prototypes.  Both labeled samples (true labels) and unlabeled samples
    (predicted pseudo-labels) are accepted so that unknown anomaly classes can
    accumulate entries even without supervised signal.
    """

    def __init__(self, n_classes: int, feat_dim: int, max_size: int = 256):
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.max_size = max_size
        self._bank: list[list] = [[] for _ in range(n_classes)]

    def enqueue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Add features to per-class banks.

        features: (n, feat_dim) CPU or GPU tensor (detached)
        labels:   (n,) long tensor, values in [0, n_classes-1]
        """
        features = features.detach().cpu()
        labels = labels.detach().cpu().long()
        for feat, lbl in zip(features, labels):
            c = int(lbl)
            if c < 0 or c >= self.n_classes:
                continue
            self._bank[c].append(feat.clone())
            if len(self._bank[c]) > self.max_size:
                self._bank[c] = self._bank[c][-self.max_size:]

    def get_prototypes(self) -> torch.Tensor | None:
        """Return (n_classes, feat_dim) prototype tensor.

        Returns None if any class bank is still empty.
        """
        if any(len(b) == 0 for b in self._bank):
            return None
        return torch.stack([torch.stack(b).mean(0) for b in self._bank])

    def n_filled_classes(self) -> int:
        return sum(1 for b in self._bank if len(b) > 0)


# ---------------------------------------------------------------------------
# EMA Model (teacher)
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential moving average of the student network.

    Non-trainable copy used for stable feature extraction.
    """

    def __init__(self, student_net: nn.Module, ema_decay: float = 0.999):
        self.ema_decay = ema_decay
        self.model = copy.deepcopy(student_net)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, student: nn.Module, step: int) -> None:
        """EMA update with linear warmup."""
        decay = min(1.0 - 1.0 / (step + 1), self.ema_decay)
        with torch.no_grad():
            for p_ema, p_s in zip(self.model.parameters(), student.parameters()):
                p_ema.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)

    @torch.no_grad()
    def encode_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Return projection-head features (detached, no grad)."""
        _, _, proj = self.model(x, is_train=True)
        return proj


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _EncoderBackbone(nn.Module):
    """MLP encoder whose state-dict keys mirror daso_timeseries TimeSeriesMLP.

    When nested as ``DASONet.encoder``, the keys become:
      encoder.encoder.0.weight  (first Linear)
      encoder.encoder.1.weight / .bias / .running_*  (first BN, affine=True)
      encoder.encoder.3.weight  (second Linear)
      ...
      encoder.encoder.N.weight  (final Linear → rep_dim)
    This is identical to the keys produced by daso_timeseries, so checkpoints
    from both implementations are directly interchangeable.
    """

    def __init__(self, input_dim: int, h_dims: list, rep_dim: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in h_dims:
            layers += [
                nn.Linear(in_dim, h, bias=False),
                nn.BatchNorm1d(h),          # affine=True (default) — matches daso_timeseries
                nn.LeakyReLU(inplace=True),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, rep_dim, bias=False))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class _ClassifierHead(nn.Module):
    """Linear classifier whose state-dict keys mirror daso_timeseries Classifier.

    When nested as ``DASONet.classifier``, the keys become:
      classifier.classifier.weight / .bias
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DASONet(nn.Module):
    """MLP encoder with projection head and linear classifier.

    State-dict layout is compatible with daso_timeseries checkpoints:
      encoder.encoder.*    — backbone (same key names as TimeSeriesMLP)
      classifier.classifier.* — head (same key names as Classifier)
      proj_head.*          — projection head (only in PIAD_Ext checkpoints;
                             silently absent when loading daso_timeseries weights)

    forward(x, is_train=True):
      train: (logits, features, proj_features)
      eval:  logits only — matches daso_timeseries model(x, is_train=False)
    """

    def __init__(self, input_dim: int, h_dims: list, rep_dim: int, n_classes: int):
        super().__init__()
        self.encoder = _EncoderBackbone(input_dim, h_dims, rep_dim)
        self.classifier = _ClassifierHead(rep_dim, n_classes)
        self.proj_head = nn.Sequential(
            nn.Linear(rep_dim, rep_dim, bias=False),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, is_train: bool = True):
        x = x.view(x.size(0), -1)
        features = self.encoder(x)
        logits = self.classifier(features)
        if not is_train:
            return logits
        proj = self.proj_head(features)
        return logits, features, proj


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def _weak_aug(x: torch.Tensor) -> torch.Tensor:
    return x + 0.01 * torch.randn_like(x)


def _strong_aug(x: torch.Tensor) -> torch.Tensor:
    noise = 0.05 * torch.randn_like(x)
    scale = torch.empty(x.size(0), 1, device=x.device).uniform_(0.8, 1.2)
    return x * scale + noise


from utils.metrics import compute_anomaly_metrics  # noqa: F401 – re-exported


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DASOTrainer:
    """DASO semi-supervised trainer.

    Interface follows the fit/predict pattern used by other PIAD_Ext baselines
    (RoSAS, TimesNet) for drop-in comparison.

    Important: X_train and semi_y must have the same number of rows.
    Use dataset.train_set (the Subset) rather than data_direct() to guarantee
    alignment — see main_DASO.py for the recommended extraction pattern.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.lr = cfg.get("lr", 0.001)
        self.n_epochs = cfg.get("n_epochs", 300)
        self.batch_size = cfg.get("batch_size", 64)
        self.h_dims = cfg.get("h_dims", [256, 128])
        self.rep_dim = cfg.get("rep_dim", 64)
        self.ema_decay = cfg.get("ema_decay", 0.999)
        self.queue_size = cfg.get("queue_size", 256)
        self.confidence_threshold = cfg.get("confidence_threshold", 0.95)
        self.pretrain_steps = cfg.get("pretrain_steps", 500)
        self.proto_temp = cfg.get("proto_temp", 0.05)
        self.psa_loss_weight = cfg.get("psa_loss_weight", 1.0)
        self.pl_dist_update_period = cfg.get("pl_dist_update_period", 10)
        self.with_dist_aware = cfg.get("with_dist_aware", True)
        self.eval_period = cfg.get("eval_period", 10)
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.net: DASONet | None = None
        self.ema: EMAModel | None = None
        self.n_classes: int | None = None
        self.input_dim: int | None = None
        self._best_auc: float = 0.0
        self._best_net_state: dict | None = None
        self._best_ema_state: dict | None = None

    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, semi_y: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train DASO on semi-supervised data and return best validation AUC.

        X_train:  (n_sub, feat_dim)  — training features (Subset of full train)
        semi_y:   (n_sub, n_ac)      — semi-supervised multi-hot labels
                  all -1  = unlabeled, all 0 = labeled normal, 1s = labeled anomaly
        X_val:    (n_val, feat_dim)
        y_val:    (n_val, n_ac)      — ground-truth multi-hot labels (no -1s)
        """
        assert X_train.shape[0] == semi_y.shape[0], (
            f"X_train has {X_train.shape[0]} rows but semi_y has {semi_y.shape[0]}. "
            "Extract data via dataset.train_set (the Subset), not data_direct()."
        )

        self.input_dim = X_train.shape[1]
        self.n_classes = semi_y.shape[1] + 1   # anomaly columns + normal class (0)
        device = torch.device(self.device)

        # ---- Convert labels ----
        single_y = multihot_to_single(semi_y)          # (n_sub,) — -1/0/1..K
        val_y_single = truth_multihot_to_single(y_val)  # (n_val,) — 0/1..K

        labeled_mask = single_y >= 0
        X_l = torch.tensor(X_train[labeled_mask], dtype=torch.float32)
        y_l = torch.tensor(single_y[labeled_mask], dtype=torch.long)
        X_u = torch.tensor(X_train[~labeled_mask], dtype=torch.float32)

        print(f"  DASO | n_classes={self.n_classes}  labeled={labeled_mask.sum()}"
              f"  unlabeled={(~labeled_mask).sum()}")

        labeled_ds = TensorDataset(X_l, y_l)
        unlabeled_ds = TensorDataset(X_u)
        labeled_loader = DataLoader(labeled_ds, batch_size=self.batch_size,
                                    shuffle=True, drop_last=True)
        # Unlabeled loader uses 2× batch size (common in SSL)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=self.batch_size * 2,
                                      shuffle=True, drop_last=True)

        # ---- Init model, EMA, queue, optimizer ----
        self.net = DASONet(self.input_dim, self.h_dims, self.rep_dim, self.n_classes).to(device)
        self.ema = EMAModel(self.net, self.ema_decay)
        self.ema.model.to(device)
        queue = FeatureQueue(self.n_classes, self.rep_dim, self.queue_size)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        milestones = [int(self.n_epochs * f) for f in (0.5, 0.75, 0.9)]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

        # Pseudo-label class distribution (EMA tracker)
        pl_dist = torch.ones(self.n_classes, device=device) / self.n_classes

        self._best_auc = 0.0
        step = 0

        for epoch in range(1, self.n_epochs + 1):
            self.net.train()
            labeled_iter = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            n_steps = max(len(labeled_loader), len(unlabeled_loader))

            for _ in range(n_steps):
                # ---- Fetch batches (cycle shorter loader) ----
                try:
                    X_lb, y_lb = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    X_lb, y_lb = next(labeled_iter)
                try:
                    (X_ub,) = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    (X_ub,) = next(unlabeled_iter)

                X_lb, y_lb = X_lb.to(device), y_lb.to(device)
                X_ub = X_ub.to(device)
                X_ub_weak = _weak_aug(X_ub)
                X_ub_strong = _strong_aug(X_ub)

                # ---- Enqueue EMA features (labeled batch → true labels) ----
                proj_ema_lb = self.ema.encode_proj(X_lb)
                queue.enqueue(proj_ema_lb, y_lb)

                # ---- Student forward (one BN pass for all inputs) ----
                n_l = len(X_lb)
                n_uw = len(X_ub_weak)
                all_x = torch.cat([X_lb, X_ub_weak, X_ub_strong], dim=0)
                all_logits, _, all_proj = self.net(all_x, is_train=True)

                logits_l      = all_logits[:n_l]
                logits_weak   = all_logits[n_l:n_l + n_uw]
                logits_strong = all_logits[n_l + n_uw:]
                proj_weak     = all_proj[n_l:n_l + n_uw]
                proj_strong   = all_proj[n_l + n_uw:]

                # ---- Supervised cross-entropy ----
                loss_ce = F.cross_entropy(logits_l, y_lb)
                loss_consist = torch.zeros(1, device=device).squeeze()
                loss_psa = torch.zeros(1, device=device).squeeze()

                # After pretrain_steps: enqueue unlabeled features with predicted
                # pseudo-labels so that unknown anomaly classes can accumulate
                # queue entries (needed before prototype matching is usable).
                if step >= self.pretrain_steps:
                    pred_unl = logits_weak.detach().argmax(dim=1)
                    proj_ema_ub = self.ema.encode_proj(X_ub_weak)
                    queue.enqueue(proj_ema_ub, pred_unl)

                # ---- Pseudo-label generation (full DASO once queue is ready) ----
                prototypes = queue.get_prototypes()
                if step >= self.pretrain_steps and prototypes is not None:
                    prototypes = prototypes.to(device)

                    # Linear pseudo-label
                    p_lin = F.softmax(logits_weak, dim=1)           # (n_uw, K)
                    confidence, pred_cls = p_lin.max(dim=1)          # (n_uw,)

                    # Semantic pseudo-label via prototype similarity
                    proj_w_n = F.normalize(proj_weak, dim=1)
                    proto_n = F.normalize(prototypes, dim=1)
                    sim_weak = proj_w_n @ proto_n.T / self.proto_temp  # (n_uw, K)
                    soft_target = F.softmax(sim_weak, dim=1)            # (n_uw, K)

                    # Distribution-aware blending
                    if self.with_dist_aware and pl_dist.max() > 1e-7:
                        pred_to_dist = pl_dist[pred_cls] / (pl_dist.max() + 1e-7)
                        alpha = pred_to_dist.unsqueeze(1)               # (n_uw, 1)
                        p = (1.0 - alpha) * p_lin + alpha * soft_target
                    else:
                        p = 0.5 * p_lin + 0.5 * soft_target

                    pseudo_labels = p.argmax(dim=1)                     # (n_uw,)
                    mask = confidence > self.confidence_threshold

                    if mask.sum() > 0:
                        # Consistency cross-entropy on strong augmentation
                        loss_consist = F.cross_entropy(
                            logits_strong[mask], pseudo_labels[mask].detach())

                        # Prototype Semantic Alignment (PSA) loss
                        proj_s_n = F.normalize(proj_strong, dim=1)
                        sim_strong = proj_s_n @ proto_n.T / self.proto_temp
                        log_sim_strong = F.log_softmax(sim_strong[mask], dim=1)
                        loss_psa = -(soft_target[mask].detach() * log_sim_strong).sum(dim=1).mean()

                    # Update pseudo-label distribution tracker
                    if step % self.pl_dist_update_period == 0:
                        new_dist = torch.zeros(self.n_classes, device=device)
                        new_dist.scatter_add_(
                            0, pseudo_labels, torch.ones(n_uw, device=device))
                        new_dist = new_dist / (n_uw + 1e-7)
                        pl_dist = 0.9 * pl_dist + 0.1 * new_dist

                # ---- Optimisation step ----
                total_loss = loss_ce + loss_consist + self.psa_loss_weight * loss_psa
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                self.ema.update(self.net, step)
                step += 1

            scheduler.step()

            # ---- Validation ----
            if epoch % self.eval_period == 0 or epoch == self.n_epochs:
                auc = self._eval_auc(X_val, val_y_single, device)
                filled = queue.n_filled_classes()
                print(f"  Epoch {epoch:4d}/{self.n_epochs} | "
                      f"AUC={auc:.4f}  best={self._best_auc:.4f}  "
                      f"queue={filled}/{self.n_classes}  step={step}")
                if auc > self._best_auc:
                    self._best_auc = auc
                    self._best_net_state = copy.deepcopy(self.net.state_dict())
                    self._best_ema_state = copy.deepcopy(self.ema.model.state_dict())

        # ---- Restore best checkpoint ----
        if self._best_net_state is not None:
            self.net.load_state_dict(self._best_net_state)
            self.ema.model.load_state_dict(self._best_ema_state)

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
                logits = self.net(x.to(device), is_train=False)
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
                logits = self.net(x.to(device), is_train=False)
                preds.append(logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds)

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save model weights and meta-data.

        Checkpoint keys match daso_timeseries save_checkpoint() so that
        test_DASO.py and test_daso.py share the same loading logic.
        """
        state = {
            "model": self.net.state_dict(),
            "ema_model": self.ema.model.state_dict(),
            "meta": {
                "input_dim": self.input_dim,
                "h_dims": self.h_dims,
                "rep_dim": self.rep_dim,
                "n_classes": self.n_classes,
                "best_auc": self._best_auc,
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
        self.net = DASONet(self.input_dim, self.h_dims, self.rep_dim, self.n_classes).to(device)
        self.ema = EMAModel(self.net, self.ema_decay)
        self.ema.model.to(device)
        self.net.load_state_dict(state["model"])
        self.ema.model.load_state_dict(state["ema_model"])

    # ------------------------------------------------------------------
    def _eval_auc(self, X_val: np.ndarray, y_val_single: np.ndarray,
                  device: torch.device) -> float:
        scores = self.predict(X_val)
        y_bin = (y_val_single > 0).astype(int)
        try:
            return float(roc_auc_score(y_bin, scores))
        except ValueError:
            return 0.0
