"""
CCL: Continuous Contrastive Learning for Long-Tailed Semi-Supervised Recognition.

Adapted for PIAD_Ext from:
  "Continuous Contrastive Learning for Long-Tailed Semi-Supervised Recognition"
  Zhou et al., NeurIPS 2024.  https://github.com/zhouzihao11/CCL

Key differences from the original image-classification implementation:
  * MLP backbone instead of WideResNet (matches PIAD_Ext network style)
  * Multi-hot → single-label conversion for ALFA / Pegasus datasets
  * Gaussian-noise augmentation instead of RandAugment (tabular/time-series)
  * Anomaly score = 1 - P(normal | x) from logit-adjusted balanced branch
  * Adam optimizer instead of SGD (consistent with other PIAD_Ext baselines)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import (
    compute_anomaly_metrics,
    multihot_to_single,
    truth_multihot_to_single,
)

__all__ = ["CCLNet", "CCLTrainer", "compute_anomaly_metrics",
           "multihot_to_single", "truth_multihot_to_single"]

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Augmentations  (same as DASO — tabular / time-series adaptations)
# ---------------------------------------------------------------------------

def _weak_aug(x: torch.Tensor) -> torch.Tensor:
    return x + 0.01 * torch.randn_like(x)


def _strong_aug(x: torch.Tensor) -> torch.Tensor:
    noise = 0.05 * torch.randn_like(x)
    shape = (x.size(0),) + (1,) * (x.dim() - 1)
    scale = torch.empty(shape, device=x.device).uniform_(0.8, 1.2)
    return x * scale + noise


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class CCLNet(nn.Module):
    """Dual-branch MLP with shared encoder and contrastive projection head.

    The architecture mirrors DASO's _EncoderBackbone so that state-dict keys
    are familiar, but adds a second (balanced) classification head and a
    projection head for contrastive learning.

    forward(x) -> (logits_s, logits_b, features, proj)
      logits_s : standard branch — trained without logit adjustment
      logits_b : balanced branch — used with logit adjustment during loss
      features : rep_dim-dimensional encoder output
      proj     : L2-normalised projection head output
    """

    def __init__(self, input_dim: int, h_dims: list, rep_dim: int,
                 n_classes: int, proj_dim: int | None = None):
        super().__init__()
        proj_dim = proj_dim or rep_dim

        # Shared MLP encoder (identical key layout to DASO _EncoderBackbone)
        layers: list[nn.Module] = []
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

        self.fs = nn.Linear(rep_dim, n_classes)   # standard branch
        self.fb = nn.Linear(rep_dim, n_classes)   # balanced branch

        # Contrastive projection head  g(·)  (Section 3.1, Fig. 3)
        self.proj = nn.Sequential(
            nn.Linear(rep_dim, proj_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        feat = self.encoder(x)
        logits_s = self.fs(feat)
        logits_b = self.fb(feat)
        proj = F.normalize(self.proj(feat), dim=1)   # L2-normalise
        return logits_s, logits_b, feat, proj


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CCLTrainer:
    """CCL semi-supervised trainer for PIAD_Ext.

    Implements Section 3 of Zhou et al. (NeurIPS 2024):
      - Balanced FixMatch with EMA class-prior estimation   (Section 3.2)
      - Continuous contrastive loss with reliable PLs       (Section 3.3, Eq. 17)
      - Continuous contrastive loss with smoothed PLs       (Section 3.4, Eq. 18-21)

    Interface follows the fit/predict pattern used by DASO, SimPro, CATS.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.lr          = cfg.get("lr",           0.001)
        self.n_epochs    = cfg.get("n_epochs",     300)
        self.batch_size  = cfg.get("batch_size",   64)
        self.h_dims      = cfg.get("h_dims",       [256, 128])
        self.rep_dim     = cfg.get("rep_dim",       64)
        # Loss weights  λ1, λ2  (Eq. 22)
        self.lambda1     = cfg.get("lambda1",       0.7)
        self.lambda2     = cfg.get("lambda2",       1.0)
        # Smoothed-PL propagation coefficient  β  (Eq. 21)
        self.beta_spl    = cfg.get("beta_spl",      0.2)
        # Logit-adjustment temperature  τ  (Eq. 12-13)
        self.tau_logit   = cfg.get("tau_logit",     2.0)
        # Energy score parameters  (Section 3.2)
        self.energy_T    = cfg.get("energy_T",      1.0)
        # energy_zeta: threshold E(x) ≤ ζ for reliable selection.
        # None = disable filtering (use all unlabeled samples).
        self.energy_zeta = cfg.get("energy_zeta",   None)
        # EMA momentum  α  for π̂^u update
        self.ema_alpha   = cfg.get("ema_alpha",      0.9)
        # Contrastive kernel temperature  τ_c
        self.tau_c       = cfg.get("tau_c",          0.07)
        self.eval_period = cfg.get("eval_period",    10)
        self.device      = cfg.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu")

        self.net: CCLNet | None       = None
        self.pi_u: torch.Tensor | None = None   # estimated unlabeled class prior
        self.n_classes: int | None    = None
        self.input_dim: int | None    = None
        self._best_auc: float         = 0.0
        self._best_net_state: dict | None = None
        self._best_pi_u: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, semi_y: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train CCL on semi-supervised data and return best validation AUC.

        X_train : (n_sub, feat_dim)  — training features (Subset of full train)
        semi_y  : (n_sub, n_ac)      — multi-hot semi-labels
                  all -1 = unlabeled, all 0 = labeled normal, 1s = labeled anomaly
        X_val   : (n_val, feat_dim)
        y_val   : (n_val, n_ac)      — ground-truth multi-hot (no -1s)
        """
        assert X_train.shape[0] == semi_y.shape[0], (
            f"X_train rows ({X_train.shape[0]}) ≠ semi_y rows ({semi_y.shape[0]}). "
            "Use dataset.train_set (the Subset), not data_direct()."
        )

        self.input_dim = int(np.prod(X_train.shape[1:]))
        self.n_classes = semi_y.shape[1] + 1   # anomaly columns + normal class 0
        device = torch.device(self.device)
        C = self.n_classes

        # ---- Label conversion ----
        single_y  = multihot_to_single(semi_y)
        val_y     = truth_multihot_to_single(y_val)

        labeled_mask = single_y >= 0
        X_l = torch.tensor(X_train[labeled_mask], dtype=torch.float32)
        y_l = torch.tensor(single_y[labeled_mask], dtype=torch.long)
        X_u = torch.tensor(X_train[~labeled_mask], dtype=torch.float32)

        print(f"  CCL | n_classes={C}  labeled={labeled_mask.sum()}"
              f"  unlabeled={(~labeled_mask).sum()}")

        # ---- Class priors ----
        pi_l_np = self._compute_prior(y_l.numpy(), C)
        pi_l_t  = torch.tensor(pi_l_np, dtype=torch.float32, device=device)
        pi_u_t  = torch.ones(C, device=device) / C   # uniform initialisation

        # ---- DataLoaders ----
        labeled_ds    = TensorDataset(X_l, y_l)
        unlabeled_ds  = TensorDataset(X_u)
        labeled_loader   = DataLoader(labeled_ds,   batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=len(labeled_ds) >= self.batch_size)
        unlabeled_loader = DataLoader(unlabeled_ds, batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=len(unlabeled_ds) >= self.batch_size)

        # ---- Validation tensors (stay on CPU until scoring) ----
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        # ---- Model and optimiser ----
        self.net = CCLNet(self.input_dim, self.h_dims, self.rep_dim, C).to(device)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        milestones = [int(self.n_epochs * f) for f in (0.5, 0.75, 0.9)]
        scheduler  = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

        self._best_auc = 0.0

        for epoch in range(1, self.n_epochs + 1):
            self.net.train()
            labeled_iter   = iter(labeled_loader)
            unlabeled_iter = iter(unlabeled_loader)
            n_steps = max(len(labeled_loader), len(unlabeled_loader))

            for _ in range(n_steps):
                # ---- Fetch batches (cycle the shorter loader) ----
                try:
                    X_lb, y_lb = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    X_lb, y_lb  = next(labeled_iter)
                try:
                    (X_ub,) = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    (X_ub,)        = next(unlabeled_iter)

                X_lb = X_lb.to(device)
                y_lb = y_lb.to(device)
                X_ub = X_ub.to(device)

                # ---- Augmentations ----
                X_ub_w = _weak_aug(X_ub)    # weak augmentation
                X_ub_s = _strong_aug(X_ub)  # strong augmentation

                # ---- Forward passes ----
                ls_lb, lb_lb, _, proj_lb = self.net(X_lb)
                ls_uw, lb_uw, _, proj_uw = self.net(X_ub_w)
                _,     lb_us, _, proj_us = self.net(X_ub_s)

                # ---- Pseudo-labels & energy (no-grad) ----
                with torch.no_grad():
                    log_pi_u = torch.log(pi_u_t.clamp(min=_EPS))

                    # P̂_u: logit-adjusted balanced-branch probs on weak aug (Eq. 13)
                    P_u_w = F.softmax(lb_uw + self.tau_logit * log_pi_u, dim=1)

                    # Energy score  E(x) = −T · log Σ_k exp(f_k / T)
                    energy = -self.energy_T * torch.logsumexp(
                        lb_uw / self.energy_T, dim=1)   # (n_ub,)

                    # Reliable mask for unlabeled samples
                    if self.energy_zeta is not None:
                        reliable = energy <= self.energy_zeta
                    else:
                        reliable = torch.ones(
                            len(X_ub), dtype=torch.bool, device=device)

                    # EMA update of π̂^u (Section 3.2, after Eq. 14)
                    if reliable.any():
                        pi_u_update = P_u_w[reliable].mean(0)
                        pi_u_t = ((1.0 - self.ema_alpha) * pi_u_t
                                  + self.ema_alpha * pi_u_update)
                        pi_u_t = (pi_u_t / pi_u_t.sum()).detach()

                    # Dual-branch fusion for soft pseudo-labels (Eq. 15)
                    # π̂* = π̂^u / (π^l + π̂^u)
                    pi_star = pi_u_t / (pi_l_t + pi_u_t).clamp(min=_EPS)
                    pi_star = pi_star / pi_star.sum()
                    P_s_w   = F.softmax(ls_uw, dim=1)
                    P_s_adj = P_s_w * pi_star
                    P_s_adj = (P_s_adj
                                / P_s_adj.sum(dim=1, keepdim=True).clamp(min=_EPS))
                    P_cls_u = 0.5 * P_u_w + 0.5 * P_s_adj   # (n_ub, C)

                # ---- L_cls: Balanced FixMatch (Section 3.2) ----
                log_pi_u = torch.log(pi_u_t.clamp(min=_EPS))
                log_pi_l = torch.log(pi_l_t.clamp(min=_EPS))

                # Standard branch on labeled — plain CE (no logit adj)
                L_std = F.cross_entropy(ls_lb, y_lb)

                # Balanced branch on labeled — logit-adjusted CE (Eq. 12)
                L_bl = F.cross_entropy(lb_lb + self.tau_logit * log_pi_l, y_lb)

                # Balanced branch on reliable unlabeled — soft pseudo-label CE (Eq. 14)
                if reliable.any():
                    logits_us_adj = lb_us + self.tau_logit * log_pi_u
                    L_bu = -(
                        P_cls_u[reliable].detach()
                        * F.log_softmax(logits_us_adj[reliable], dim=1)
                    ).sum(dim=1).mean()
                else:
                    L_bu = torch.zeros([], device=device)

                L_cls = L_std + L_bl + L_bu

                # ---- L_rpl: reliable pseudo-label contrastive loss (Eq. 17) ----
                L_rpl = self._L_rpl(
                    proj_lb, y_lb, proj_uw, P_cls_u, reliable, pi_u_t, C, device)

                # ---- L_spl: smoothed pseudo-label contrastive loss (Eq. 18-21) ----
                L_spl = self._L_spl(
                    proj_lb, y_lb, proj_uw, proj_us, pi_u_t, C, device)

                # ---- Total loss (Eq. 22) ----
                loss = (self.lambda1 * L_cls
                        + (1.0 - self.lambda1) * L_rpl
                        + self.lambda2 * L_spl)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            # ---- Periodic validation ----
            if epoch % self.eval_period == 0 or epoch == self.n_epochs:
                self.pi_u = pi_u_t
                val_auc = self._eval_auc(X_val_t, val_y, device)
                if val_auc > self._best_auc:
                    self._best_auc = val_auc
                    self._best_net_state = {
                        k: v.cpu().clone()
                        for k, v in self.net.state_dict().items()
                    }
                    self._best_pi_u = pi_u_t.cpu().clone()
                print(f"  Epoch {epoch:4d}  val_AUC={val_auc:.4f}"
                      f"  best={self._best_auc:.4f}")

        # Restore best checkpoint
        if self._best_net_state is not None:
            self.net.load_state_dict(self._best_net_state)
        self.pi_u = (self._best_pi_u.to(device)
                     if self._best_pi_u is not None else pi_u_t)

        return self._best_auc

    # ------------------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return anomaly scores: 1 − P(normal | x) from balanced branch."""
        assert self.net is not None, "Call fit() first."
        device = torch.device(self.device)
        self.net.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32)
        return self._score_tensor(X_t, device)

    def predict_labels(self, X_test: np.ndarray) -> np.ndarray:
        """Return predicted class indices from logit-adjusted balanced branch."""
        assert self.net is not None, "Call fit() first."
        device = torch.device(self.device)
        self.net.eval()
        loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
            batch_size=512, shuffle=False)
        preds = []
        with torch.no_grad():
            log_pi_u = torch.log(self.pi_u.to(device).clamp(min=_EPS))
            for (x,) in loader:
                _, logits_b, _, _ = self.net(x.to(device))
                preds.append(
                    (logits_b + self.tau_logit * log_pi_u)
                    .argmax(dim=1).cpu().numpy()
                )
        return np.concatenate(preds)

    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        assert self.net is not None
        torch.save({
            "model": self.net.state_dict(),
            "pi_u":  self.pi_u.cpu() if self.pi_u is not None else None,
            "meta": {
                "input_dim":  self.input_dim,
                "h_dims":     self.h_dims,
                "rep_dim":    self.rep_dim,
                "n_classes":  self.n_classes,
                "tau_logit":  self.tau_logit,
                "tau_c":      self.tau_c,
                "lambda1":    self.lambda1,
                "lambda2":    self.lambda2,
                "beta_spl":   self.beta_spl,
            },
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt   = torch.load(path, map_location="cpu")
        meta   = ckpt.get("meta", {})
        self.input_dim  = meta["input_dim"]
        self.h_dims     = meta.get("h_dims",    self.h_dims)
        self.rep_dim    = meta.get("rep_dim",    self.rep_dim)
        self.n_classes  = meta["n_classes"]
        self.tau_logit  = meta.get("tau_logit",  self.tau_logit)
        self.tau_c      = meta.get("tau_c",      self.tau_c)
        self.lambda1    = meta.get("lambda1",    self.lambda1)
        self.lambda2    = meta.get("lambda2",    self.lambda2)
        self.beta_spl   = meta.get("beta_spl",   self.beta_spl)
        device = torch.device(self.device)
        self.net = CCLNet(
            self.input_dim, self.h_dims, self.rep_dim, self.n_classes
        ).to(device)
        self.net.load_state_dict(ckpt["model"])
        if ckpt.get("pi_u") is not None:
            self.pi_u = ckpt["pi_u"].to(device)
        else:
            self.pi_u = torch.ones(self.n_classes, device=device) / self.n_classes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _L_rpl(self, proj_lb: torch.Tensor, y_lb: torch.Tensor,
               proj_uw: torch.Tensor, P_cls_u: torch.Tensor,
               reliable: torch.Tensor,
               pi_u_t: torch.Tensor, C: int,
               device: torch.device) -> torch.Tensor:
        """Continuous contrastive loss with reliable pseudo-labels (Eq. 17).

        Build batch B = labeled ∪ energy-filtered unlabeled, estimate the
        kernel-density class posterior via Gaussian kernel (Eq. 16 / 10),
        apply logit adjustment, and compute soft cross-entropy.
        """
        n_rel = int(reliable.sum())
        if n_rel == 0:
            return torch.zeros([], device=device)

        # L2-normalised projections
        z_lb  = F.normalize(proj_lb, dim=1)                    # (n_lb, D)
        z_rel = F.normalize(proj_uw[reliable], dim=1)          # (n_rel, D)
        z_B   = torch.cat([z_lb, z_rel], dim=0)                # (B, D)

        # Soft labels for the full batch B
        y_lb_oh  = F.one_hot(y_lb, C).float()                  # (n_lb, C)
        P_cls_rel = P_cls_u[reliable].detach()                  # (n_rel, C)
        P_cls_B   = torch.cat([y_lb_oh, P_cls_rel], dim=0)     # (B, C)

        # Gaussian kernel  S[i,j] = exp(z_i · z_j / τ_c)
        S = torch.exp(z_B @ z_B.T / self.tau_c)                # (B, B)

        # Class-weighted numerator (Eq. 16)
        w   = P_cls_B.sum(0).clamp(min=_EPS)                   # (C,)
        num = (S @ P_cls_B) / w                                  # (B, C)

        # Logit adjustment by π̂^u  (Eq. 17 denominator structure)
        num_adj = num * pi_u_t                                   # (B, C)
        P_t     = num_adj / num_adj.sum(dim=1, keepdim=True).clamp(min=_EPS)

        # Loss only for unlabeled samples in B
        P_t_rel = P_t[len(y_lb):]                               # (n_rel, C)
        return -(P_cls_rel * torch.log(P_t_rel.clamp(min=_EPS))).sum(dim=1).mean()

    # ------------------------------------------------------------------

    def _L_spl(self, proj_lb: torch.Tensor, y_lb: torch.Tensor,
               proj_uw: torch.Tensor, proj_us: torch.Tensor,
               pi_u_t: torch.Tensor, C: int,
               device: torch.device) -> torch.Tensor:
        """Continuous contrastive loss with smoothed pseudo-labels (Eq. 18-21).

        Propagate labels from labeled data to both weak- and strong-aug
        unlabeled views (Eq. 19-21), then enforce weak→strong consistency.
        """
        n_ub = proj_uw.size(0)
        if n_ub < 2:
            return torch.zeros([], device=device)

        # Labeled anchors are detached — they provide supervision, not contrastive signal
        z_lb     = F.normalize(proj_lb.detach(), dim=1)    # (n_lb, D)
        y_lb_oh  = F.one_hot(y_lb, C).float()              # (n_lb, C)
        z_uw     = F.normalize(proj_uw, dim=1)             # (n_ub, D)
        z_us     = F.normalize(proj_us, dim=1)             # (n_ub, D)

        def propagate(z_ub: torch.Tensor) -> torch.Tensor:
            """Eqs. 19-21: propagate labels from labeled anchors to unlabeled."""
            # Eq. 19: P̂(Y | X^u; B^l) — similarity to labeled per class
            S_ul = torch.exp(z_ub @ z_lb.T / self.tau_c)   # (n_ub, n_lb)
            class_sim = S_ul @ y_lb_oh                       # (n_ub, C)
            class_cnt = y_lb_oh.sum(0).clamp(min=_EPS)      # (C,)
            # Logit adjustment by π̂^u (Eq. 19 denominator)
            P_lb = (class_sim / class_cnt) * pi_u_t         # (n_ub, C)
            # Normalise to proper probability distribution
            P_lb = P_lb.clamp(min=0.0)
            P_lb = P_lb / P_lb.sum(dim=1, keepdim=True).clamp(min=_EPS)

            # Eq. 20-21: self-propagation within unlabeled batch
            S_uu = torch.exp(z_ub @ z_ub.T / self.tau_c)   # (n_ub, n_ub)
            G    = S_uu / S_uu.sum(dim=1, keepdim=True).clamp(min=_EPS)

            # Solve (I − βG) P = (1 − β) P_lb   →   P = (I − βG)^{−1} (1−β) P_lb
            I_mat = torch.eye(n_ub, device=device, dtype=z_ub.dtype)
            A     = I_mat - self.beta_spl * G               # (n_ub, n_ub)
            rhs   = (1.0 - self.beta_spl) * P_lb           # (n_ub, C)
            try:
                P_hat = torch.linalg.solve(A, rhs)
            except Exception:
                P_hat = rhs     # fallback: no unlabeled self-propagation
            P_hat = P_hat.clamp(min=_EPS)
            return P_hat / P_hat.sum(dim=1, keepdim=True).clamp(min=_EPS)

        # Weak view = teacher (detach so it provides a stable target)
        P_w = propagate(z_uw.detach())   # (n_ub, C)
        # Strong view = student (gradients flow back through the encoder)
        P_s = propagate(z_us)            # (n_ub, C)

        # Eq. 18: weak → strong consistency (cross-entropy)
        return -(P_w.detach() * torch.log(P_s.clamp(min=_EPS))).sum(dim=1).mean()

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_prior(y: np.ndarray, n_classes: int) -> np.ndarray:
        """Class frequency prior with Laplace smoothing."""
        counts = np.bincount(y, minlength=n_classes).astype(float) + _EPS
        return counts / counts.sum()

    def _eval_auc(self, X_val_t: torch.Tensor, y_val: np.ndarray,
                  device: torch.device) -> float:
        self.net.eval()
        scores = self._score_tensor(X_val_t, device)
        y_bin  = (y_val > 0).astype(int)
        try:
            return float(roc_auc_score(y_bin, scores))
        except ValueError:
            return 0.0

    def _score_tensor(self, X_t: torch.Tensor,
                      device: torch.device) -> np.ndarray:
        """Anomaly score = 1 − P(normal | x) from logit-adjusted balanced branch."""
        loader = DataLoader(TensorDataset(X_t), batch_size=512, shuffle=False)
        scores: list[np.ndarray] = []
        log_pi_u = torch.log(self.pi_u.to(device).clamp(min=_EPS))
        with torch.no_grad():
            for (x,) in loader:
                _, logits_b, _, _ = self.net(x.to(device))
                probs = F.softmax(logits_b + self.tau_logit * log_pi_u, dim=1)
                scores.append((1.0 - probs[:, 0]).cpu().numpy())
        return np.concatenate(scores)
