"""
SimAD baseline adapter for PIAD_Ext.

Wraps SimAD (Simple Dissimilarity-Based Anomaly Detection, IEEE TNNLS 2025) in
the same Trainer interface used by CATS and DASO so it can be trained and
evaluated on ALFA / Pegasus / spoofing datasets.

Architecture
------------
  FeatureExtractor  → InstanceNorm + sinusoidal PE + patching + value embedding
  EmbedPatchEncoder → L layers of multi-head attention where V comes from
                      learnable patch embeddings E (not from the input)
  ContrastFusionHead → mean-pool + 2-layer MLP projection (training only)

Semi-supervised adaptation
--------------------------
SimAD is unsupervised; it trains on normal data only.  Rows where any column of
semi_y > 0 (labeled anomalies) are excluded before training.  Unlabeled (-1)
and labeled-normal (0) rows are both used.
"""

import copy
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding, added to (B, T, C) after InstanceNorm."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """
    (B, T, C) → (B, M, D) patch embeddings.

    Steps:
      1. InstanceNorm1d (per-sample normalisation along T)
      2. Sinusoidal positional encoding
      3. Patching: fold into (B, M, P*C) where T = M * patch_size
      4. Value embedding: LN → Linear(P*C, D) → LN
    """

    def __init__(self, win_size: int, n_features: int, patch_size: int, d_model: int):
        super().__init__()
        assert win_size % patch_size == 0, "win_size must be divisible by patch_size"
        self.win_size   = win_size
        self.n_features = n_features
        self.patch_size = patch_size
        self.n_patches  = win_size // patch_size   # M
        patch_dim       = patch_size * n_features  # P * C

        self.instance_norm = nn.InstanceNorm1d(win_size, affine=False)
        self.pos_enc       = SinusoidalPE(win_size, n_features)
        self.value_emb     = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        x = self.instance_norm(x)           # (B, T, C)
        x = self.pos_enc(x)                 # (B, T, C)
        # Patching: (B, M, P*C)
        x = x.reshape(B, self.n_patches, self.patch_size * C)
        return self.value_emb(x)            # (B, M, D)


# ---------------------------------------------------------------------------
# EmbedPatch attention layer
# ---------------------------------------------------------------------------

class EmbedPatchLayer(nn.Module):
    """
    Single Transformer layer where the Value matrix derives from learnable
    patch embeddings E rather than from the input N.

    Q = N W_Q,  K = N W_K
    V = W_V @ E   (W_V: (U, M, V_dim),  E: (U, V_dim, d))  → (B, U, M, d)
    Z = Softmax(QK^T / sqrt(d)) V
    out = LayerNorm(N + linear(Z)) → LayerNorm(· + FFN(·))
    """

    def __init__(self, d_model: int, n_heads: int, n_patches: int, n_patch_emb: int,
                 ffn_mult: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads   = n_heads
        self.n_patches = n_patches
        self.head_dim  = d_model // n_heads
        self.scale     = self.head_dim ** -0.5

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable patch embeddings: E ∈ (U, V_dim, d) and W_V ∈ (U, M, V_dim)
        self.E   = nn.Parameter(torch.randn(n_heads, n_patch_emb, self.head_dim) * 0.02)
        self.W_V = nn.Parameter(torch.randn(n_heads, n_patches, n_patch_emb) * 0.02)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, N: torch.Tensor) -> torch.Tensor:
        B, M, D = N.shape
        U, h = self.n_heads, self.head_dim

        Q = self.W_Q(N).view(B, M, U, h).permute(0, 2, 1, 3)  # (B, U, M, h)
        K = self.W_K(N).view(B, M, U, h).permute(0, 2, 1, 3)  # (B, U, M, h)

        # V from learnable embeddings: (U, M, h)
        V = torch.einsum('umv,uvd->umd', self.W_V, self.E)     # (U, M, h)
        V = V.unsqueeze(0).expand(B, -1, -1, -1)               # (B, U, M, h)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale   # (B, U, M, M)
        attn = F.softmax(attn, dim=-1)
        Z = torch.matmul(attn, V)                                   # (B, U, M, h)
        Z = Z.permute(0, 2, 1, 3).reshape(B, M, D)                 # (B, M, D)
        Z = self.out_proj(Z)

        N = self.norm1(N + Z)
        N = self.norm2(N + self.ffn(N))
        return N


# ---------------------------------------------------------------------------
# EmbedPatch encoder (stack of L layers)
# ---------------------------------------------------------------------------

class EmbedPatchEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 n_patches: int, n_patch_emb: int):
        super().__init__()
        self.layers = nn.ModuleList([
            EmbedPatchLayer(d_model, n_heads, n_patches, n_patch_emb)
            for _ in range(n_layers)
        ])

    def forward(self, N: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            N = layer(N)
        return N


# ---------------------------------------------------------------------------
# Contrastive projection head (training only)
# ---------------------------------------------------------------------------

class ContrastFusionHead(nn.Module):
    """Mean-pool encoder output → 2-layer MLP projection."""

    def __init__(self, d_model: int, proj_dim: int | None = None):
        super().__init__()
        proj_dim = proj_dim or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, N: torch.Tensor) -> torch.Tensor:
        # N: (B, M, D) → mean-pool → (B, D) → projection
        return self.mlp(N.mean(dim=1))


# ---------------------------------------------------------------------------
# Full SimAD model
# ---------------------------------------------------------------------------

class SimADModel(nn.Module):
    def __init__(self, win_size: int, n_features: int, patch_size: int,
                 d_model: int, n_heads: int, n_layers: int, n_patch_emb: int,
                 proj_dim: int | None = None):
        super().__init__()
        n_patches   = win_size // patch_size
        patch_dim   = patch_size * n_features

        self.feature_extractor  = FeatureExtractor(win_size, n_features, patch_size, d_model)
        self.encoder            = EmbedPatchEncoder(d_model, n_heads, n_layers,
                                                    n_patches, n_patch_emb)
        self.output_linear      = nn.Linear(d_model, patch_dim)
        self.contrast_head      = ContrastFusionHead(d_model, proj_dim or d_model)

        self.patch_size  = patch_size
        self.n_patches   = n_patches
        self.n_features  = n_features

    def forward(self, x: torch.Tensor, return_proj: bool = True):
        """
        x: (B, T, C)
        Returns: (x_hat_patches, proj_h) if return_proj else (x_hat_patches, None)
          x_hat_patches: (B, M, P*C)
          proj_h:        (B, proj_dim)
        """
        N     = self.feature_extractor(x)     # (B, M, D)
        N     = self.encoder(N)               # (B, M, D)
        x_hat = self.output_linear(N)         # (B, M, P*C)
        proj  = self.contrast_head(N) if return_proj else None
        return x_hat, proj


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _reconstruction_loss(x_hat: torch.Tensor, x_patch: torch.Tensor) -> torch.Tensor:
    """MSE + (1 - cosine similarity) per patch, averaged."""
    mse  = F.mse_loss(x_hat, x_patch)
    cos  = F.cosine_similarity(x_hat, x_patch, dim=-1)  # (B, M)
    return mse + (1.0 - cos).mean()


def _contrastive_loss(h_pos: torch.Tensor, h_neg: torch.Tensor) -> torch.Tensor:
    """Symmetric MSE + cosine similarity loss (stop-gradient on target)."""
    sg_neg = h_neg.detach()
    sg_pos = h_pos.detach()
    mse1   = F.mse_loss(h_pos, sg_neg)
    mse2   = F.mse_loss(h_neg, sg_pos)
    cos1   = 1.0 - F.cosine_similarity(h_pos, sg_neg, dim=-1).mean()
    cos2   = 1.0 - F.cosine_similarity(h_neg, sg_pos, dim=-1).mean()
    return mse1 + cos1 + mse2 + cos2


# ---------------------------------------------------------------------------
# SimADTrainer
# ---------------------------------------------------------------------------

class SimADTrainer:
    """Semi-supervised wrapper around SimAD for PIAD_Ext datasets.

    Config keys
    -----------
    win_size        : int   — temporal window length
    n_features      : int   — features per timestep
    patch_size      : int   — P (must divide win_size)
    d_model         : int   — transformer hidden dim     (default: 128)
    n_heads         : int   — attention heads            (default: 4)
    n_layers        : int   — encoder depth              (default: 3)
    n_patch_emb     : int   — V (patch embedding count)  (default: 50)
    proj_dim        : int   — contrastive proj dim       (default: d_model)
    noise_level     : float — Gaussian noise std for negatives (default: 0.3)
    beta_max        : float — max contrastive loss weight       (default: 0.1)
    n_warmup_epochs : int   — epochs to ramp beta               (default: 20)
    lr              : float — Adam lr                           (default: 1e-3)
    n_epochs        : int   — max epochs                        (default: 100)
    batch_size      : int   — mini-batch size                   (default: 256)
    weight_decay    : float — Adam weight decay                 (default: 1e-5)
    patience        : int   — early-stopping patience           (default: 30)
    device          : str   — 'cuda' or 'cpu'
    """

    def __init__(self, config: dict):
        self.win_size        = config['win_size']
        self.n_features      = config['n_features']
        self.patch_size      = config['patch_size']
        self.d_model         = config.get('d_model', 128)
        self.n_heads         = config.get('n_heads', 4)
        self.n_layers        = config.get('n_layers', 3)
        self.n_patch_emb     = config.get('n_patch_emb', 50)
        self.proj_dim        = config.get('proj_dim', self.d_model)
        self.noise_level     = config.get('noise_level', 0.3)
        self.beta_max        = config.get('beta_max', 0.1)
        self.n_warmup_epochs = config.get('n_warmup_epochs', 20)
        self.lr              = config.get('lr', 1e-3)
        self.n_epochs        = config.get('n_epochs', 100)
        self.batch_size      = config.get('batch_size', 256)
        self.weight_decay    = config.get('weight_decay', 1e-5)
        self.patience        = config.get('patience', 30)
        self.device          = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model: SimADModel | None = None
        self.threshold: float = 0.5
        self.best_val_auc: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshape(self, X: np.ndarray) -> np.ndarray:
        """(N, T*C) → (N, T, C)."""
        return X.reshape(-1, self.win_size, self.n_features)

    def _build_model(self) -> SimADModel:
        return SimADModel(
            win_size    = self.win_size,
            n_features  = self.n_features,
            patch_size  = self.patch_size,
            d_model     = self.d_model,
            n_heads     = self.n_heads,
            n_layers    = self.n_layers,
            n_patch_emb = self.n_patch_emb,
            proj_dim    = self.proj_dim,
        ).to(self.device)

    def _patches(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, M, P*C) ground-truth patches (for loss computation)."""
        B, T, C = x.shape
        M = T // self.patch_size
        return x.reshape(B, M, self.patch_size * C)

    def _compute_scores_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample anomaly scores (B,) from a batch tensor (B, T, C)."""
        x_hat, _ = self.model(x, return_proj=False)         # (B, M, P*C)
        x_patch  = self._patches(x)                          # (B, M, P*C)

        # MSE per patch, replicate P times, mean over T
        mse_patch  = ((x_hat - x_patch) ** 2).mean(dim=-1)  # (B, M)
        mse_per_t  = mse_patch.repeat_interleave(self.patch_size, dim=1)  # (B, T)
        mse_score  = mse_per_t.mean(dim=1)                  # (B,)

        # Cosine similarity per patch, replicate P times, mean over T
        cos_patch  = F.cosine_similarity(x_hat, x_patch, dim=-1)  # (B, M)
        cos_per_t  = cos_patch.repeat_interleave(self.patch_size, dim=1)  # (B, T)
        sim_score  = (1.0 - cos_per_t).mean(dim=1)          # (B,)

        return mse_score + sim_score                         # (B,)

    def _compute_val_auc(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        from utils.metrics import truth_multihot_to_single
        scores    = self.predict(X_val)
        y_true    = truth_multihot_to_single(y_val)
        y_true_bin = (y_true > 0).astype(int)
        try:
            return float(roc_auc_score(y_true_bin, scores))
        except ValueError:
            return 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, semi_y: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Train SimAD on normal/unlabeled X_train, validate on (X_val, y_val).

        Returns best validation AUC.
        """
        # Filter out labeled anomalies
        is_anomaly = np.any(semi_y > 0, axis=1)
        X_normal   = X_train[~is_anomaly]
        if len(X_normal) == 0:
            logger.warning('All training samples are labeled anomalies — using full train set.')
            X_normal = X_train

        logger.info(f'Training on {len(X_normal)} normal/unlabeled samples '
                    f'(excluded {is_anomaly.sum()} labeled anomalies)')

        X_r = self._reshape(X_normal)
        tensor_train = torch.tensor(X_r, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor_train),
                            batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = self._build_model()
        optimizer  = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_auc = 0.0
        patience_ctr = 0
        best_state   = None

        for epoch in range(self.n_epochs):
            beta = min((epoch + 1) / max(self.n_warmup_epochs, 1), 1.0) * self.beta_max

            self.model.train()
            epoch_loss = 0.0
            n_batches  = 0

            for (batch,) in loader:
                batch    = batch.to(self.device)                         # (B, T, C)
                x_patch  = self._patches(batch)                          # (B, M, P*C)

                # Positive view (original) — reconstruction + denoising loss
                x_hat_pos, h_pos = self.model(batch, return_proj=True)
                L_rec = _reconstruction_loss(x_hat_pos, x_patch)

                # Negative view (Gaussian noise augmentation)
                noise       = torch.randn_like(batch) * self.noise_level
                x_neg       = batch + noise
                x_hat_neg, h_neg = self.model(x_neg, return_proj=True)
                L_denoise   = _reconstruction_loss(x_hat_neg, x_patch)

                L_cont = _contrastive_loss(h_pos, h_neg)
                loss   = L_rec + L_denoise - beta * L_cont

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                logger.info(f'Epoch {epoch:4d}  loss={epoch_loss / max(n_batches,1):.4f}  beta={beta:.4f}')

            val_auc = self._compute_val_auc(X_val, y_val)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_ctr = 0
                best_state   = copy.deepcopy(self.model.state_dict())
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    logger.info(f'Early stopping at epoch {epoch}  (best val AUC = {best_val_auc:.4f})')
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Fit threshold on validation set
        val_scores = self.predict(X_val)
        from utils.metrics import truth_multihot_to_single
        y_val_true  = truth_multihot_to_single(y_val)
        y_val_bin   = (y_val_true > 0).astype(int)
        self.threshold = _best_binary_threshold(val_scores, y_val_bin)

        self.best_val_auc = best_val_auc
        logger.info(f'Training complete — best val AUC = {best_val_auc:.4f}  threshold = {self.threshold:.4f}')
        return best_val_auc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores (N,). Higher = more anomalous."""
        self.model.eval()
        X_r    = torch.tensor(self._reshape(X), dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_r), batch_size=self.batch_size, shuffle=False)
        scores = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                scores.append(self._compute_scores_tensor(batch).cpu().numpy())
        return np.concatenate(scores, axis=0)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Return 0/1 binary predictions using validation-set threshold."""
        return (self.predict(X) >= self.threshold).astype(np.int64)

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold':   self.threshold,
            'config': {
                'win_size':        self.win_size,
                'n_features':      self.n_features,
                'patch_size':      self.patch_size,
                'd_model':         self.d_model,
                'n_heads':         self.n_heads,
                'n_layers':        self.n_layers,
                'n_patch_emb':     self.n_patch_emb,
                'proj_dim':        self.proj_dim,
            },
        }, path)
        logger.info(f'Checkpoint saved to {path}')

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt['config']
        for k, v in cfg.items():
            setattr(self, k, v)
        self.model     = self._build_model()
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        self.threshold = ckpt['threshold']
        logger.info(f'Checkpoint loaded from {path}')


# ---------------------------------------------------------------------------
# Threshold helper (copied from main_CATS.py)
# ---------------------------------------------------------------------------

def _best_binary_threshold(scores: np.ndarray, y_true_bin: np.ndarray) -> float:
    """Return the score threshold that maximises binary F1."""
    best_t, best_f1 = 0.0, -1.0
    for q in np.linspace(1, 99, 99):
        t = float(np.percentile(scores, q))
        preds = (scores >= t).astype(int)
        f = f1_score(y_true_bin, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t
