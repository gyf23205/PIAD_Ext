# PIAD: Physics-Informed Anomaly Detection for Unmanned Aerial Vehicles

This repository provides a [PyTorch](https://pytorch.org/) implementation of the **PIPD (Physics-informed Prediction for Detection)** framework presented in our IEEE RA-L paper:

> **"Physics-informed Anomaly Detection for Unmanned Aerial Vehicles"**
> Yifan Guo, Kartik A. Pant, Inseok Hwang — *IEEE Robotics and Automation Letters*, 2025

---

## PIPDall Method

The entry point is `src/main_all.py`. PIPDall is a **multi-label semi-supervised anomaly detector** that fuses a hypersphere-based anomaly detection objective with a physics-informed prediction branch.

<!-- FIGURE: Overall framework diagram showing the encoder, prediction branch, and hypersphere detection boundary. -->

### Network Architecture

`MLP_Physical` takes a flattened sensor window `(batch, x_dim)` and produces two outputs:

```
Input ──► Encoder (MLP + BN + LeakyReLU) ──► z  (batch, rep_dim=64)
                                               │
                                               └──► DecoderSimple ──► ŝ_{t+1}  (batch, 44)
```

- **Encoder**: two hidden layers (256 → 512 → 64) with BatchNorm and LeakyReLU, no bias.
- **Predictor** (`DecoderSimple`): a single linear layer `rep_dim → x_dim` followed by Sigmoid, predicting all 44 sensor entries at the next timestep.

<!-- FIGURE: Detailed network architecture diagram of MLP_Physical, showing the encoder layers, representation z, and the DecoderSimple prediction head. -->

### Training Objective

The total loss combines four terms:

```
L = λ_sad · L_sad  +  λ_pred · L_pred  +  λ_dir · L_dir  +  λ_cluster · L_cluster
```

with default coefficients `λ_sad=1.0`, `λ_pred=4.8`, `λ_dir=5.0`, `λ_cluster=1.7`.

**L_sad — Deep SAD hypersphere loss**

Each sample's squared distance to the normal centroid `c_normal` in representation space:
- Labeled normals: minimize `‖z − c_normal‖²`
- Labeled anomalies: minimize `1 / (‖z − c_normal‖² + ε)`, i.e., maximize distance

**L_pred — Physics prediction loss**

Mean squared error between the predicted next sensor state `ŝ_{t+1}` and the ground-truth next state `s_{t+1}`:

```
L_pred = MSE(ŝ_{t+1}, s_{t+1})
```

This forces the encoder to capture system dynamics, making the representation sensitive to physically anomalous behavior.

**L_dir — Multi-label directional contrastive loss**

Applied only to labeled samples. Pairwise cosine similarities are computed in representation space and passed through a contrastive loss with temperature τ=0.5:

```
same_mask[i,j] = True  if both normal, or if samples i and j share ≥1 active anomaly class
loss_dir = logsumexp(A/τ, non-self) − mean(A/τ, positives)
```

This pulls same-class samples together and pushes different-class samples apart.

<!-- FIGURE: 2D t-SNE visualization of the representation space showing per-class cluster separation achieved by L_dir and L_cluster. -->

**L_cluster — Soft clustering loss**

Maintains per-class centroids updated every epoch:
- Labeled normals → pulled toward `c_normal`
- Labeled anomalies → each pulled toward its class-specific centroid `c_outlier_k`
- Unlabeled samples → pulled toward their nearest centroid (soft assignment)

### Semi-Supervised Setup

Each sample carries a **multi-hot** `semi_target` vector of shape `(n_anomaly_classes,)`:
- All `-1`: unlabeled
- All `0`: labeled normal
- Has `1`s: labeled anomaly (one or more classes active simultaneously)

Training uses `ratio_known_normal=0.2`, `ratio_known_outlier=0.3`, `ratio_pollution=0.1` by default. Known outlier classes are `[1, 3, 4, 6]` (one subtype per anomaly kind).

### Data Augmentation

Augmentation is applied each batch to labeled samples only (`aug_mode='gaussian'` by default):

- **Gaussian**: adds i.i.d. Gaussian noise (σ=0.05) to both the input window and `signal_next`, duplicating each labeled sample twice.
- **NNGMix**: builds KD-trees over the training set; mixes each labeled anomaly with a nearest neighbor (normal or anomaly) using a Beta(0.2, 0.2) coefficient λ. The same λ is applied to both the feature and `signal_next` to preserve physics consistency.
- **Both**: Gaussian first, then NNGMix on the result.

<!-- FIGURE: Illustration of the NNGMix augmentation strategy — showing an anomaly sample being mixed with a KD-tree nearest neighbor in both feature and signal_next space. -->

### Detection

At inference, the anomaly score for a sample is:

```
score = ‖z − c_normal‖²
```

A binary threshold is selected via Youden's index on the validation ROC curve. For multi-label classification, per-class thresholds are calibrated on each known anomaly class using distance to its class centroid `c_outlier_k`.

<!-- FIGURE: ROC curve on the test set and the per-class threshold calibration results from the validation split. -->

### Training Schedule

- **Optimizer**: Adam, lr=1e-4, weight decay=5×10⁻⁷
- **Epochs**: 1000, with MultiStepLR decay at epochs 200, 400, 600, 800 (γ=0.1)
- **Batch size**: 128
- **Validation**: every 100 epochs; best model selected by validation AUC
- **Gradient clipping**: max norm 1.0

<!-- FIGURE: Training curves — total loss and each component (L_sad, L_pred, L_dir, L_cluster) over 1000 epochs, along with validation AUC at each checkpoint. -->

<!-- FIGURE: Loss landscape comparison between PIPDall (physics-informed) and the Deep SAD baseline, showing the smoothing effect of the prediction branch. -->

---

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python src/main_all.py      # PIPDall (proposed)
python src/main_res.py      # PIPDres (residual variant)
python src/main_SAD.py      # Deep SAD baseline
```

## Citation

```bibtex
@article{guo2025physics,
  title={Physics-informed Anomaly Detection for Unmanned Aerial Vehicles},
  author={Guo, Yifan and Pant, Kartik A and Hwang, Inseok},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```
