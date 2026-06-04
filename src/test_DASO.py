"""
Standalone DASO evaluation script.

Mirrors C:/Code/daso_timeseries/test_daso.py in structure, interface, and
output format so that results from both implementations can be compared
side-by-side to verify correctness.

Checkpoint compatibility
------------------------
Accepts checkpoints from BOTH sources:

* main_DASO.py (this project) — meta contains input_dim / h_dims / rep_dim /
  n_classes; all state-dict keys are present including proj_head.
* daso_timeseries — meta contains iter / valid/top1 / ...; architecture must
  be supplied via --input_dim / --h_dims / --rep_dim / --n_classes; proj_head
  keys are silently absent (strict=False load).

Usage:
  # checkpoint from this project (meta has arch info):
  python src/test_DASO.py --checkpoint ./saved_model/daso_checkpoint.pt --dataset ALFA

  # checkpoint from daso_timeseries (arch must be given on CLI):
  python src/test_DASO.py --checkpoint /path/to/model_best.pth.tar \\
      --dataset ALFA --input_dim 44 --h_dims 256 512 --rep_dim 64 --n_classes 7

  # evaluate EMA model:
  python src/test_DASO.py --checkpoint ./saved_model/daso_checkpoint.pt --dataset ALFA --use-ema
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from baselines.DASO import DASONet
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single
from datasets.main import load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a DASO checkpoint (from main_DASO.py or daso_timeseries)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint file (.pt or .pth.tar)")
    p.add_argument("--dataset", required=True,
                   choices=["ALFA", "Pegasus"],
                   help="Dataset to evaluate on")
    p.add_argument("--data_path", default="./data",
                   help="Root directory for data files")
    p.add_argument("--known_outlier_class", type=int, nargs="+", default=[1])
    p.add_argument("--n_known_outlier_classes", type=int, default=1)
    p.add_argument("--ratio_known_normal", type=float, default=0.2)
    p.add_argument("--ratio_known_outlier", type=float, default=0.3)
    p.add_argument("--ratio_pollution", type=float, default=0.1)
    p.add_argument("--subclasses", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-ema", action="store_true",
                   help="Load EMA model weights instead of the student model")
    # Architecture overrides — required when loading a daso_timeseries checkpoint
    # whose meta only stores {iter, valid/top1, ...} and not network dimensions.
    # Silently ignored when the checkpoint meta already contains these fields.
    p.add_argument("--input_dim", type=int, default=None,
                   help="Input feature dimension (override if not in checkpoint meta)")
    p.add_argument("--h_dims", type=int, nargs="+", default=None,
                   help="Hidden layer sizes, e.g. --h_dims 256 128")
    p.add_argument("--rep_dim", type=int, default=None,
                   help="Representation (encoder output) dimension")
    p.add_argument("--n_classes", type=int, default=None,
                   help="Number of output classes including normal (class 0)")
    return p.parse_args()


def _resolve_arch(state, key, args):
    """Return (input_dim, h_dims, rep_dim, n_classes) for network construction.

    Priority: checkpoint meta > CLI args > inferred from weight shapes.
    Raises ValueError if any value cannot be determined.
    """
    meta = state.get("meta") or {}

    input_dim = meta.get("input_dim") or args.input_dim
    h_dims    = meta.get("h_dims")    or args.h_dims
    rep_dim   = meta.get("rep_dim")   or args.rep_dim
    n_classes = meta.get("n_classes") or args.n_classes

    # Last-resort: infer from weight tensors (works for both checkpoint formats
    # because the encoder.encoder.* key structure is now shared).
    sd = state[key]
    if input_dim is None:
        # encoder.encoder.0 is the first Linear layer; its weight is (h1, input_dim)
        w = sd.get("encoder.encoder.0.weight")
        if w is not None:
            input_dim = w.shape[1]

    if h_dims is None:
        # Walk the Sequential indices: 0=Linear, 1=BN, 2=LeakyReLU, 3=Linear, ...
        # Each Linear at index 3k has shape (h_{k+1}, h_k); last Linear → rep_dim
        h_list = []
        idx = 0
        while True:
            w = sd.get(f"encoder.encoder.{idx}.weight")
            if w is None:
                break
            # next layer at idx+3 (Linear) or we've hit the final Linear
            next_w = sd.get(f"encoder.encoder.{idx + 3}.weight")
            if next_w is not None:
                h_list.append(w.shape[0])
            idx += 3
        if h_list:
            h_dims = h_list

    if rep_dim is None:
        # The last Linear in encoder.encoder has output size = rep_dim
        idx = 0
        last_w = None
        while True:
            w = sd.get(f"encoder.encoder.{idx}.weight")
            if w is None:
                break
            last_w = w
            idx += 3
        if last_w is not None:
            rep_dim = last_w.shape[0]

    if n_classes is None:
        w = sd.get("classifier.classifier.weight")
        if w is not None:
            n_classes = w.shape[0]

    missing = [name for name, val in [
        ("input_dim", input_dim), ("h_dims", h_dims),
        ("rep_dim", rep_dim), ("n_classes", n_classes)
    ] if val is None]
    if missing:
        raise ValueError(
            f"Cannot determine architecture parameter(s): {missing}. "
            "Pass them explicitly via --input_dim / --h_dims / --rep_dim / --n_classes."
        )

    return input_dim, h_dims, rep_dim, n_classes


def main():
    args = parse_args()

    # Seed BEFORE dataset creation so the global numpy state is identical to
    # main_DASO.py, which also calls np.random.seed(args.seed) before
    # load_dataset.  ALFA.py and Pegasus.py use np.random.choice (global state)
    # for their val/test split; without matching the seed here, test_DASO.py
    # would load a different test set than main_DASO.py and all metrics would
    # differ even for the same checkpoint.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint ----
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)

    key = "ema_model" if args.use_ema else "model"
    if state.get(key) is None:
        raise ValueError(f"Checkpoint key '{key}' is None. "
                         "Try without --use-ema or vice versa.")

    # ---- Determine architecture ----
    input_dim, h_dims, rep_dim, n_classes = _resolve_arch(state, key, args)

    # ---- Rebuild network ----
    model = DASONet(
        input_dim=input_dim,
        h_dims=h_dims,
        rep_dim=rep_dim,
        n_classes=n_classes,
    ).to(device)

    # strict=False: proj_head keys are absent in daso_timeseries checkpoints;
    # they are unused at inference time (forward with is_train=False).
    missing_keys, unexpected_keys = model.load_state_dict(state[key], strict=False)
    if unexpected_keys:
        print(f"  [warn] Unexpected keys in checkpoint: {unexpected_keys[:5]}")
    non_proj_missing = [k for k in missing_keys if not k.startswith("proj_head")]
    if non_proj_missing:
        raise ValueError(f"Missing required model keys: {non_proj_missing[:5]}")

    model.eval()

    # ---- Load test data ----
    known_outlier_class = tuple(args.known_outlier_class)
    dataset = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        normal_class=0,
        known_outlier_class=known_outlier_class,
        n_known_outlier_classes=args.n_known_outlier_classes,
        ratio_known_normal=args.ratio_known_normal,
        ratio_known_outlier=args.ratio_known_outlier,
        ratio_pollution=args.ratio_pollution,
        random_state=np.random.RandomState(args.seed),
        subclasses=args.subclasses,
    )

    test_loader = DataLoader(
        dataset.test_set, batch_size=512,
        shuffle=False, drop_last=False,
    )

    # ---- Run inference ----
    all_targets, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            sample, target, *_ = batch
            sample = sample.to(device)
            # target is multi-hot; convert to single-class integer label for metrics
            y_single = truth_multihot_to_single(target.numpy())

            logits = model(sample, is_train=False)
            probs = F.softmax(logits, dim=1)

            all_targets.append(y_single)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_scores.append((1.0 - probs[:, 0]).cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_score = np.concatenate(all_scores)

    # ---- Compute metrics (identical to daso_timeseries/lib/utils/anomaly_metrics.py) ----
    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    # ---- Print results (identical format to daso_timeseries/test_daso.py) ----
    width = 31
    print("=" * width)
    print("    DASO Test Results")
    print("=" * width)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Model key   : {key}")
    print(f"  Test samples: {len(y_true)}")
    print("-" * width)
    print(f"  AUC            : {stats['auc']:.4f}")
    print(f"  F1 (macro)     : {stats['f1_macro']:.4f}")
    print(f"  F1 (weighted)  : {stats['f1_weighted']:.4f}")
    print(f"  Accuracy       : {stats['accuracy']:.4f}")
    print(f"  Anomaly Recall : {stats['anomaly_recall']:.4f}")
    print("=" * width)


if __name__ == "__main__":
    main()
