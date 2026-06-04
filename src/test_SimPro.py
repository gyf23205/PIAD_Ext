"""
Standalone SimPro evaluation script.

Mirrors test_DASO.py in structure and output format so that results from
different baselines can be compared side-by-side.

Usage:
  python src/test_SimPro.py --checkpoint ./saved_model/simpro_checkpoint.pt --dataset ALFA

  # Override architecture if meta is absent (uncommon):
  python src/test_SimPro.py --checkpoint ./saved_model/simpro_checkpoint.pt \\
      --dataset ALFA --input_dim 875 --h_dims 256 128 --rep_dim 64 --n_classes 7
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from baselines.SimPro import SimProNet
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single
from datasets.main import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a SimPro checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint file (.pt)")
    p.add_argument("--dataset", required=True,
                   choices=["ALFA", "Pegasus"],
                   help="Dataset to evaluate on")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--known_outlier_class", type=int, nargs="+", default=[1])
    p.add_argument("--ratio_known_normal",  type=float, default=0.2)
    p.add_argument("--ratio_known_outlier", type=float, default=0.3)
    p.add_argument("--ratio_pollution",     type=float, default=0.1)
    p.add_argument("--subclasses", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    # Architecture overrides (only needed if checkpoint meta is missing)
    p.add_argument("--input_dim", type=int, default=None)
    p.add_argument("--h_dims",    type=int, nargs="+", default=None)
    p.add_argument("--rep_dim",   type=int, default=None)
    p.add_argument("--n_classes", type=int, default=None)
    p.add_argument("--tau",       type=float, default=None)
    return p.parse_args()


def _resolve_arch(state: dict, args) -> tuple:
    """Return (input_dim, h_dims, rep_dim, n_classes, tau) from checkpoint meta or CLI."""
    meta = state.get("meta") or {}

    input_dim = meta.get("input_dim") or args.input_dim
    h_dims    = meta.get("h_dims")    or args.h_dims
    rep_dim   = meta.get("rep_dim")   or args.rep_dim
    n_classes = meta.get("n_classes") or args.n_classes
    tau       = meta.get("tau")       or args.tau or 1.0

    # Last-resort inference from weight shapes
    sd = state.get("model", {})
    if input_dim is None:
        w = sd.get("encoder.0.weight")
        if w is not None:
            input_dim = w.shape[1]

    if rep_dim is None:
        # Last Linear in encoder has shape (rep_dim, h_last)
        idx, last_w = 0, None
        while True:
            w = sd.get(f"encoder.{idx}.weight")
            if w is None:
                break
            last_w = w
            idx += 3
        if last_w is not None:
            rep_dim = last_w.shape[0]

    if n_classes is None:
        w = sd.get("classifier.weight")
        if w is not None:
            n_classes = w.shape[0]

    missing = [name for name, val in [
        ("input_dim", input_dim), ("h_dims", h_dims),
        ("rep_dim", rep_dim),     ("n_classes", n_classes),
    ] if val is None]
    if missing:
        raise ValueError(
            f"Cannot determine architecture parameter(s): {missing}. "
            "Pass them via --input_dim / --h_dims / --rep_dim / --n_classes."
        )

    return input_dim, h_dims, rep_dim, n_classes, tau


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load checkpoint ----
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)

    input_dim, h_dims, rep_dim, n_classes, tau = _resolve_arch(state, args)

    # ---- Rebuild network ----
    model = SimProNet(
        input_dim=input_dim,
        h_dims=h_dims,
        rep_dim=rep_dim,
        n_classes=n_classes,
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # ---- Restore Bayes prior φ ----
    phi = state.get("phi")
    if phi is None:
        print("[warn] φ not found in checkpoint; using uniform prior.")
        phi = torch.ones(n_classes) / n_classes
    phi = phi.to(device)
    adj = tau * torch.log(phi + 1e-12)

    # ---- Load test data ----
    known_outlier_class = tuple(args.known_outlier_class)
    dataset = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        normal_class=0,
        known_outlier_class=known_outlier_class,
        n_known_outlier_classes=len(known_outlier_class),
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
            y_single = truth_multihot_to_single(target.numpy())

            logits = model(sample)
            probs  = F.softmax(logits + adj, dim=1)

            all_targets.append(y_single)
            all_preds.append((logits + adj).argmax(dim=1).cpu().numpy())
            all_scores.append((1.0 - probs[:, 0]).cpu().numpy())

    y_true  = np.concatenate(all_targets)
    y_pred  = np.concatenate(all_preds)
    y_score = np.concatenate(all_scores)

    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    width = 35
    print("=" * width)
    print("    SimPro Test Results")
    print("=" * width)
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Dataset     : {args.dataset}")
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
