"""
Standalone CCL evaluation script.

Loads a checkpoint saved by main_CCL.py and evaluates it on the test split of
a given dataset.  Architecture and hyperparameters are read from the checkpoint
meta block, so no architecture flags are needed.

Usage:
  python src/test_CCL.py --checkpoint ./saved_model/ccl_checkpoint.pt --dataset ALFA
  python src/test_CCL.py --checkpoint ./saved_model/ccl_checkpoint.pt --dataset Pegasus --seed 0
"""

import argparse
import os
import sys

import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))

from baselines.CCL import CCLTrainer
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single
from utils.data import extract_numpy
from datasets.main import load_dataset
from main_CCL import DATASET_CONFIGS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a CCL checkpoint produced by main_CCL.py"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS),
                   help="Dataset to evaluate on (must match the training dataset)")
    p.add_argument("--data_path", default="./data",
                   help="Root directory for data files")
    p.add_argument("--known_outlier_class", type=int, nargs="+", default=None,
                   help="Known anomaly class indices (overrides dataset default)")
    p.add_argument("--ratio_known_normal",  type=float, default=None)
    p.add_argument("--ratio_known_outlier", type=float, default=None)
    p.add_argument("--ratio_pollution",     type=float, default=None)
    p.add_argument("--seed", type=int, default=42,
                   help="Must match the seed used during training for reproducible splits")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Seed BEFORE dataset creation — ensures the same val/test split as training
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    defaults = DATASET_CONFIGS[args.dataset]

    known_outlier_class = tuple(
        args.known_outlier_class
        if args.known_outlier_class is not None
        else defaults["known_outlier_classes"]
    )
    ratio_known_normal  = args.ratio_known_normal  or defaults["ratio_known_normal"]
    ratio_known_outlier = args.ratio_known_outlier or defaults["ratio_known_outlier"]
    ratio_pollution     = args.ratio_pollution     or defaults["ratio_pollution"]

    # ---- Load dataset ----
    dataset = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        normal_class=defaults["normal_class"],
        known_outlier_class=known_outlier_class,
        n_known_outlier_classes=defaults["n_known_outlier_classes"],
        ratio_known_normal=ratio_known_normal,
        ratio_known_outlier=ratio_known_outlier,
        ratio_pollution=ratio_pollution,
        random_state=np.random.RandomState(args.seed),
        subclasses=defaults.get("subclasses", True),
    )

    X_test, y_test, _ = extract_numpy(dataset.test_set)
    y_true = truth_multihot_to_single(y_test)

    # ---- Reconstruct trainer from checkpoint ----
    trainer = CCLTrainer({"device": device.type})
    trainer.load_checkpoint(args.checkpoint)

    # ---- Run inference ----
    y_score = trainer.predict(X_test)
    y_pred  = trainer.predict_labels(X_test)

    # ---- Metrics ----
    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    width = 40
    print("=" * width)
    print("    CCL Test Results")
    print("=" * width)
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Dataset      : {args.dataset}")
    print(f"  Test samples : {len(y_true)}")
    print("-" * width)
    print(f"  AUC            : {stats['auc']:.4f}")
    print(f"  F1 (macro)     : {stats['f1_macro']:.4f}")
    print(f"  F1 (weighted)  : {stats['f1_weighted']:.4f}")
    print(f"  Accuracy       : {stats['accuracy']:.4f}")
    print(f"  Anomaly Recall : {stats['anomaly_recall']:.4f}")
    print("=" * width)


if __name__ == "__main__":
    main()
