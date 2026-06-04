"""
Evaluate a saved RoSAS checkpoint.

Usage:
  python src/test_RoSAS.py --checkpoint ./saved_model/rosas_checkpoint.pt --dataset ALFA
  python src/test_RoSAS.py --checkpoint ./saved_model/rosas_checkpoint.pt --dataset Pegasus
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from baselines.RoSAS import RoSAS
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single
from utils.data import extract_numpy
from datasets.main import load_dataset
from main_RoSAS import DATASET_CONFIGS


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a RoSAS checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to saved checkpoint")
    p.add_argument("--dataset", default="ALFA", choices=list(DATASET_CONFIGS))
    p.add_argument("--data_path", default="./data")
    p.add_argument("--seed", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    defaults = DATASET_CONFIGS[args.dataset]

    dataset = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        normal_class=0,
        known_outlier_class=tuple(defaults['known_outlier_classes']),
        n_known_outlier_classes=defaults['n_known_outlier_classes'],
        ratio_known_normal=defaults['ratio_known_normal'],
        ratio_known_outlier=defaults['ratio_known_outlier'],
        ratio_pollution=defaults['ratio_pollution'],
        random_state=np.random.RandomState(args.seed),
        subclasses=True,
    )

    X_test, y_test, _ = extract_numpy(dataset.test_set)

    trainer = RoSAS()
    trainer.load_checkpoint(args.checkpoint)

    t0 = time.time()
    y_score = trainer.predict(X_test)
    y_pred  = trainer.predict_labels(X_test)
    y_true  = truth_multihot_to_single(y_test)
    test_time = time.time() - t0

    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    width = 35
    print("=" * width)
    print("    RoSAS Test Results")
    print("=" * width)
    print(f"  Dataset     : {args.dataset}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Test samples: {len(y_true)}")
    print(f"  Test time   : {test_time:.3f}s")
    print("-" * width)
    print(f"  AUC            : {stats['auc']:.4f}")
    print(f"  F1 (macro)     : {stats['f1_macro']:.4f}")
    print(f"  F1 (weighted)  : {stats['f1_weighted']:.4f}")
    print(f"  Accuracy       : {stats['accuracy']:.4f}")
    print(f"  Anomaly Recall : {stats['anomaly_recall']:.4f}")
    print("=" * width)


if __name__ == "__main__":
    main()
