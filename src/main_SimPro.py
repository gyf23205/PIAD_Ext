"""
Entry point for the SimPro semi-supervised baseline.

Usage examples:
  python src/main_SimPro.py --dataset ALFA
  python src/main_SimPro.py --dataset Pegasus --n_known_outlier_classes 2 --known_outlier_class 1 2
  python src/main_SimPro.py --dataset ALFA --tau 2.0 --threshold 0.95
  python src/main_SimPro.py --dataset ALFA --save_path ./saved_model/simpro_alfa.pt
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from baselines.SimPro import SimProTrainer
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single
from utils.data import extract_numpy
from datasets.main import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="SimPro semi-supervised baseline for PIAD_Ext")
    # Dataset
    p.add_argument("--dataset", default="ALFA",
                   choices=["ALFA", "Pegasus"],
                   help="Dataset name")
    p.add_argument("--data_path", default="./data",
                   help="Root directory for data files")
    p.add_argument("--known_outlier_class", type=int, nargs="+", default=[1],
                   help="Known anomaly class index/indices (e.g. 1 or 1 2 3)")
    p.add_argument("--ratio_known_normal", type=float, default=0.2,
                   help="Fraction of normal training samples that are labeled")
    p.add_argument("--ratio_known_outlier", type=float, default=0.3,
                   help="Fraction of known-anomaly training samples that are labeled")
    p.add_argument("--ratio_pollution", type=float, default=0.1,
                   help="Contamination rate in unlabeled pool")
    p.add_argument("--subclasses", action="store_true", default=True,
                   help="Use fine-grained subclasses (ALFA only)")
    p.add_argument("--seed", type=int, default=42)
    # Model
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--rep_dim", type=int, default=64)
    p.add_argument("--h_dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--eval_period", type=int, default=10)
    # SimPro-specific
    p.add_argument("--tau", type=float, default=1.0,
                   help="Logit-adjustment temperature τ (paper: 1.0 for CIFAR-100 / 2.0 for CIFAR-10)")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Confidence threshold for pseudo-label filtering")
    p.add_argument("--ema_u", type=float, default=0.9,
                   help="EMA decay for π_u and φ distribution updates")
    # Output
    p.add_argument("--save_path", default="./saved_model/simpro_checkpoint.pt",
                   help="Where to save the trained checkpoint")
    p.add_argument("--no_save", action="store_true",
                   help="Skip saving the checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logger = logging.getLogger()

    logger.info(f"Loading dataset: {args.dataset}")
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

    logger.info("Extracting training / validation / test arrays …")
    X_train, _,     semi_y = extract_numpy(dataset.train_set)
    X_val,   y_val, _      = extract_numpy(dataset.val_set)
    X_test,  y_test, _     = extract_numpy(dataset.test_set)

    logger.info(f"  Train subset : {X_train.shape}")
    logger.info(f"  Val          : {X_val.shape}")
    logger.info(f"  Test         : {X_test.shape}")

    cfg = {
        "n_epochs":    args.n_epochs,
        "lr":          args.lr,
        "batch_size":  args.batch_size,
        "h_dims":      args.h_dims,
        "rep_dim":     args.rep_dim,
        "tau":         args.tau,
        "threshold":   args.threshold,
        "ema_u":       args.ema_u,
        "eval_period": args.eval_period,
    }
    trainer = SimProTrainer(cfg)

    logger.info("Training …")
    t0 = time.time()
    best_auc = trainer.fit(X_train, semi_y, X_val, y_val)
    train_time = time.time() - t0
    logger.info(f"Training done in {train_time:.1f}s  (best val AUC = {best_auc:.4f})")

    if not args.no_save:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        trainer.save_checkpoint(args.save_path)

    logger.info("Evaluating on test set …")
    t1 = time.time()
    y_score = trainer.predict(X_test)
    y_pred  = trainer.predict_labels(X_test)
    y_true  = truth_multihot_to_single(y_test)
    test_time = time.time() - t1

    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    width = 35
    print("=" * width)
    print("    SimPro Test Results")
    print("=" * width)
    print(f"  Dataset     : {args.dataset}")
    print(f"  Test samples: {len(y_true)}")
    print(f"  Train time  : {train_time:.1f}s")
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
