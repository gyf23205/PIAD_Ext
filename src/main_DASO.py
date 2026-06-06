"""
Entry point for the DASO semi-supervised baseline.

Usage examples:
  python src/main_DASO.py --dataset ALFA
  python src/main_DASO.py --dataset Pegasus --n_known_outlier_classes 2 --known_outlier_class 1 2
  python src/main_DASO.py --dataset ALFA --save_path ./saved_model/daso_alfa.pt
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import wandb
# Allow running from src/ or from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from baselines.DASO import DASOTrainer
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single
from datasets.main import load_dataset
from utils.data import extract_numpy


def parse_args():
    p = argparse.ArgumentParser(description="DASO semi-supervised baseline for PIAD_Ext")
    # Dataset
    p.add_argument("--dataset", default="ALFA",
                   choices=["ALFA", "Pegasus"],
                   help="Dataset name")
    p.add_argument("--data_path", default="./data",
                   help="Root directory for data files")
    p.add_argument("--known_outlier_class", type=int, nargs="+", default=[1],
                   help="Known anomaly class index/indices (e.g. 1 or 1 2 3)")
    p.add_argument("--n_known_outlier_classes", type=int, default=1,
                   help="Number of known outlier classes")
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
    p.add_argument("--n_epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--rep_dim", type=int, default=64)
    p.add_argument("--h_dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--pretrain_steps", type=int, default=500)
    p.add_argument("--confidence_threshold", type=float, default=0.95)
    p.add_argument("--psa_loss_weight", type=float, default=1.0)
    p.add_argument("--eval_period", type=int, default=10)
    # Output
    p.add_argument("--save_path", default="./saved_model/daso_checkpoint.pt",
                   help="Where to save the trained checkpoint")
    p.add_argument("--no_save", action="store_true",
                   help="Skip saving the checkpoint")
    return p.parse_args()


def main(ratio_pollution=None, ratio_known_outlier=None, ratio_known_normal=None, seed=None):
    args = parse_args()
    if ratio_pollution     is not None: args.ratio_pollution     = ratio_pollution
    if ratio_known_outlier is not None: args.ratio_known_outlier = ratio_known_outlier
    if ratio_known_normal  is not None: args.ratio_known_normal  = ratio_known_normal
    if seed                is not None: args.seed                = seed

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    logger = logging.getLogger()

    # ---- Load dataset ----
    logger.info(f"Loading dataset: {args.dataset}")
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

    # ---- Extract aligned numpy arrays ----
    # Use dataset.train_set (the Subset) directly so that X_train and semi_y
    # have the same number of rows.  data_direct() returns X_train (full) and
    # semi_y (subset) which have different sizes.
    logger.info("Extracting training / validation / test arrays …")
    X_train, _,     semi_y = extract_numpy(dataset.train_set)
    X_val,   y_val, _      = extract_numpy(dataset.val_set)
    X_test,  y_test, _     = extract_numpy(dataset.test_set)

    logger.info(f"  Train subset : {X_train.shape}")
    logger.info(f"  Val          : {X_val.shape}")
    logger.info(f"  Test         : {X_test.shape}")

    # ---- Build trainer ----
    cfg = {
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "h_dims": args.h_dims,
        "rep_dim": args.rep_dim,
        "pretrain_steps": args.pretrain_steps,
        "confidence_threshold": args.confidence_threshold,
        "psa_loss_weight": args.psa_loss_weight,
        "eval_period": args.eval_period,
    }
    trainer = DASOTrainer(cfg)

    # ---- Train ----
    logger.info("Training …")
    t0 = time.time()
    best_auc = trainer.fit(X_train, semi_y, X_val, y_val)
    train_time = time.time() - t0
    logger.info(f"Training done in {train_time:.1f}s  (best val AUC = {best_auc:.4f})")

    # ---- Save checkpoint ----
    if not args.no_save:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        trainer.save_checkpoint(args.save_path)

    # ---- Evaluate on test set ----
    logger.info("Evaluating on test set …")
    t1 = time.time()
    y_score = trainer.predict(X_test)
    y_pred = trainer.predict_labels(X_test)
    y_true = truth_multihot_to_single(y_test)
    test_time = time.time() - t1

    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    # ---- Print results (same table format as daso_timeseries/test_daso.py) ----
    width = 35
    print("=" * width)
    print("    DASO Test Results")
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

    wandb.log({
        'val_auc':        best_auc,
        'test_auc':       stats['auc'],
        'f1_macro':       stats['f1_macro'],
        'f1_weighted':    stats['f1_weighted'],
        'accuracy':       stats['accuracy'],
        'anomaly_recall': stats['anomaly_recall'],
    })


if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project='PIAD_Ext',
        name='DASO',
        config={
            'lr': 0.001,
            'n_epochs': 300,
            'batch_size': 64,
        }
    )
    ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
    seed = wandb.config.seed
    main(ratio_pollution=ratio_pollution, ratio_known_outlier=ratio_known_outlier,
         ratio_known_normal=ratio_known_normal, seed=seed)
wandb.finish()
