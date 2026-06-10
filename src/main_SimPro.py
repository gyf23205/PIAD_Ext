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
import wandb
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from baselines.SimPro import SimProTrainer
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from utils.metrics import (truth_multihot_to_single, compute_affiliation_metrics,
                           single_to_multihot, compute_multihot_metrics)
from utils.data import extract_numpy
from datasets.main import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="SimPro semi-supervised baseline for PIAD_Ext")
    # Dataset
    p.add_argument("--dataset", default="ALFA",
                   choices=["ALFA", "Pegasus", "spoofing_multi_profile", "spoofing_wind"],
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
    return p.parse_known_args()[0]


def main(ratio_pollution=None, ratio_known_outlier=None, ratio_known_normal=None, seed=None):
    args = parse_args()
    if ratio_pollution     is not None: args.ratio_pollution     = ratio_pollution
    if ratio_known_outlier is not None: args.ratio_known_outlier = ratio_known_outlier
    if ratio_known_normal  is not None: args.ratio_known_normal  = ratio_known_normal
    if seed                is not None: args.seed                = seed

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

    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)

    try:
        test_auc = float(roc_auc_score(y_true_bin, y_score))
    except ValueError:
        test_auc = float('nan')
    bin_f1     = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_acc    = float(np.mean(y_true_bin == y_pred_bin))
    anomaly_mask = y_true_bin == 1
    bin_recall = (float(y_pred_bin[anomaly_mask].sum() / anomaly_mask.sum())
                  if anomaly_mask.any() else float('nan'))
    mcc = float(matthews_corrcoef(y_true_bin, y_pred_bin))

    aff = compute_affiliation_metrics(y_true_bin, y_pred_bin)
    p_aff, r_aff, f_aff = aff['p_aff'], aff['r_aff'], aff['f_aff']

    # Multi-hot metrics (multi-label indicator format)
    n_ac      = y_test.shape[1]
    y_pred_mh = single_to_multihot(y_pred, n_ac)
    y_true_mh = y_test.astype(int)
    mh        = compute_multihot_metrics(y_true_mh, y_pred_mh)
    f1_macro, f1_weighted = mh['f1_macro'], mh['f1_weighted']
    mh_acc,   mh_recall   = mh['mh_acc'],   mh['mh_recall']

    def _fmt(v):
        try:
            return 'N/A' if np.isnan(v) else f'{v:.4f}'
        except (TypeError, ValueError):
            return f'{v:.4f}'

    W = 46
    print("=" * W)
    print("    SimPro Test Results")
    print("=" * W)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Test samples : {len(y_true_bin)}")
    print(f"  Train time   : {train_time:.1f}s")
    print(f"  Test time    : {test_time:.3f}s")
    print("-" * W)
    print("  -- Multi-hot --")
    print(f"  F1 macro      : {_fmt(f1_macro)}")
    print(f"  F1 weighted   : {_fmt(f1_weighted)}")
    print(f"  MH accuracy   : {_fmt(mh_acc)}")
    print(f"  MH recall     : {_fmt(mh_recall)}")
    print("-" * W)
    print("  -- Binary --")
    print(f"  AUC           : {_fmt(test_auc)}")
    print(f"  F1            : {_fmt(bin_f1)}")
    print(f"  Accuracy      : {_fmt(bin_acc)}")
    print(f"  Recall        : {_fmt(bin_recall)}")
    print(f"  MCC           : {_fmt(mcc)}")
    print("-" * W)
    print("  -- Affiliation --")
    print(f"  P_aff (UAff)  : {_fmt(p_aff)}")
    print(f"  R_aff (NAff)  : {_fmt(r_aff)}")
    print(f"  F_aff         : {_fmt(f_aff)}")
    print("=" * W)

    wandb.log({
        'val_auc':      best_auc,
        'test_auc':     test_auc,
        'f1_macro':     f1_macro,
        'f1_weighted':  f1_weighted,
        'mh_acc':       mh_acc,
        'mh_recall':    mh_recall,
        'bin_f1':       bin_f1,
        'bin_acc':      bin_acc,
        'bin_recall':   bin_recall,
        'mcc':          mcc,
        'p_aff':        p_aff,
        'r_aff':        r_aff,
        'f_aff':        f_aff,
    })


if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project='PIAD_Ext',
        name='SimPro',
        config={
            'lr': 0.01,
            'n_epochs': 200,
            'batch_size': 64,
            'tau': 1.0,
            'threshold': 0.95,
        }
    )
    ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
    seed = wandb.config.seed
    main(ratio_pollution=ratio_pollution, ratio_known_outlier=ratio_known_outlier,
         ratio_known_normal=ratio_known_normal, seed=seed)
wandb.finish()
