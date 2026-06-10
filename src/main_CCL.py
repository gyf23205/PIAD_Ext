"""
Entry point for the CCL semi-supervised baseline.

CCL: Continuous Contrastive Learning for Long-Tailed Semi-Supervised Recognition
(NeurIPS 2024, Zhou et al.)  https://github.com/zhouzihao11/CCL

Usage examples:
  python src/main_CCL.py --dataset ALFA
  python src/main_CCL.py --dataset Pegasus --n_epochs 200
  python src/main_CCL.py --dataset ALFA --save_path ./saved_model/ccl_alfa.pt
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import wandb
sys.path.insert(0, os.path.dirname(__file__))

from baselines.CCL import CCLTrainer
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single, compute_affiliation_metrics
from utils.data import extract_numpy
from datasets.main import load_dataset


# ---------------------------------------------------------------------------
# Dataset defaults  (mirrors main_CATS.py / main_DASO.py structure)
# ---------------------------------------------------------------------------

_CCL_HYPERS_DEFAULT = {
    "h_dims":       [256, 128],
    "rep_dim":      64,
    "lr":           0.001,
    "n_epochs":     300,
    "batch_size":   64,
    "lambda1":      0.7,
    "lambda2":      1.0,
    "beta_spl":     0.2,
    "tau_logit":    2.0,
    "energy_T":     1.0,
    "energy_zeta":  None,
    "ema_alpha":    0.9,
    "tau_c":        0.07,
    "eval_period":  10,
}

DATASET_CONFIGS = {
    "Pegasus": {
        "normal_class":             0,
        "known_outlier_classes":    [1, 3, 4, 6],
        "n_known_outlier_classes":  4,
        "ratio_known_normal":       0.2,
        "ratio_known_outlier":      0.3,
        "ratio_pollution":          0.1,
        "ccl_hypers":               dict(_CCL_HYPERS_DEFAULT),
    },
    "ALFA": {
        "normal_class":             0,
        "known_outlier_classes":    [1, 2, 3, 4],
        "n_known_outlier_classes":  4,
        "ratio_known_normal":       0.2,
        "ratio_known_outlier":      0.3,
        "ratio_pollution":          0.1,
        "ccl_hypers":               dict(_CCL_HYPERS_DEFAULT),
    },
    "spoofing_multi_profile": {
        "normal_class":             0,
        "known_outlier_classes":    [1, 2],
        "n_known_outlier_classes":  2,
        "ratio_known_normal":       0.2,
        "ratio_known_outlier":      0.3,
        "ratio_pollution":          0.1,
        "ccl_hypers":               dict(_CCL_HYPERS_DEFAULT),
    },
    "spoofing_wind": {
        "normal_class":             0,
        "known_outlier_classes":    [1, 2],
        "n_known_outlier_classes":  2,
        "ratio_known_normal":       0.2,
        "ratio_known_outlier":      0.3,
        "ratio_pollution":          0.1,
        "ccl_hypers":               dict(_CCL_HYPERS_DEFAULT),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="CCL continuous contrastive learning baseline for PIAD_Ext"
    )
    # Dataset
    p.add_argument("--dataset", default="ALFA", choices=list(DATASET_CONFIGS),
                   help="Dataset name")
    p.add_argument("--data_path", default="./data",
                   help="Root directory for data files")
    p.add_argument("--known_outlier_class", type=int, nargs="+", default=None,
                   help="Known anomaly class indices (overrides dataset default)")
    p.add_argument("--ratio_known_normal",  type=float, default=None)
    p.add_argument("--ratio_known_outlier", type=float, default=None)
    p.add_argument("--ratio_pollution",     type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    # Model / training
    p.add_argument("--n_epochs",    type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--rep_dim",     type=int,   default=None)
    p.add_argument("--lambda1",     type=float, default=None,
                   help="Weight for classification loss (Eq. 22)")
    p.add_argument("--lambda2",     type=float, default=None,
                   help="Weight for smoothed-PL contrastive loss (Eq. 22)")
    p.add_argument("--beta_spl",    type=float, default=None,
                   help="Label-propagation coefficient β (Eq. 21)")
    p.add_argument("--tau_logit",   type=float, default=None,
                   help="Logit-adjustment temperature τ")
    p.add_argument("--energy_zeta", type=float, default=None,
                   help="Energy threshold ζ (None = use all unlabeled)")
    p.add_argument("--eval_period", type=int,   default=None)
    # Output
    p.add_argument("--save_path", default="./saved_model/ccl_checkpoint.pt",
                   help="Where to save the trained checkpoint")
    p.add_argument("--no_save", action="store_true",
                   help="Skip saving the checkpoint")
    return p.parse_known_args()[0]




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    defaults = DATASET_CONFIGS[args.dataset]

    known_outlier_class = tuple(
        args.known_outlier_class
        if args.known_outlier_class is not None
        else defaults["known_outlier_classes"]
    )
    ratio_known_normal  = args.ratio_known_normal  or defaults["ratio_known_normal"]
    ratio_known_outlier = args.ratio_known_outlier or defaults["ratio_known_outlier"]
    ratio_pollution     = args.ratio_pollution     or defaults["ratio_pollution"]

    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        normal_class=defaults["normal_class"],
        known_outlier_class=known_outlier_class,
        n_known_outlier_classes=len(known_outlier_class),
        ratio_known_normal=ratio_known_normal,
        ratio_known_outlier=ratio_known_outlier,
        ratio_pollution=ratio_pollution,
        random_state=np.random.RandomState(args.seed),
    )

    logger.info("Extracting training / validation / test arrays ...")
    X_train, _,     semi_y = extract_numpy(dataset.train_set)
    X_val,   y_val, _      = extract_numpy(dataset.val_set)
    X_test,  y_test, _     = extract_numpy(dataset.test_set)

    logger.info(f"  Train subset : {X_train.shape}")
    logger.info(f"  Val          : {X_val.shape}")
    logger.info(f"  Test         : {X_test.shape}")

    # Build trainer config — CLI overrides dataset defaults
    h = defaults["ccl_hypers"]
    cfg = {
        "h_dims":       h["h_dims"],
        "rep_dim":      args.rep_dim     if args.rep_dim     is not None else h["rep_dim"],
        "lr":           args.lr          if args.lr          is not None else h["lr"],
        "n_epochs":     args.n_epochs    if args.n_epochs    is not None else h["n_epochs"],
        "batch_size":   args.batch_size  if args.batch_size  is not None else h["batch_size"],
        "lambda1":      args.lambda1     if args.lambda1     is not None else h["lambda1"],
        "lambda2":      args.lambda2     if args.lambda2     is not None else h["lambda2"],
        "beta_spl":     args.beta_spl    if args.beta_spl    is not None else h["beta_spl"],
        "tau_logit":    args.tau_logit   if args.tau_logit   is not None else h["tau_logit"],
        "energy_T":     h["energy_T"],
        "energy_zeta":  args.energy_zeta if args.energy_zeta is not None else h["energy_zeta"],
        "ema_alpha":    h["ema_alpha"],
        "tau_c":        h["tau_c"],
        "eval_period":  args.eval_period if args.eval_period is not None else h["eval_period"],
        "device":       "cuda" if torch.cuda.is_available() else "cpu",
    }

    trainer = CCLTrainer(cfg)

    logger.info("Training CCL ...")
    t0 = time.time()
    best_auc = trainer.fit(X_train, semi_y, X_val, y_val)
    train_time = time.time() - t0
    logger.info(f"Training done in {train_time:.1f}s  (best val AUC = {best_auc:.4f})")

    if not args.no_save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        trainer.save_checkpoint(args.save_path)
        logger.info(f"Checkpoint saved to {args.save_path}")

    logger.info("Evaluating on test set ...")
    t1 = time.time()
    y_score = trainer.predict(X_test)
    y_pred  = trainer.predict_labels(X_test)
    y_true  = truth_multihot_to_single(y_test)
    test_time = time.time() - t1

    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    aff = compute_affiliation_metrics(y_true_bin, y_pred_bin)

    width = 40
    print("=" * width)
    print("    CCL Test Results")
    print("=" * width)
    print(f"  Dataset      : {args.dataset}")
    print(f"  Test samples : {len(y_true)}")
    print(f"  Train time   : {train_time:.1f}s")
    print(f"  Test time    : {test_time:.3f}s")
    print("-" * width)
    print(f"  AUC            : {stats['auc']:.4f}")
    print(f"  F1 (macro)     : {stats['f1_macro']:.4f}")
    print(f"  F1 (weighted)  : {stats['f1_weighted']:.4f}")
    print(f"  Accuracy       : {stats['accuracy']:.4f}")
    print(f"  Anomaly Recall : {stats['anomaly_recall']:.4f}")
    print(f"  P_aff (UAff)   : {aff['p_aff']:.4f}")
    print(f"  R_aff (NAff)   : {aff['r_aff']:.4f}")
    print(f"  F_aff          : {aff['f_aff']:.4f}")
    print("=" * width)

    wandb.log({
        'val_auc':        best_auc,
        'test_auc':       stats['auc'],
        'f1_macro':       stats['f1_macro'],
        'f1_weighted':    stats['f1_weighted'],
        'accuracy':       stats['accuracy'],
        'anomaly_recall': stats['anomaly_recall'],
        'p_aff':          aff['p_aff'],
        'r_aff':          aff['r_aff'],
        'f_aff':          aff['f_aff'],
    })


if __name__ == "__main__":
    wandb.login()
    wandb.init(
        project='PIAD_Ext',
        name='CCL',
        config={
            'lr': 0.001,
            'n_epochs': 300,
            'batch_size': 64,
            'lambda1': 0.7,
            'lambda2': 1.0,
        }
    )
    ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
    seed = wandb.config.seed
    main(ratio_pollution=ratio_pollution, ratio_known_outlier=ratio_known_outlier,
         ratio_known_normal=ratio_known_normal, seed=seed)
wandb.finish()
