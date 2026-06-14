"""
Entry point for the RoSAS semi-supervised baseline.

RoSAS: Robust Semi-supervised Anomaly Selection
(SIGKDD 2023, Ding et al.)

Usage examples:
  python src/main_RoSAS.py --dataset ALFA
  python src/main_RoSAS.py --dataset Pegasus --n_epochs 200
  python src/main_RoSAS.py --dataset ALFA --save_path ./saved_model/rosas_alfa.pt
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

from baselines.RoSAS import RoSAS
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from utils.metrics import truth_multihot_to_single, compute_affiliation_metrics
from utils.data import extract_numpy
from datasets.main import load_dataset


DATASET_CONFIGS = {
    'Pegasus': {
        'known_outlier_classes': [1, 3, 4, 7],
        'n_known_outlier_classes': 4,
        'ratio_known_normal': 0.0,
        'ratio_known_outlier': 0.0,
        'ratio_pollution': 0.0,
    },
    'ALFA': {
        'known_outlier_classes': [1, 3, 4, 6],
        'n_known_outlier_classes': 4,
        'ratio_known_normal': 0.0,
        'ratio_known_outlier': 0.0,
        'ratio_pollution': 0.0,
    },
    'spoofing_multi_profile': {
        'known_outlier_classes': [1],
        'n_known_outlier_classes': 1,
        'ratio_known_normal': 0.0,
        'ratio_known_outlier': 0.0,
        'ratio_pollution': 0.0,
    },
    'spoofing_wind': {
        'known_outlier_classes': [1],
        'n_known_outlier_classes': 1,
        'ratio_known_normal': 0.0,
        'ratio_known_outlier': 0.0,
        'ratio_pollution': 0.0,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="RoSAS semi-supervised baseline for PIAD_Ext")
    p.add_argument("--dataset", default="ALFA", choices=list(DATASET_CONFIGS))
    p.add_argument("--data_path", default="./data")
    p.add_argument("--known_outlier_class", type=int, nargs="+", default=None)
    p.add_argument("--ratio_known_normal", type=float, default=None)
    p.add_argument("--ratio_known_outlier", type=float, default=None)
    p.add_argument("--ratio_pollution", type=float, default=None)
    p.add_argument("--seed", type=int, default=4)
    # Model
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--n_emb", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--nbatch_per_epoch", type=int, default=16)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--score_loss", default="smooth")
    p.add_argument("--no_early_stop", action="store_true")
    # Output
    p.add_argument("--save_path", default="./saved_model/rosas_checkpoint.pt")
    p.add_argument("--no_save", action="store_true")
    return p.parse_known_args()[0]


def main(ratio_pollution=None, ratio_known_outlier=None, ratio_known_normal=None, seed=None):
    args = parse_args()
    if ratio_pollution     is not None: args.ratio_pollution     = ratio_pollution
    if ratio_known_outlier is not None: args.ratio_known_outlier = ratio_known_outlier
    if ratio_known_normal  is not None: args.ratio_known_normal  = ratio_known_normal
    if seed                is not None: args.seed                = seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    logger = logging.getLogger()

    defaults = DATASET_CONFIGS[args.dataset]
    known_outlier_class = tuple(
        args.known_outlier_class if args.known_outlier_class is not None
        else defaults['known_outlier_classes']
    )

    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        normal_class=0,
        known_outlier_class=known_outlier_class,
        n_known_outlier_classes=defaults['n_known_outlier_classes'],
        ratio_known_normal=args.ratio_known_normal or defaults['ratio_known_normal'],
        ratio_known_outlier=args.ratio_known_outlier or defaults['ratio_known_outlier'],
        ratio_pollution=args.ratio_pollution or defaults['ratio_pollution'],
        random_state=np.random.RandomState(args.seed)
    )

    logger.info("Extracting arrays …")
    X_train, _,     semi_y = extract_numpy(dataset.train_set)
    X_val,   y_val, _      = extract_numpy(dataset.val_set)
    X_test,  y_test, _     = extract_numpy(dataset.test_set)
    logger.info(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    trainer = RoSAS(
        epochs=args.n_epochs,
        n_emb=args.n_emb,
        lr=args.lr,
        batch_size=args.batch_size,
        nbatch_per_epoch=args.nbatch_per_epoch,
        alpha=args.alpha,
        margin=args.margin,
        beta=args.beta,
        score_loss=args.score_loss,
        use_es=not args.no_early_stop,
        seed=args.seed,
    )

    logger.info("Training …")
    t0 = time.time()
    trainer.fit(X_train, semi_y, X_val, y_val)
    train_time = time.time() - t0
    logger.info(f"Training done in {train_time:.1f}s")

    if not args.no_save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        trainer.save_checkpoint(args.save_path)
        logger.info(f"Checkpoint saved to {args.save_path}")

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

    # Binary method — multi-hot metrics are not applicable
    f1_macro = f1_weighted = mh_acc = mh_recall = float('nan')

    def _fmt(v):
        try:
            return 'N/A' if np.isnan(v) else f'{v:.4f}'
        except (TypeError, ValueError):
            return f'{v:.4f}'

    W = 46
    print("=" * W)
    print("    RoSAS Test Results")
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
        name='RoSAS',
        config={
            'lr': 0.005,
            'n_epochs': 200,
            'batch_size': 32,
            'alpha': 0.5,
            'margin': 1.0,
            'beta': 1.0,
        }
    )
    ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
    seed = wandb.config.seed
    main(ratio_pollution=ratio_pollution, ratio_known_outlier=ratio_known_outlier,
         ratio_known_normal=ratio_known_normal, seed=seed)
wandb.finish()
