"""
Entry point for the SimAD semi-supervised baseline.

SimAD: Simple Dissimilarity-Based Approach for Time-Series Anomaly Detection
(IEEE TNNLS 2025)

Usage examples:
  python src/main_SimAD.py --dataset Pegasus
  python src/main_SimAD.py --dataset ALFA --d_model 64 --n_layers 2
  python src/main_SimAD.py --dataset spoofing_wind --n_epochs 200 --save_path ./saved_model/simad_wind.pt
"""

import argparse
import logging
import os
import sys
import time
import wandb

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))

from baselines.SimAD import SimADTrainer
from utils.metrics import (truth_multihot_to_single, compute_affiliation_metrics,
                           single_to_multihot, compute_multihot_metrics)
from utils.data import extract_numpy
from datasets.main import load_dataset


def _best_binary_threshold(scores: np.ndarray, y_true_bin: np.ndarray) -> float:
    """Return the score threshold that maximises binary F1 on the given set."""
    best_t, best_f1 = 0.0, -1.0
    for q in np.linspace(1, 99, 99):
        t = float(np.percentile(scores, q))
        preds = (scores >= t).astype(int)
        f = f1_score(y_true_bin, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


DATASET_CONFIGS = {
    'Pegasus': {
        'net_name': 'simad_pegasus',
        'win_size': 20,
        'n_features': 44,
        'normal_class': 0,
        'known_outlier_classes': [1, 3, 4, 7],
        'n_known_outlier_classes': 4,
        'ratio_known_outlier': 0.3,
        'ratio_known_normal': 0.2,
        'ratio_pollution': 0.1,
        'simad_hypers': {
            'patch_size': 4,       # 20 = 5 * 4
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'n_patch_emb': 50,
            'noise_level': 0.3,
            'beta_max': 0.1,
            'n_warmup_epochs': 20,
            'lr': 1e-3,
            'n_epochs': 100,
            'batch_size': 256,
            'patience': 30,
        },
    },
    'ALFA': {
        'net_name': 'simad_alfa',
        'win_size': 25,
        'n_features': 35,
        'normal_class': 0,
        'known_outlier_classes': [1, 3, 4, 6],
        'n_known_outlier_classes': 4,
        'ratio_known_outlier': 0.3,
        'ratio_known_normal': 0.2,
        'ratio_pollution': 0.1,
        'simad_hypers': {
            'patch_size': 5,       # 25 = 5 * 5
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'n_patch_emb': 50,
            'noise_level': 0.3,
            'beta_max': 0.1,
            'n_warmup_epochs': 20,
            'lr': 1e-3,
            'n_epochs': 100,
            'batch_size': 256,
            'patience': 30,
        },
    },
    'spoofing_multi_profile': {
        'net_name': 'simad_spoofing_mp',
        'win_size': 100,
        'n_features': 12,
        'normal_class': 0,
        'known_outlier_classes': [1],
        'n_known_outlier_classes': 1,
        'ratio_known_outlier': 0.3,
        'ratio_known_normal': 0.2,
        'ratio_pollution': 0.1,
        'simad_hypers': {
            'patch_size': 10,      # 100 = 10 * 10
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 3,
            'n_patch_emb': 50,
            'noise_level': 0.3,
            'beta_max': 0.1,
            'n_warmup_epochs': 20,
            'lr': 1e-3,
            'n_epochs': 100,
            'batch_size': 256,
            'patience': 30,
        },
    },
    'spoofing_wind': {
        'net_name': 'simad_spoofing_wind',
        'win_size': 100,
        'n_features': 12,
        'normal_class': 0,
        'known_outlier_classes': [1],
        'n_known_outlier_classes': 1,
        'ratio_known_outlier': 0.3,
        'ratio_known_normal': 0.2,
        'ratio_pollution': 0.1,
        'simad_hypers': {
            'patch_size': 10,      # 100 = 10 * 10
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 3,
            'n_patch_emb': 50,
            'noise_level': 0.3,
            'beta_max': 0.1,
            'n_warmup_epochs': 20,
            'lr': 1e-3,
            'n_epochs': 100,
            'batch_size': 256,
            'patience': 30,
        },
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='SimAD anomaly detection baseline for PIAD_Ext')
    # Dataset
    p.add_argument('--dataset', default='Pegasus', choices=list(DATASET_CONFIGS),
                   help='Dataset name')
    p.add_argument('--data_path', default='./data',
                   help='Root directory for data files')
    p.add_argument('--known_outlier_class', type=int, nargs='+', default=None)
    p.add_argument('--ratio_known_normal',  type=float, default=None)
    p.add_argument('--ratio_known_outlier', type=float, default=None)
    p.add_argument('--ratio_pollution',     type=float, default=None)
    p.add_argument('--seed', type=int, default=42)
    # Architecture
    p.add_argument('--patch_size',      type=int,   default=None)
    p.add_argument('--d_model',         type=int,   default=None)
    p.add_argument('--n_heads',         type=int,   default=None)
    p.add_argument('--n_layers',        type=int,   default=None)
    p.add_argument('--n_patch_emb',     type=int,   default=None)
    # Training
    p.add_argument('--noise_level',     type=float, default=None)
    p.add_argument('--beta_max',        type=float, default=None)
    p.add_argument('--n_warmup_epochs', type=int,   default=None)
    p.add_argument('--n_epochs',        type=int,   default=None)
    p.add_argument('--batch_size',      type=int,   default=None)
    p.add_argument('--lr',              type=float, default=None)
    p.add_argument('--patience',        type=int,   default=None)
    # Output
    p.add_argument('--save_path', default='./saved_model/simad_checkpoint.pt')
    p.add_argument('--no_save', action='store_true')
    return p.parse_known_args()[0]


def main(ratio_pollution=None, ratio_known_outlier=None, ratio_known_normal=None,
         seed=None, dataset=None):
    args = parse_args()
    if ratio_pollution     is not None: args.ratio_pollution     = ratio_pollution
    if ratio_known_outlier is not None: args.ratio_known_outlier = ratio_known_outlier
    if ratio_known_normal  is not None: args.ratio_known_normal  = ratio_known_normal
    if seed                is not None: args.seed                = seed
    if dataset             is not None: args.dataset             = dataset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)s  %(message)s')
    logger = logging.getLogger()

    defaults = DATASET_CONFIGS[args.dataset]
    h        = defaults['simad_hypers']

    known_outlier_class = tuple(
        args.known_outlier_class if args.known_outlier_class is not None
        else defaults['known_outlier_classes']
    )
    ratio_known_normal  = (args.ratio_known_normal  if args.ratio_known_normal  is not None
                           else defaults['ratio_known_normal'])
    ratio_known_outlier = (args.ratio_known_outlier if args.ratio_known_outlier is not None
                           else defaults['ratio_known_outlier'])
    ratio_pollution     = (args.ratio_pollution     if args.ratio_pollution     is not None
                           else defaults['ratio_pollution'])

    logger.info(f'Loading dataset: {args.dataset}')
    ds = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        normal_class=defaults['normal_class'],
        known_outlier_class=known_outlier_class,
        n_known_outlier_classes=defaults['n_known_outlier_classes'],
        ratio_known_normal=ratio_known_normal,
        ratio_known_outlier=ratio_known_outlier,
        ratio_pollution=ratio_pollution,
        random_state=np.random.RandomState(args.seed),
    )

    X_train, _,     semi_y = extract_numpy(ds.train_set)
    X_val,   y_val, _      = extract_numpy(ds.val_set)
    X_test,  y_test, _     = extract_numpy(ds.test_set)
    logger.info(f'  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}')

    # Build config — CLI overrides dataset defaults
    def _pick(attr, key):
        v = getattr(args, attr)
        return v if v is not None else h[key]

    cfg = {
        'win_size':        defaults['win_size'],
        'n_features':      defaults['n_features'],
        'patch_size':      _pick('patch_size',      'patch_size'),
        'd_model':         _pick('d_model',          'd_model'),
        'n_heads':         _pick('n_heads',          'n_heads'),
        'n_layers':        _pick('n_layers',         'n_layers'),
        'n_patch_emb':     _pick('n_patch_emb',      'n_patch_emb'),
        'noise_level':     _pick('noise_level',      'noise_level'),
        'beta_max':        _pick('beta_max',          'beta_max'),
        'n_warmup_epochs': _pick('n_warmup_epochs',  'n_warmup_epochs'),
        'lr':              _pick('lr',               'lr'),
        'n_epochs':        _pick('n_epochs',         'n_epochs'),
        'batch_size':      _pick('batch_size',       'batch_size'),
        'patience':        _pick('patience',         'patience'),
        'weight_decay':    1e-5,
        'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    }

    trainer = SimADTrainer(cfg)

    logger.info('Training SimAD ...')
    t0 = time.time()
    best_auc = trainer.fit(X_train, semi_y, X_val, y_val)
    train_time = time.time() - t0
    logger.info(f'Training done in {train_time:.1f}s  (best val AUC = {best_auc:.4f})')

    if not args.no_save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        trainer.save_checkpoint(args.save_path)

    logger.info('Evaluating on test set ...')
    t1 = time.time()
    y_score = trainer.predict(X_test)
    y_pred  = trainer.predict_labels(X_test)
    y_true  = truth_multihot_to_single(y_test)
    test_time = time.time() - t1

    logger.info(f"Score stats — normal: mean={y_score[y_true==0].mean():.3f}"
                f"  anomaly: mean={y_score[y_true>0].mean():.3f}")

    # Binary metrics — threshold from validation set
    val_score    = trainer.predict(X_val)
    y_val_true   = truth_multihot_to_single(y_val)
    val_true_bin = (y_val_true > 0).astype(int)
    threshold    = _best_binary_threshold(val_score, val_true_bin)
    y_true_bin   = (y_true > 0).astype(int)
    y_pred_bin   = (y_score >= threshold).astype(int)

    try:
        test_auc = float(roc_auc_score(y_true_bin, y_score))
    except ValueError:
        test_auc = float('nan')
    bin_f1     = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    mcc        = float(matthews_corrcoef(y_true_bin, y_pred_bin))
    bin_acc    = float(np.mean(y_true_bin == y_pred_bin))
    anomaly_mask = y_true_bin == 1
    bin_recall   = (float(y_pred_bin[anomaly_mask].sum() / anomaly_mask.sum())
                    if anomaly_mask.any() else float('nan'))

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
    print('=' * W)
    print('    SimAD Test Results')
    print('=' * W)
    print(f'  Dataset      : {args.dataset}')
    print(f'  Test samples : {len(y_true_bin)}')
    print(f'  Train time   : {train_time:.1f}s')
    print(f'  Test time    : {test_time:.3f}s')
    print('-' * W)
    print('  -- Multi-hot --')
    print(f'  F1 macro      : {_fmt(f1_macro)}')
    print(f'  F1 weighted   : {_fmt(f1_weighted)}')
    print(f'  MH accuracy   : {_fmt(mh_acc)}')
    print(f'  MH recall     : {_fmt(mh_recall)}')
    print('-' * W)
    print('  -- Binary (threshold from val) --')
    print(f'  AUC           : {_fmt(test_auc)}')
    print(f'  F1            : {_fmt(bin_f1)}')
    print(f'  Accuracy      : {_fmt(bin_acc)}')
    print(f'  Recall        : {_fmt(bin_recall)}')
    print(f'  MCC           : {_fmt(mcc)}')
    print('-' * W)
    print('  -- Affiliation --')
    print(f'  P_aff (UAff)  : {_fmt(p_aff)}')
    print(f'  R_aff (NAff)  : {_fmt(r_aff)}')
    print(f'  F_aff         : {_fmt(f_aff)}')
    print('=' * W)

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


if __name__ == '__main__':
    wandb.login()
    wandb.init(
        project='PIAD_Ext',
        name='SimAD',
        config={
            'patch_size': 4,
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'lr': 1e-3,
            'n_epochs': 100,
            'batch_size': 256,
        }
    )
    if hasattr(wandb.config, 'ratios'):
        ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
        seed    = wandb.config.seed
        dataset = wandb.config.dataset
        main(ratio_pollution=ratio_pollution, ratio_known_outlier=ratio_known_outlier,
             ratio_known_normal=ratio_known_normal, seed=seed, dataset=dataset)
    else:
        main()
wandb.finish()
