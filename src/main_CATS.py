"""
Entry point for the CATS semi-supervised baseline.

CATS: Contrastive learning for Anomaly detection in Time Series
(IEEE BigData 2024, doi:10.1109/BigData62323.2024.10825476)

Usage examples:
  python src/main_CATS.py --dataset Pegasus
  python src/main_CATS.py --dataset ALFA --encoder_type mlp
  python src/main_CATS.py --dataset Pegasus --n_epochs 200 --save_path ./saved_model/cats_pegasus.pt
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef
sys.path.insert(0, os.path.dirname(__file__))

import setting
from baselines.CATS import CATSTrainer
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single
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


# Dataset configs mirror those in main_all.py / test_physical.py / main_NNGMix.py
DATASET_CONFIGS = {
    'Pegasus': {
        'net_name': 'cats_ts2vec_pegasus',
        'win_size': 20,
        'n_features': 44,
        'normal_class': 0,
        'known_outlier_classes': [1, 3, 4, 6],
        'n_known_outlier_classes': 4,
        'ratio_known_outlier': 0.3,
        'ratio_known_normal': 0.2,
        'ratio_pollution': 0.1,
        'subclasses': True,
        'setting_hypers': [256, 512, 64, 2.0],   # hd1, hd2, rep, T
        'cats_hypers': {
            'coef_gcl': 0.5,
            'coef_tcl': 0.5,
            'lr': 0.001,
            'n_epochs': 100,
            'patience': 50,
            'batch_size': 512,
            'temperature': 0.1,
            'gamma': 1.0,
            'margin': 5.0,
        },
    },
    'ALFA': {
        'net_name': 'cats_ts2vec_alfa',
        'win_size': 25,
        'n_features': 35,
        'normal_class': 0,
        'known_outlier_classes': [1, 2, 3, 4],
        'n_known_outlier_classes': 4,
        'ratio_known_outlier': 0.3,
        'ratio_known_normal': 0.2,
        'ratio_pollution': 0.1,
        'subclasses': True,
        'setting_hypers': [256, 512, 64, 2.0],
        'cats_hypers': {
            'coef_gcl': 0.5,
            'coef_tcl': 0.5,
            'lr': 0.001,
            'n_epochs': 100,
            'patience': 50,
            'batch_size': 512,
            'temperature': 0.1,
            'gamma': 1.0,
            'margin': 5.0,
        },
    },
}


def parse_args():
    p = argparse.ArgumentParser(description='CATS contrastive baseline for PIAD_Ext')
    # Dataset
    p.add_argument('--dataset', default='Pegasus', choices=list(DATASET_CONFIGS),
                   help='Dataset name')
    p.add_argument('--data_path', default='./data',
                   help='Root directory for data files')
    p.add_argument('--known_outlier_class', type=int, nargs='+', default=None,
                   help='Known anomaly class indices (overrides dataset default)')
    p.add_argument('--ratio_known_normal',  type=float, default=None)
    p.add_argument('--ratio_known_outlier', type=float, default=None)
    p.add_argument('--ratio_pollution',     type=float, default=None)
    p.add_argument('--seed', type=int, default=4)
    # Model architecture
    p.add_argument('--encoder_type', default=None, choices=['ts2vec', 'mlp'],
                   help='CATS encoder backbone (overrides dataset default)')
    p.add_argument('--output_size', type=int, default=None,
                   help='Encoder embedding dim (default: setting.rep from dataset config)')
    # Training
    p.add_argument('--n_epochs',    type=int,   default=None)
    p.add_argument('--batch_size',  type=int,   default=None)
    p.add_argument('--lr',          type=float, default=None)
    p.add_argument('--patience',    type=int,   default=None)
    p.add_argument('--coef_gcl',    type=float, default=None)
    p.add_argument('--coef_tcl',    type=float, default=None)
    p.add_argument('--temperature', type=float, default=None)
    p.add_argument('--gamma',       type=float, default=None)
    p.add_argument('--margin',      type=float, default=None)
    # Output
    p.add_argument('--save_path', default='./saved_model/cats_checkpoint.pt',
                   help='Where to save the trained checkpoint')
    p.add_argument('--no_save', action='store_true',
                   help='Skip saving the checkpoint')
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)s  %(message)s',
    )
    logger = logging.getLogger()

    defaults = DATASET_CONFIGS[args.dataset]
    setting.init(defaults['setting_hypers'])

    known_outlier_class = tuple(
        args.known_outlier_class if args.known_outlier_class is not None
        else defaults['known_outlier_classes']
    )
    ratio_known_normal  = args.ratio_known_normal  or defaults['ratio_known_normal']
    ratio_known_outlier = args.ratio_known_outlier or defaults['ratio_known_outlier']
    ratio_pollution     = args.ratio_pollution     or defaults['ratio_pollution']

    logger.info(f'Loading dataset: {args.dataset}')
    dataset = load_dataset(
        dataset_name=args.dataset,
        data_path=args.data_path,
        normal_class=defaults['normal_class'],
        known_outlier_class=known_outlier_class,
        n_known_outlier_classes=defaults['n_known_outlier_classes'],
        ratio_known_normal=ratio_known_normal,
        ratio_known_outlier=ratio_known_outlier,
        ratio_pollution=ratio_pollution,
        random_state=np.random.RandomState(args.seed),
        subclasses=defaults.get('subclasses', True),
    )

    logger.info('Extracting training / validation / test arrays ...')
    X_train, _,     semi_y = extract_numpy(dataset.train_set)
    X_val,   y_val, _      = extract_numpy(dataset.val_set)
    X_test,  y_test, _     = extract_numpy(dataset.test_set)

    logger.info(f'  Train subset : {X_train.shape}')
    logger.info(f'  Val          : {X_val.shape}')
    logger.info(f'  Test         : {X_test.shape}')

    # Build trainer config — CLI overrides dataset defaults
    h = defaults['cats_hypers']
    output_size = args.output_size if args.output_size is not None else defaults['setting_hypers'][2]
    encoder_type = args.encoder_type if args.encoder_type is not None else (
        'ts2vec' if 'ts2vec' in defaults['net_name'] else 'mlp'
    )

    cfg = {
        'win_size':     defaults['win_size'],
        'n_features':   defaults['n_features'],
        'output_size':  output_size,
        'proj_size':    output_size // 2,
        'encoder_type': encoder_type,
        'lr':           args.lr          if args.lr          is not None else h['lr'],
        'n_epochs':     args.n_epochs    if args.n_epochs    is not None else h['n_epochs'],
        'batch_size':   args.batch_size  if args.batch_size  is not None else h['batch_size'],
        'patience':     args.patience    if args.patience    is not None else h['patience'],
        'coef_gcl':     args.coef_gcl    if args.coef_gcl    is not None else h['coef_gcl'],
        'coef_tcl':     args.coef_tcl    if args.coef_tcl    is not None else h['coef_tcl'],
        'temperature':  args.temperature if args.temperature is not None else h['temperature'],
        'gamma':        args.gamma       if args.gamma       is not None else h['gamma'],
        'margin':       args.margin      if args.margin      is not None else h['margin'],
        'weight_decay': 1e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    trainer = CATSTrainer(cfg)

    logger.info('Training CATS ...')
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

    # Multi-class metrics (uses per-class centroids seeded from labeled anomalies)
    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    # Binary metrics — purely unsupervised: threshold chosen on validation set only
    val_score    = trainer.predict(X_val)
    y_val_true   = truth_multihot_to_single(y_val)
    val_true_bin = (y_val_true > 0).astype(int)
    threshold    = _best_binary_threshold(val_score, val_true_bin)
    y_true_bin   = (y_true > 0).astype(int)
    y_pred_bin   = (y_score >= threshold).astype(int)

    from sklearn.metrics import roc_auc_score
    try:
        bin_auc = float(roc_auc_score(y_true_bin, y_score))
    except ValueError:
        bin_auc = float('nan')
    bin_f1  = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_mcc = float(matthews_corrcoef(y_true_bin, y_pred_bin))
    bin_acc = float(np.mean(y_true_bin == y_pred_bin))
    anomaly_mask = y_true_bin == 1
    bin_recall = (float(y_pred_bin[anomaly_mask].sum() / anomaly_mask.sum())
                  if anomaly_mask.any() else float('nan'))

    width = 40
    print('=' * width)
    print('    CATS Test Results')
    print('=' * width)
    print(f'  Dataset      : {args.dataset}')
    print(f'  Encoder      : {encoder_type}')
    print(f'  Test samples : {len(y_true)}')
    print(f'  Train time   : {train_time:.1f}s')
    print(f'  Test time    : {test_time:.3f}s')
    print('-' * width)
    print('  -- Binary (unsupervised, threshold from val) --')
    print(f'  AUC            : {bin_auc:.4f}')
    print(f'  F1 (binary)    : {bin_f1:.4f}')
    print(f'  MCC (binary)   : {bin_mcc:.4f}')
    print(f'  Accuracy       : {bin_acc:.4f}')
    print(f'  Anomaly Recall : {bin_recall:.4f}')
    print('-' * width)
    print('  -- Multi-class (semi-supervised, centroid per class) --')
    print(f'  AUC            : {stats["auc"]:.4f}')
    print(f'  F1 (macro)     : {stats["f1_macro"]:.4f}')
    print(f'  F1 (weighted)  : {stats["f1_weighted"]:.4f}')
    print(f'  Accuracy       : {stats["accuracy"]:.4f}')
    print(f'  Anomaly Recall : {stats["anomaly_recall"]:.4f}')
    print('=' * width)


if __name__ == '__main__':
    main()
