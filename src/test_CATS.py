"""
Evaluate a saved CATS checkpoint.  Output format mirrors test_physical.py exactly
so results can be compared directly.

Usage:
  python src/test_CATS.py --checkpoint ./saved_model/cats_pegasus.pt --dataset Pegasus
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
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


DATASET_CONFIGS = {
    'Pegasus': {
        'win_size': 20,
        'n_features': 44,
        'normal_class': 0,
        'known_outlier_classes': [1, 3, 4, 6],
        'n_known_outlier_classes': 4,
        'ratio_known_outlier': 0.3,
        'ratio_known_normal': 0.2,
        'ratio_pollution': 0.1,
        'subclasses': True,
        'setting_hypers': [256, 512, 64, 2.0],
    },
    'ALFA': {
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
    },
}



def test(checkpoint_path: str, dataset_name: str,
         data_path: str = './data', device: str = None, seed: int = 4):

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f'Unknown dataset "{dataset_name}". Supported: {list(DATASET_CONFIGS)}')

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = DATASET_CONFIGS[dataset_name]
    setting.init(cfg['setting_hypers'])

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    known_outlier_classes = cfg['known_outlier_classes']
    n_known_outlier_classes = len(known_outlier_classes)

    logger.info(f'Loading dataset {dataset_name} from {data_path}')
    dataset = load_dataset(
        dataset_name, data_path,
        cfg['normal_class'], known_outlier_classes, n_known_outlier_classes,
        cfg['ratio_known_normal'], cfg['ratio_known_outlier'], cfg['ratio_pollution'],
        random_state=np.random.RandomState(seed),
        subclasses=cfg['subclasses'],
    )

    X_val,  y_val,  _ = extract_numpy(dataset.val_set)
    X_test, y_test, _ = extract_numpy(dataset.test_set)

    # CATSTrainer.load_checkpoint restores model architecture from the checkpoint
    trainer = CATSTrainer({'win_size': cfg['win_size'], 'n_features': cfg['n_features'],
                           'device': device})
    logger.info(f'Loading checkpoint from {checkpoint_path}')
    trainer.load_checkpoint(checkpoint_path)

    t0 = time.time()
    y_score = trainer.predict(X_test)
    y_pred  = trainer.predict_labels(X_test)
    test_time = time.time() - t0

    y_true = truth_multihot_to_single(y_test)

    # Multi-class metrics (uses per-class centroids seeded from labeled anomalies)
    r = compute_anomaly_metrics(y_true, y_pred, y_score)

    # Binary metrics — purely unsupervised: threshold chosen on validation set only
    val_score    = trainer.predict(X_val)
    y_val_true   = truth_multihot_to_single(y_val)
    val_true_bin = (y_val_true > 0).astype(int)
    threshold    = _best_binary_threshold(val_score, val_true_bin)
    y_true_bin   = (y_true > 0).astype(int)
    y_pred_bin   = (y_score >= threshold).astype(int)

    try:
        bin_auc = float(roc_auc_score(y_true_bin, y_score))
    except ValueError:
        bin_auc = float('nan')
    bin_f1     = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_mcc    = float(matthews_corrcoef(y_true_bin, y_pred_bin))
    bin_acc    = float(np.mean(y_true_bin == y_pred_bin))
    anomaly_mask = y_true_bin == 1
    bin_recall = (float(y_pred_bin[anomaly_mask].sum() / anomaly_mask.sum())
                  if anomaly_mask.any() else float('nan'))

    print('\n--- Test Results ---')
    print('  -- Binary (unsupervised, threshold from val) --')
    print(f'  AUC:           {bin_auc:.4f}')
    print(f'  F1 (binary):   {bin_f1:.4f}')
    print(f'  MCC (binary):  {bin_mcc:.4f}')
    print(f'  Accuracy:      {bin_acc:.4f}')
    print(f'  Recall:        {bin_recall:.4f}')
    print('  -- Multi-class (semi-supervised, centroid per class) --')
    print(f'  AUC:           {r["auc"]:.4f}')
    print(f'  F1 (macro):    {r["f1_macro"]:.4f}')
    print(f'  F1 (weighted): {r["f1_weighted"]:.4f}')
    print(f'  Accuracy:      {r["accuracy"]:.4f}')
    print(f'  Recall:        {r["anomaly_recall"]:.4f}')
    print(f'  Test time:     {test_time:.3f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a saved CATS checkpoint.')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint file')
    parser.add_argument('--dataset',    required=True, help='Dataset name (e.g. Pegasus)')
    parser.add_argument('--data_path',  default='./data', help='Root data directory')
    parser.add_argument('--device',     default=None,  help='Compute device (cuda/cpu)')
    parser.add_argument('--seed',       type=int, default=4, help='Random seed')
    args = parser.parse_args()

    test(args.checkpoint, args.dataset, args.data_path, args.device, args.seed)
