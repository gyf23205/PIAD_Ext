"""
Evaluate a saved NNG-Mix checkpoint.

Outputs the same metrics as test_physical.py so results are directly comparable:
  AUC, F1 (macro), F1 (weighted), Accuracy, Recall, Test time.

Usage:
  python src/test_NNGMix.py --checkpoint ./saved_model/nngmix_checkpoint.pt --dataset Pegasus
  python src/test_NNGMix.py --checkpoint ./saved_model/nngmix_alfa.pt --dataset ALFA
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))

import setting
from baselines.NNGMix import NNGMixTrainer
from utils.metrics import compute_anomaly_metrics, truth_multihot_to_single
from utils.data import extract_numpy
from datasets.main import load_dataset


DATASET_CONFIGS = {
    'Pegasus': {
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
         data_path: str = './data', device: str | None = None, seed: int = 4):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f'Unknown dataset "{dataset_name}". Supported: {list(DATASET_CONFIGS)}'
        )
    cfg = DATASET_CONFIGS[dataset_name]

    setting.init(cfg['setting_hypers'])

    np.random.seed(seed)
    torch.manual_seed(seed)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger()

    known_outlier_classes = tuple(cfg['known_outlier_classes'])
    n_known_outlier_classes = len(known_outlier_classes)

    logger.info(f'Loading dataset {dataset_name} from {data_path}')
    dataset = load_dataset(
        dataset_name, data_path,
        cfg['normal_class'], known_outlier_classes, n_known_outlier_classes,
        cfg['ratio_known_normal'], cfg['ratio_known_outlier'], cfg['ratio_pollution'],
        random_state=np.random.RandomState(seed),
        subclasses=cfg.get('subclasses', True),
        training=False
    )

    trainer = NNGMixTrainer({'device': device})

    logger.info(f'Loading checkpoint from {checkpoint_path}')
    trainer.load_checkpoint(checkpoint_path)

    logger.info('Running inference on test set ...')
    t0 = time.time()
    X_test, y_test, _ = extract_numpy(dataset.test_set)
    y_true = truth_multihot_to_single(y_test)
    y_score = trainer.predict(X_test)
    y_pred = trainer.predict_labels(X_test)
    test_time = time.time() - t0

    stats = compute_anomaly_metrics(y_true, y_pred, y_score)

    # Same output format as test_physical.py
    print('\n--- NNG-Mix Test Results ---')
    print(f'AUC:           {stats["auc"]:.4f}')
    print(f'F1 (macro):    {stats["f1_macro"]:.4f}')
    print(f'F1 (weighted): {stats["f1_weighted"]:.4f}')
    print(f'Accuracy:      {stats["accuracy"]:.4f}')
    print(f'Recall:        {stats["anomaly_recall"]:.4f}')
    print(f'Test time:     {test_time:.3f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a saved NNG-Mix checkpoint.'
    )
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--dataset', required=True,
                        help='Dataset name (e.g. Pegasus, ALFA)')
    parser.add_argument('--data_path', default='./data',
                        help='Root data directory')
    parser.add_argument('--device', default=None,
                        help='Compute device (cuda / cpu)')
    parser.add_argument('--seed', type=int, default=4,
                        help='Random seed')
    args = parser.parse_args()

    test(args.checkpoint, args.dataset, args.data_path, args.device, args.seed)
