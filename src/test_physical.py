import argparse
import logging
import random

import numpy as np
import torch
import wandb

import setting
from DeepSAD import DeepSAD
from datasets.main import load_dataset
from networks.main import build_network_physical
from optim.DeepSAD_trainer_physical import DeepSADTrainerPhysical

DATASET_CONFIGS = {
    'Pegasus': {
        'net_name': 'mlp_pegasus',
        'normal_class': 0,
        'known_outlier_classes': [1, 3, 4, 6],
        'ratio_known_outlier': 0.3,
        'ratio_known_normal': 0.2,
        'ratio_pollution': 0.1,
        'subclasses': True,
        'eta': 6.9264986318494515,
        'tau': 0.5,
        'coeff': {'sad': 1.0, 'pred': 4.8, 'dir': 5.0, 'cluster': 1.7},
        'setting_hypers': [256, 512, 64, 2.0],  # hd1, hd2, rep, T
    },
}


def test(checkpoint_path, dataset_name, data_path='./data', device=None, seed=4):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f'Unknown dataset "{dataset_name}". Supported: {list(DATASET_CONFIGS)}')
    cfg = DATASET_CONFIGS[dataset_name]

    # setting.init must come before dataset load and trainer creation
    setting.init(cfg['setting_hypers'])

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

    deepSAD = DeepSAD(cfg['eta'])
    deepSAD.net_name = cfg['net_name']
    deepSAD.net = build_network_physical(cfg['net_name'])

    logger.info(f'Loading checkpoint from {checkpoint_path}')
    deepSAD.load_model(checkpoint_path, map_location=device)

    trainer = DeepSADTrainerPhysical(
        n_known_outlier_classes,
        known_outlier_classes,
        dataset.outlier_classes,
        cfg['coeff'],
        device=device,
        tau=cfg['tau'],
    )
    trainer.centroids = {k: v.to(device) for k, v in deepSAD.centroids.items()}
    trainer.roc_curve = deepSAD.roc_curve
    trainer.per_class_thresholds = deepSAD.per_class_thresholds
    deepSAD.trainer = trainer

    wandb.init(mode='disabled')
    deepSAD.test_physical(dataset, device=device)
    wandb.finish()

    r = deepSAD.results
    print('\n--- Test Results ---')
    print(f'AUC:              {r["test_auc"]:.4f}')
    print(f'F1 (macro, mh):   {r["test_f1_macro_mh"]:.4f}')
    print(f'F1 (micro, mh):   {r["test_f1_micro_mh"]:.4f}')
    print(f'Hamming Accuracy: {r["test_hamming_acc"]:.4f}')
    print(f'Subset Accuracy:  {r["test_subset_acc"]:.4f}')
    print(f'\n--- Binary Detection ---')
    print(f'F1 (binary):      {r["test_f1_binary"]:.4f}')
    print(f'Precision:        {r["test_precision_binary"]:.4f}')
    print(f'Recall:           {r["test_recall_binary"]:.4f}')
    print(f'Accuracy:         {r["test_acc_binary"]:.4f}')
    print(f'\nTest time:        {r["test_time"]:.3f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a saved physical DeepSAD checkpoint.')
    parser.add_argument('--checkpoint', required=True, help='Path to .tar checkpoint file')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. Pegasus)')
    parser.add_argument('--data_path', default='./data', help='Root data directory')
    parser.add_argument('--device', default=None, help='Compute device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=4, help='Random seed')
    args = parser.parse_args()

    test(args.checkpoint, args.dataset, args.data_path, args.device, args.seed)
