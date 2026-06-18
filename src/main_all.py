import torch
import logging
import random
import argparse
import numpy as np
from datetime import datetime
import wandb
import setting
import os

from sklearn.metrics import matthews_corrcoef
from utils.config import Config
from utils.metrics import compute_affiliation_metrics
from DeepSAD import DeepSAD
from datasets.main import load_dataset


_COEFF = {'sad': 1.0, 'pred': 4.8, 'dir': 5.0, 'cluster': 1.7}
_PHY_HYPERS = dict(
    eta=6.9264986318494515, tau=0.5, coeff=_COEFF,
    lr=0.0001, n_epochs=700, lr_milestone=[200, 400, 600, 800],
    batch_size=128, weight_decay=0.5e-6,
    setting_hypers=[256, 512, 256, 2.0],
)

DATASET_CONFIGS = {
    'Pegasus': {
        **_PHY_HYPERS,
        'net_name': 'mlp_pegasus',
        'normal_class': 0,
        'known_outlier_classes': [1, 3, 4, 7],
        'n_known_outlier_classes': 4,
    },
    'ALFA': {
        **_PHY_HYPERS,
        'net_name': 'mlp_alfa',
        'normal_class': 0,
        'known_outlier_classes': [1, 3, 4, 6],
        'n_known_outlier_classes': 4,
    },
    'spoofing_multi_profile': {
        **_PHY_HYPERS,
        'net_name': 'spoof_mlp',
        'normal_class': 0,
        'known_outlier_classes': [1],
        'n_known_outlier_classes': 1,
    },
    'spoofing_wind': {
        **_PHY_HYPERS,
        'net_name': 'spoof_mlp',
        'normal_class': 0,
        'known_outlier_classes': [1],
        'n_known_outlier_classes': 1,
    },
}


def main(dataset_name, net_name, xp_path, data_path,
         load_config=None, load_model=None, load_path=None,
         num_threads=0, n_jobs_dataloader=0, optimizer_name='adam',
         eta=6.9264986318494515,
         ratio_known_normal=0.0, ratio_known_outlier=0.0, ratio_pollution=0.0,
         device='cpu', seed=-1,
         normal_class=0, known_outlier_classes=None, n_known_outlier_classes=1,
         lr=0.0001, n_epochs=700, lr_milestone=None,
         batch_size=128, weight_decay=0.5e-6,
         pretrain=False, tau=0.5, aug_mode='gaussian', save=False,
         coeff=None, model_path='.'):

    if known_outlier_classes is None:
        known_outlier_classes = []
    if lr_milestone is None:
        lr_milestone = [200, 400, 600, 800]
    if coeff is None:
        coeff = _COEFF

    cfg = Config(locals().copy())

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    log_file = xp_path + '/log-' + time + '.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    logger.info('Known anomaly classes: %s' % known_outlier_classes)
    logger.info('Network: %s' % net_name)

    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % seed)

    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_classes, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(seed))

    deepSAD = DeepSAD(eta)
    deepSAD.net_name = net_name

    if load_model:
        deepSAD.load_model(model_path=load_path, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_path)

    logger.info('Pretraining: %s' % pretrain)

    logger.info('Training optimizer: %s' % optimizer_name)
    logger.info('Training learning rate: %g' % lr)
    logger.info('Training epochs: %d' % n_epochs)
    logger.info('Training learning rate scheduler milestones: %s' % (lr_milestone,))
    logger.info('Training batch size: %d' % batch_size)
    logger.info('Training weight decay: %g' % weight_decay)

    deepSAD.train_physical(dataset,
                           n_known_outlier_classes,
                           known_outlier_classes,
                           coeff,
                           optimizer_name=optimizer_name,
                           lr=lr,
                           n_epochs=n_epochs,
                           lr_milestones=lr_milestone,
                           batch_size=batch_size,
                           weight_decay=weight_decay,
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader,
                           tau=tau,
                           model_path=model_path,
                           save=save,
                           aug_mode=aug_mode)

    deepSAD.test_physical(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # --- Compute standardized evaluation metrics ---
    trainer    = deepSAD.trainer
    labels_bin = trainer.labels_bin
    y_pred_bin = trainer.y_pred_bin

    test_auc    = float(deepSAD.results['test_auc'])
    f1_macro    = float(deepSAD.results['test_f1_macro_mh'])
    f1_weighted = float(deepSAD.results['test_f1_weighted_mh'])
    mh_acc      = float(deepSAD.results['test_hamming_acc'])
    mh_recall   = float(deepSAD.results['test_mh_recall'])
    bin_f1      = float(deepSAD.results['test_f1_binary'])
    bin_acc     = float(deepSAD.results['test_acc_binary'])
    bin_recall  = float(deepSAD.results['test_recall_binary'])
    mcc         = float(matthews_corrcoef(labels_bin, y_pred_bin))
    aff         = compute_affiliation_metrics(labels_bin, y_pred_bin)
    p_aff, r_aff, f_aff = aff['p_aff'], aff['r_aff'], aff['f_aff']

    def _fmt(v):
        try:
            return 'N/A' if np.isnan(v) else f'{v:.4f}'
        except (TypeError, ValueError):
            return f'{v:.4f}'

    W = 46
    print('=' * W)
    print('    DeepSAD-Physical Test Results')
    print('=' * W)
    print(f'  Dataset      : {dataset_name}')
    print(f'  Test samples : {len(labels_bin)}')
    print('-' * W)
    print('  -- Multi-hot --')
    print(f'  F1 macro      : {_fmt(f1_macro)}')
    print(f'  F1 weighted   : {_fmt(f1_weighted)}')
    print(f'  MH accuracy   : {_fmt(mh_acc)}')
    print(f'  MH recall     : {_fmt(mh_recall)}')
    print('-' * W)
    print('  -- Binary --')
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
        'test_auc':    test_auc,
        'f1_macro':    f1_macro,
        'f1_weighted': f1_weighted,
        'mh_acc':      mh_acc,
        'mh_recall':   mh_recall,
        'bin_f1':      bin_f1,
        'bin_acc':     bin_acc,
        'bin_recall':  bin_recall,
        'mcc':         mcc,
        'p_aff':       p_aff,
        'r_aff':       r_aff,
        'f_aff':       f_aff,
    })

    if save:
        deepSAD.save_results(export_json=model_path + f'/results_physical_seed{seed}.json')
        deepSAD.save_model(export_model=model_path + f'/model_physical_best_seed{seed}.tar', save_ae=pretrain)
        cfg.save_config(export_json=model_path + f'/config_physical_seed{seed}.json')


def parse_args():
    p = argparse.ArgumentParser(description='DeepSAD-Physical (PCAD) for PIAD_Ext')
    p.add_argument('--dataset', default='Pegasus', choices=list(DATASET_CONFIGS),
                   help='Dataset name.')
    p.add_argument('--ratio_pollution',     type=float, default=0.0,
                   help='Pollution ratio of unlabeled train data.')
    p.add_argument('--ratio_known_outlier', type=float, default=0.0,
                   help='Ratio of labeled anomalous train samples.')
    p.add_argument('--ratio_known_normal',  type=float, default=0.0,
                   help='Ratio of labeled normal train samples.')
    p.add_argument('--seed', type=int, default=-1, help='Random seed (-1 to disable).')
    p.add_argument('--save', action='store_true', help='Save model/results/config.')
    p.add_argument('--no_wandb', action='store_true',
                   help='Disable wandb (run offline from the terminal, no login required).')
    # Sweep agents inject extra CLI args (e.g. --coeff_sad, --ratios) that this
    # parser doesn't define; wandb reads those into wandb.config, so ignore them.
    args, _ = p.parse_known_args()
    return args


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parse_args()

    if not args.no_wandb:
        wandb.login()
    wandb.init(
        project='PIAD_Ext',
        name='PCAD',
        mode='disabled' if args.no_wandb else 'online',
        config={
            'lr': 0.0001,
            'batch_size': 128,
            'weight_decay': 0.5e-6,
            'physical': True,
            'pretrain': False,
        }
    )

    # Sweep agents populate wandb.config; terminal runs fall back to CLI args.
    if hasattr(wandb.config, 'ratios'):
        dataset_name = wandb.config.dataset
        ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
        seed = wandb.config.seed
        save = args.save
    else:
        dataset_name = args.dataset
        ratio_pollution = args.ratio_pollution
        ratio_known_outlier = args.ratio_known_outlier
        ratio_known_normal = args.ratio_known_normal
        seed = args.seed
        save = args.save

    # Sweep agents may override the loss-component weights; otherwise fall back
    # to the dataset defaults.
    coeff_override = None
    if hasattr(wandb.config, 'coeff_sad'):
        coeff_override = {
            'sad':     float(wandb.config.coeff_sad),
            'pred':    float(wandb.config.coeff_pred),
            'dir':     float(wandb.config.coeff_dir),
            'cluster': float(wandb.config.coeff_cluster),
        }

    defaults = DATASET_CONFIGS[dataset_name]
    setting.init(defaults['setting_hypers'])

    known_outlier_classes = defaults['known_outlier_classes'] if ratio_known_outlier > 0 else []
    n_known_outlier_classes = len(known_outlier_classes)

    rko = str(ratio_known_outlier).replace('.', '')
    rp  = str(ratio_pollution).replace('.', '')
    xp_path    = f'./log/{dataset_name}'
    model_path = f'./saved_model/physical/{dataset_name}/model_{rko}_{rp}'
    os.makedirs(xp_path,    exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    main(dataset_name, defaults['net_name'], xp_path, './data',
         eta=defaults['eta'], tau=defaults['tau'],
         coeff=coeff_override if coeff_override is not None else defaults['coeff'],
         ratio_known_normal=ratio_known_normal,
         ratio_known_outlier=ratio_known_outlier,
         ratio_pollution=ratio_pollution,
         seed=seed, device=device,
         normal_class=defaults['normal_class'],
         known_outlier_classes=known_outlier_classes,
         n_known_outlier_classes=n_known_outlier_classes,
         lr=defaults['lr'], n_epochs=defaults['n_epochs'],
         lr_milestone=defaults['lr_milestone'],
         batch_size=defaults['batch_size'],
         weight_decay=defaults['weight_decay'],
         aug_mode='gaussian', save=save, model_path=model_path)

wandb.finish()
