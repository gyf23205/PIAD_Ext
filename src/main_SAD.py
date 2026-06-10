import torch
import logging
import random
import numpy as np
import wandb
import os
from datetime import datetime
import setting

from sklearn.metrics import f1_score, matthews_corrcoef
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from utils.metrics import compute_affiliation_metrics
from DeepSAD import DeepSAD
from datasets.main import load_dataset

def main(dataset_name, net_name, xp_path, data_path, load_config=None, load_model=None, load_path=None, eta=1.0,
         ratio_known_normal=0.0, ratio_known_outlier=0.0, ratio_pollution=0.0, device='cuda', seed=-1,
         optimizer_name='adam', lr=0.001, n_epochs=50, lr_milestone=50, batch_size=128, weight_decay=1e-6,
         pretrain=True, ae_optimizer_name='adam', ae_lr=0.001, ae_n_epochs=100, ae_lr_milestone=[0], ae_batch_size=128, ae_weight_decay=1e-6,
         num_threads=0, n_jobs_dataloader=0, normal_class=0, known_outlier_class=1, n_known_outlier_classes=0, with_next=False):
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    log_file = xp_path + '/log-'+ time + '.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % cfg.settings['eta'])

    # Set seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % seed)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    # print(seed)
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(seed))
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(eta)
    deepSAD.set_network(net_name)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_path, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_path)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % ae_optimizer_name)
        logger.info('Pretraining learning rate: %g' % ae_lr)
        logger.info('Pretraining epochs: %d' % ae_n_epochs)
        logger.info('Pretraining learning rate scheduler milestones: %s' % (ae_lr_milestone,))
        logger.info('Pretraining batch size: %d' % ae_batch_size)
        logger.info('Pretraining weight decay: %g' % ae_weight_decay)

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain(dataset,
                         optimizer_name=ae_optimizer_name,
                         lr=ae_lr,
                         n_epochs=ae_n_epochs,
                         lr_milestones=ae_lr_milestone,
                         batch_size=ae_batch_size,
                         weight_decay=ae_weight_decay,
                         device=device,
                         n_jobs_dataloader=n_jobs_dataloader)

        # Save pretraining results
        deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')

    # Log training details
    logger.info('Training optimizer: %s' % optimizer_name)
    logger.info('Training learning rate: %g' % lr)
    logger.info('Training epochs: %d' % n_epochs)
    logger.info('Training learning rate scheduler milestones: %s' % (lr_milestone,))
    logger.info('Training batch size: %d' % batch_size)
    logger.info('Training weight decay: %g' % weight_decay)

        # Train model on dataset
    deepSAD.train(dataset,
                optimizer_name=optimizer_name,
                lr=lr,
                n_epochs=n_epochs,
                lr_milestones=lr_milestone,
                batch_size=batch_size,
                weight_decay=weight_decay,
                device=device,
                n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # --- Compute standardized evaluation metrics ---
    _, _labels, _scores = zip(*deepSAD.results['test_scores'])
    y_true_bin = np.array(_labels, dtype=int)
    y_score    = np.array(_scores, dtype=float)

    # Percentile threshold derived from true anomaly fraction
    anomaly_fraction = float(y_true_bin.mean())
    threshold  = float(np.percentile(y_score, 100 - anomaly_fraction * 100))
    y_pred_bin = (y_score >= threshold).astype(int)

    nan        = float('nan')
    test_auc   = float(deepSAD.results['test_auc'])
    # Binary method — multi-hot metrics are not applicable
    f1_macro = f1_weighted = mh_acc = mh_recall = nan
    bin_f1     = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
    bin_acc    = float(np.mean(y_true_bin == y_pred_bin))
    anomaly_mask = y_true_bin == 1
    bin_recall = (float(y_pred_bin[anomaly_mask].sum() / anomaly_mask.sum())
                  if anomaly_mask.any() else nan)
    mcc        = float(matthews_corrcoef(y_true_bin, y_pred_bin))
    aff        = compute_affiliation_metrics(y_true_bin, y_pred_bin)
    p_aff, r_aff, f_aff = aff['p_aff'], aff['r_aff'], aff['f_aff']

    def _fmt(v):
        try:
            return 'N/A' if np.isnan(v) else f'{v:.4f}'
        except (TypeError, ValueError):
            return f'{v:.4f}'

    W = 46
    print('=' * W)
    print('    DeepSAD Test Results')
    print('=' * W)
    print(f'  Dataset      : {dataset_name}')
    print(f'  Test samples : {len(y_true_bin)}')
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

    # Save results, model, and configuration
    deepSAD.save_results(export_json=model_path + '/results.json')
    deepSAD.save_model(export_model=model_path + '/model.tar', save_ae=pretrain)
    cfg.save_config(export_json=model_path + '/config.json')


if __name__ == '__main__':
    dataset_name = 'spoofing_physical'
    net_name = 'spoof_mlp'
    xp_path = './log/DeepSAD/spoofing'
    data_path = './data'
    lr = 0.0001
    eta = 5.0
    n_epochs = 300
    lr_milestone = [50]
    batch_size = 64
    weight_decay = 0.5e-6
    pretrain = False
    ae_lr = 0.0001
    ae_n_epochs = 150
    ae_batch_size = 128
    ae_weight_decay = 0.5e-3
    normal_class = 0
    known_outlier_class = 1
    n_known_outlier_classes = 1

    wandb.login()
    wandb.init(
        project='PIAD_Ext',
        name='DeepSAD_vanilla',
        config={
            'dataset': dataset_name,
            'lr': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'physical': False,
            'n_epochs': n_epochs,
            'pretrain': pretrain,
        }
    )

    ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
    seed = wandb.config.seed

    rko = str(ratio_known_outlier).replace('.', '')
    rp  = str(ratio_pollution).replace('.', '')
    model_path = f'./saved_model/vanilla/model_{rko}_{rp}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    setting.init([512, 512, 1024])

    main(dataset_name, net_name, xp_path, data_path, eta=eta,
         ratio_known_outlier=ratio_known_outlier,
         ratio_pollution=ratio_pollution,
         ratio_known_normal=ratio_known_normal,
         lr=lr, n_epochs=n_epochs, lr_milestone=lr_milestone,
         weight_decay=weight_decay, pretrain=pretrain,
         ae_lr=ae_lr, ae_n_epochs=ae_n_epochs,
         batch_size=batch_size, ae_batch_size=ae_batch_size,
         ae_weight_decay=ae_weight_decay, normal_class=normal_class,
         known_outlier_class=known_outlier_class,
         n_known_outlier_classes=n_known_outlier_classes, seed=seed)

wandb.finish()
