import torch
import logging
import random
import numpy as np
from datetime import datetime
import wandb
import setting

from sklearn.metrics import matthews_corrcoef
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from utils.metrics import compute_affiliation_metrics
from DeepSAD import DeepSAD
from datasets.main import load_dataset
import os

def main(dataset_name, net_name, xp_path, data_path, load_config=None, load_model=None, load_path=None, eta=1.0,
         ratio_known_normal=0.0, ratio_known_outlier=0.0, ratio_pollution=0.0, device='cuda', seed=-1,
         optimizer_name='adam', lr=0.001, n_epochs=50, lr_milestone=50, batch_size=128, weight_decay=1e-6,
         pretrain=True, ae_optimizer_name='adam', ae_lr=0.001, ae_n_epochs=100, ae_lr_milestone=[0], ae_batch_size=128, ae_weight_decay=1e-6,
         num_threads=0, n_jobs_dataloader=0, normal_class=0, known_outlier_class=1, n_known_outlier_classes=0, weight_pred=5):
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
    deepSAD.train_physical(dataset,
                  optimizer_name=optimizer_name,
                  lr=lr,
                  n_epochs=n_epochs,
                  lr_milestones=lr_milestone,
                  batch_size=batch_size,
                  weight_decay=weight_decay,
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader,
                  weight_pred=weight_pred)

    # Test model
    deepSAD.test_physical(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader) # Need to comment this line if want to save the pred branch and also the end of train_physical

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
    deepSAD.save_results(export_json=model_path + f'/results_physical_res.json')
    deepSAD.save_model(export_model=model_path + f'/model_physical_res.tar', save_ae=pretrain)
    cfg.save_config(export_json=model_path + f'/config_physical_res.json')


if __name__ == '__main__':
    # Log in wandb and setup hyperparameters
    wandb.login(key='1888b9830153065d084181ffc29812cd1011b84b')

    dataset_name = 'spoofing_state_only'
    net_name = 'spoof_mlp_res'
    xp_path = './log/DeepSAD/spoofing_physical' # Log path
    data_path = './data'
    ratio_known_outlier = 0.003
    ratio_pollution = 0.05
    rko = str(ratio_known_outlier).replace('.','')
    rp = str(ratio_pollution).replace('.','')
    model_path = f'./saved_model/physical_res/model_{rko}_{rp}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    lr = 0.0001
    eta = 9.6450178070626
    n_epochs = 300
    lr_milestone = [50]
    batch_size = 128
    weight_decay = 0.5e-6
    pretrain = False
    ae_lr = 0.0001
    ae_n_epochs = 150
    ae_batch_size = 128
    ae_weight_decay = 0.5e-3
    normal_class = 0
    known_outlier_class = 1
    weight_pred = 7.343276666841756
    n_known_outlier_classes = 1 # Number of known outlier classes. If 0, no anomalies are known. 
                                # If 1, outlier class as specified in --known_outlier_class option.
                                # If > 1, the specified number of outlier classes will be sampled at random.
    
    wandb.init(
        project='PIAD',
        name='Physical_res',
        config={
            'dataset':'scaled',
           'ratio_known_outlier': ratio_known_outlier,
           'ratio_pollution': ratio_pollution,
           'lr': lr,
           'batch size': batch_size,
           'weight decay': weight_decay,
           'weight pred': weight_pred, 
           'physical': True,
           'pretrain': pretrain
        }
    )

    # hypers = wandb.config
    setting.init([256, 256, 512])
    # Make the code deterministic
    seed = 4

    main(dataset_name, net_name, xp_path, data_path, eta=eta, ratio_known_outlier=ratio_known_outlier,
          ratio_pollution=ratio_pollution, lr=lr, n_epochs=n_epochs, lr_milestone=lr_milestone,
          weight_decay=weight_decay, pretrain=pretrain, ae_lr=ae_lr, ae_n_epochs=ae_n_epochs,
          batch_size=batch_size, ae_batch_size=ae_batch_size, ae_weight_decay=ae_weight_decay, normal_class=normal_class,
          known_outlier_class=known_outlier_class, n_known_outlier_classes=n_known_outlier_classes,seed=seed, weight_pred=weight_pred)
wandb.finish()