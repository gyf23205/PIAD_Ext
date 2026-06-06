import torch
import logging
import random
import numpy as np
from datetime import datetime
import wandb
import setting

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset
import os

def main(dataset_name, net_name, xp_path, data_path, load_config=None, load_model=None, load_path=None, num_threads=0,
         n_jobs_dataloader=0, optimizer_name='adam'):

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
    logger.info('Known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Set seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % seed)

    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    # print(seed)
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_classes, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(seed))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(eta)
    deepSAD.net_name = net_name
    # deepSAD.set_network(net_name)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_path, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_path)

    logger.info('Pretraining: %s' % pretrain)
    # if pretrain:
    #     # Log pretraining details
    #     logger.info('Pretraining optimizer: %s' % ae_optimizer_name)
    #     logger.info('Pretraining learning rate: %g' % ae_lr)
    #     logger.info('Pretraining epochs: %d' % ae_n_epochs)
    #     logger.info('Pretraining learning rate scheduler milestones: %s' % (ae_lr_milestone,))
    #     logger.info('Pretraining batch size: %d' % ae_batch_size)
    #     logger.info('Pretraining weight decay: %g' % ae_weight_decay)

    #     # Pretrain model on dataset (via autoencoder)
    #     deepSAD.pretrain(dataset,
    #                      optimizer_name=ae_optimizer_name,
    #                      lr=ae_lr,
    #                      n_epochs=ae_n_epochs,
    #                      lr_milestones=ae_lr_milestone,
    #                      batch_size=ae_batch_size,
    #                      weight_decay=ae_weight_decay,
    #                      device=device,
    #                      n_jobs_dataloader=n_jobs_dataloader)

    #     # Save pretraining results
    #     deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')

    # Log training details
    logger.info('Training optimizer: %s' % optimizer_name)
    logger.info('Training learning rate: %g' % lr)
    logger.info('Training epochs: %d' % n_epochs)
    logger.info('Training learning rate scheduler milestones: %s' % (lr_milestone,))
    logger.info('Training batch size: %d' % batch_size)
    logger.info('Training weight decay: %g' % weight_decay)

    # Train model on dataset
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

    # Test model
    deepSAD.test_physical(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader) # Need to comment this line if want to save the pred branch and also the end of train_physical

    if save:
        # Save results, model, and configuration
        deepSAD.save_results(export_json=model_path + f'/results_physical_seed{seed}.json')
        deepSAD.save_model(export_model=model_path + f'/model_physical_best_seed{seed}.tar', save_ae=pretrain)
        cfg.save_config(export_json=model_path + f'/config_physical_seed{seed}.json')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set training to be deterministic
    seed = 10
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    dataset_name = 'spoofing_multi_profile'
    net_name = 'spoof_mlp'
    data_path = './data'
    lr = 0.0001
    eta = 6.9264986318494515
    n_epochs = 700
    lr_milestone = [200, 400, 600, 800]
    batch_size = 128
    weight_decay = 0.5e-6
    pretrain = False
    tau = 0.5
    aug_mode = 'gaussian'   # 'gaussian' | 'nngmix' | 'both'
    save = True
    normal_class = 0

    coeff = {
            'sad': 1.0,
            'pred': 4.8,
            'dir': 5.0,
            'cluster': 1.7
        }

    # Parse args first so all derived paths use the actual runtime values
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--known_outlier_classes', type=int, nargs='+', default=None,
    #                help='Known anomaly class indices (overrides dataset default)')
    
    # args = parser.parse_args()
    # seed              = args.seed
    # dataset_name      = args.dataset
    # ratio_known_outlier = args.ratio_known_outlier
    # ratio_known_normal  = args.ratio_known_normal
    # ratio_pollution     = args.ratio_pollution

    wandb.login()
    wandb.init(
        project='PIAD_Ext',
        name='PCAD_spoofing_multi_anomalies',
        config={
            'dataset': dataset_name,
            'lr': lr,
            'batch size': batch_size,
            'weight decay': weight_decay,
            'physical': True,
            'pretrain': pretrain,
            'tau': tau,
            "seed":seed
        }
    )

    # ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
    # seed = wandb.config.seed
    ratio_pollution, ratio_known_outlier, ratio_known_normal = 0.05, 0.05, 0.05
    seed = 43
    rko = str(ratio_known_outlier).replace('.', '')
    rp  = str(ratio_pollution).replace('.', '')
    known_outlier_classes = [1] if ratio_known_outlier > 0 else []
    n_known_outlier_classes = len(known_outlier_classes)

    # Derive log and model paths from dataset
    dataset_dir_map = {'spoofing_multi_profile': 'Pegasus', 'ALFA': 'ALFA'}
    dataset_dir = dataset_dir_map.get(dataset_name, dataset_name)
    xp_path = f'./log/{dataset_dir}'
    model_path = f"./saved_model/physical/{dataset_dir}/model_{rko}_{rp}_{coeff['sad']}_{coeff['pred']}_{coeff['dir']}_{coeff['cluster']}"
    model_path = '.' + model_path[1:].replace('.', 'd')
    print(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    setting.init([256, 512, 64, 2.0]) # hd1, hd2, rep, T

    main(dataset_name, net_name, xp_path, data_path)
wandb.finish()