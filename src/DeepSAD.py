import json
import torch
import wandb
import copy

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder, build_network_physical
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer import AETrainer
from optim.DeepSAD_trainer_physical import DeepSADTrainerPhysical


class DeepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.c = None  # hypersphere center c
        self.roc_curve = None  # Best decision threshold according to validtions

        self.net_name = None
        self.net = None  # neural network phi
        # self.net_store = None

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        # self.ae_net_store = None
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'test_loss':             None,
            'test_auc':              None,
            'test_time':             None,
            'test_f1_macro_mh':      None,
            'test_f1_micro_mh':      None,
            'test_hamming_acc':      None,
            'test_subset_acc':       None,
            'test_f1_binary':        None,
            'test_precision_binary': None,
            'test_recall_binary':    None,
            'test_acc_binary':       None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    # def set_network_physical(self, net_name):
    #     '''Build network with the state predication part'''
    #     self.net_name = net_name
    #     self.net = build_network_physical(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0, with_physical=False):
        """Trains the Deep SAD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net, best_auc = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list
        self.roc_curve = self.trainer.roc_curve
        wandb.log({'best_val_auc': best_auc})

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)

        # Get results (wandb logging is handled in main_SAD.py after metric computation)
        self.results['test_auc']    = self.trainer.test_auc
        self.results['test_time']   = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def test_physical(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        self.trainer.test(dataset, self.net)

        # Get results (wandb logging is handled in main_res.py after full metric computation)
        self.results['test_loss']             = self.trainer.test_loss
        self.results['test_auc']              = self.trainer.test_auc
        self.results['test_time']             = self.trainer.test_time
        self.results['test_f1_macro_mh']      = self.trainer.test_f1_macro_mh
        self.results['test_f1_micro_mh']      = self.trainer.test_f1_micro_mh
        self.results['test_f1_weighted_mh']   = self.trainer.test_f1_weighted_mh
        self.results['test_hamming_acc']      = self.trainer.test_hamming_acc
        self.results['test_subset_acc']       = self.trainer.test_subset_acc
        self.results['test_mh_recall']        = self.trainer.test_mh_recall
        self.results['test_f1_binary']        = self.trainer.test_f1_binary
        self.results['test_precision_binary'] = self.trainer.test_precision_binary
        self.results['test_recall_binary']    = self.trainer.test_recall_binary
        self.results['test_acc_binary']       = self.trainer.test_acc_binary

    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        # self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()
    
    def pretrain_physical(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                n_jobs_dataloader: int = 0, weight_pred=5):
        """Train with system dynamics"""

        # Set physics-informed network
        self.net = build_network_physical(self.net_name)


        # Train
        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainerPhysical(self.c, self.eta, optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.net = self.trainer.initilize_physical(dataset, self.net, weight_pred)

        # Get train results
        self.results['train_time'] = self.trainer.train_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.create_from_physically_informed()


    def train_physical(self, dataset: BaseADDataset, n_outlier_classes: int, known_outlier_classes, coeff: dict, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                n_jobs_dataloader: int = 0, tau=0.1, model_path=None, save=False,
                aug_mode: str = 'gaussian', nngmix_cfg: dict | None = None,
                eval_rule: str = 'threshold'):
        """Train with system dynamics"""

        # Set autoencoder network
        self.net = build_network_physical(self.net_name)

        # Derive the full ordered outlier_classes from the dataset
        outlier_classes = dataset.outlier_classes

        # Train
        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainerPhysical(n_outlier_classes, known_outlier_classes, outlier_classes,
                                    coeff, optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader, tau=tau,
                                    aug_mode=aug_mode, nngmix_cfg=nngmix_cfg, eval_rule=eval_rule)
        self.net, best_auc = self.trainer.train(dataset, self.net, model_path, save)

        # Get train results
        self.results['train_time'] = self.trainer.train_time
        self.centroids            = self.trainer.centroids
        self.roc_curve            = self.trainer.roc_curve
        self.per_class_thresholds = self.trainer.per_class_thresholds

        self.net_wo_pred = self.create_from_physically_informed()
        wandb.log({'best_auc': best_auc})

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        dict_temp = {k: ae_net_dict['encoder.'+k] for k in net_dict.keys() if 'encoder.'+k in ae_net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(dict_temp)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def create_from_physically_informed(self):
        """Create a network without a state estimator from one trained with estimator"""

        net_temp = build_network(self.net_name)
        net_dict = net_temp.state_dict()
        net_physical_dict = self.net.state_dict()

        # Filter out decoder network keys
        dict_temp = {k: net_physical_dict['encoder.'+k] for k in net_dict.keys() if 'encoder.'+k in net_physical_dict.keys()}
        # Overwrite values in the existing state_dict
        net_dict.update(dict_temp)
        # Load the new state_dict
        net_temp.load_state_dict(net_dict)
        # self.net = copy.deepcopy(net_temp)
        print('created')
        return net_temp

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()
        net_wo_pred_dict = self.net_wo_pred.state_dict() if hasattr(self, 'net_wo_pred') else None
        # ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'centroids': self.centroids,
                    'roc': self.roc_curve,
                    'per_class_thresholds': getattr(self, 'per_class_thresholds', None),
                    'net_dict': net_dict,
                    'net_wo_pred_dict': net_wo_pred_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location, weights_only=False)

        if 'centroids' in model_dict:   # physical model checkpoint
            self.centroids = model_dict['centroids']
        else:                            # standard model checkpoint
            self.c = model_dict['c']

        self.roc_curve = model_dict['roc']
        self.per_class_thresholds = model_dict.get('per_class_thresholds', None)
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)
