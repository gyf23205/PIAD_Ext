"""
Entry point for the TimesNet unsupervised anomaly detection baseline.

TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
(ICLR 2023, Wu et al.)  https://openreview.net/pdf?id=ju_Uqw384Oq

TimesNet is unsupervised: it reconstructs the input via MSE and uses the
reconstruction error as the anomaly score.  Labels are only used at evaluation.

Usage examples:
  python src/main_TimesNet.py --dataset ALFA
  python src/main_TimesNet.py --dataset Pegasus --train_epochs 20
  python src/main_TimesNet.py --dataset ALFA --save_path ./saved_model/timesnet_alfa.pt
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import optim

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from base.exp_basic import Exp_Basic
from baselines.util_TimesNet import EarlyStopping, adjust_learning_rate
from datasets.main import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from utils.metrics import compute_affiliation_metrics


DATASET_CONFIGS = {
    'Pegasus': {
        'known_outlier_classes': [1, 3, 4, 7],
        'n_known_outlier_classes': 4,
        'ratio_known_normal': 0.0,
        'ratio_known_outlier': 0.0,
        'ratio_pollution': 0.1,
        'enc_in': 44,
        'c_out': 44,
        'seq_len': 20,
    },
    'ALFA': {
        'known_outlier_classes': [1, 3, 4, 6],
        'n_known_outlier_classes': 4,
        'ratio_known_normal': 0.0,
        'ratio_known_outlier': 0.0,
        'ratio_pollution': 0.1,
        'enc_in': 47,
        'c_out': 47,
        'seq_len': 40,
    },
    'spoofing_multi_profile': {
        'known_outlier_classes': [1],
        'n_known_outlier_classes': 1,
        'ratio_known_normal': 0.0,
        'ratio_known_outlier': 0.0,
        'ratio_pollution': 0.1,
        'enc_in': 12,
        'c_out': 12,
        'seq_len': 100,
    },
    'spoofing_wind': {
        'known_outlier_classes': [1],
        'n_known_outlier_classes': 1,
        'ratio_known_normal': 0.0,
        'ratio_known_outlier': 0.0,
        'ratio_pollution': 0.1,
        'enc_in': 12,
        'c_out': 12,
        'seq_len': 100,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="TimesNet unsupervised baseline for PIAD_Ext")
    p.add_argument("--dataset", default="ALFA", choices=list(DATASET_CONFIGS))
    p.add_argument("--data_path", default="./data")
    p.add_argument("--seed", type=int, default=4)
    # Training
    p.add_argument("--train_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lradj", default="type2")
    p.add_argument("--ratio_pollution", type=float, default=None,
                   help="Override dataset-default anomaly ratio for threshold")
    # Model architecture
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--num_kernels", type=int, default=3)
    p.add_argument("--e_layers", type=int, default=3)
    p.add_argument("--embed", default="fixed")
    p.add_argument("--freq", default="h")
    p.add_argument("--dropout", type=float, default=0.0)
    # Output
    p.add_argument("--save_path", default="./saved_model/timesnet_checkpoint.pt")
    p.add_argument("--no_save", action="store_true")
    return p.parse_known_args()[0]


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device(
            'cuda:{}'.format(self.args['gpu']) if self.args['use_gpu'] else 'cpu')
        self.model = self._build_model().to(self.device)
        self.model_optim = self._select_optimizer()
        self.criterion = self._select_criterion()

    def _build_model(self):
        model = self.model_dict['TimesNet'].Model(self.args).float()
        if self.args['use_multi_gpu'] and self.args['use_gpu']:
            model = nn.DataParallel(model, device_ids=self.args['device_ids'])
        return model

    def get_data(self, dataset_name, data_path, ratio_pollution=None):
        dcfg = DATASET_CONFIGS[dataset_name]
        dataset = load_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            normal_class=0,
            known_outlier_class=tuple(dcfg['known_outlier_classes']),
            n_known_outlier_classes=dcfg['n_known_outlier_classes'],
            ratio_known_normal=dcfg['ratio_known_normal'],
            ratio_known_outlier=dcfg['ratio_known_outlier'],
            ratio_pollution=ratio_pollution or dcfg['ratio_pollution'],
            random_state=np.random.RandomState(self.args['seed']),
        )
        return dataset.loaders(batch_size=self.args['batch_size'])

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args['learning_rate'])

    def _select_criterion(self):
        return nn.MSELoss()

    def _reshape(self, sample):
        """Reshape flat (B, seq_len*enc_in) → (B, seq_len, enc_in)."""
        return sample.view(sample.size(0), self.args['seq_len'], self.args['enc_in'])

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                sample = batch[0].float().to(self.device)
                sample = self._reshape(sample)
                outputs = self.model(sample)
                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs.detach().cpu(), sample.detach().cpu())
                total_loss.append(loss.item())
        self.model.train()
        return np.average(total_loss)

    def train(self, train_loader, vali_loader, test_loader, model_path):
        os.makedirs(model_path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=30, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        time_now = time.time()
        for epoch in range(self.args['train_epochs']):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                sample = batch[0].float().to(self.device)
                sample = self._reshape(sample)
                outputs = self.model(sample)
                f_dim = -1 if self.args['features'] == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, sample)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args['train_epochs'] - epoch) * train_steps - i)
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}"
                          f"  speed: {speed:.4f}s/iter; left: {left_time:.1f}s")
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)
            print(f"Epoch {epoch+1} | cost: {time.time()-epoch_time:.1f}s "
                  f"| Train: {train_loss:.7f}  Vali: {vali_loss:.7f}  Test: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(model_path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, train_loader, test_loader):
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) collect train reconstruction errors to set threshold
        self.model.eval()
        attens_energy = []
        with torch.no_grad():
            for batch in train_loader:
                sample = batch[0].float().to(self.device)
                sample = self._reshape(sample)
                outputs = self.model(sample)
                score = self.anomaly_criterion(sample, outputs).mean(dim=(-1, -2))
                attens_energy.append(score.detach().cpu().numpy())
        train_energy = np.concatenate(attens_energy).reshape(-1)

        # (2) collect test reconstruction errors + labels
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for batch in test_loader:
                sample = batch[0].float().to(self.device)
                target = batch[1]   # (B, n_ac) multi-hot
                sample = self._reshape(sample)
                outputs = self.model(sample)
                score = self.anomaly_criterion(sample, outputs).mean(dim=(-1, -2))
                attens_energy.append(score.detach().cpu().numpy())
                test_labels.append(target.numpy())

        test_energy = np.concatenate(attens_energy).reshape(-1)

        # (3) Build gt BEFORE threshold so the actual anomaly fraction drives
        #     the percentile cut instead of the dataset-level ratio_pollution.
        test_labels_np = np.concatenate(test_labels, axis=0)  # (N, n_ac)
        if test_labels_np.ndim == 2:
            gt = (test_labels_np.sum(axis=-1) > 0).astype(int)
        else:
            gt = (test_labels_np > 0).astype(int)

        anomaly_fraction = float(gt.mean())
        combined_energy = np.concatenate([train_energy, test_energy])
        threshold = np.percentile(combined_energy, 100 - anomaly_fraction * 100)
        print(f"Threshold: {threshold:.6f}  (anomaly fraction: {anomaly_fraction:.4f})")

        # Auto-detect inverted score direction.
        # For GPS-spoofing data, smooth fake signals reconstruct better than
        # turbulent normal flight, so anomaly MSE < normal MSE (AUC < 0.5).
        try:
            raw_auc = roc_auc_score(gt, test_energy)
        except ValueError:
            raw_auc = float('nan')

        score_sign = 1.0
        if not np.isnan(raw_auc) and raw_auc < 0.5:
            score_sign = -1.0
            print(f"Inverted scores detected (raw AUC={raw_auc:.4f}); negating for classification.")

        effective_energy = score_sign * test_energy
        effective_train  = score_sign * train_energy
        combined_eff     = np.concatenate([effective_train, effective_energy])
        threshold_eff    = np.percentile(combined_eff, 100 - anomaly_fraction * 100)

        # Binary classification — no adjustment(), data is not temporally ordered.
        pred = (effective_energy > threshold_eff).astype(int)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt, pred, average='binary', zero_division=0)
        roc_auc = raw_auc if score_sign == 1.0 else 1.0 - raw_auc
        if np.isnan(roc_auc):
            roc_auc = float('nan')

        aff = compute_affiliation_metrics(gt, pred)
        mcc = float(matthews_corrcoef(gt, pred))
        nan = float('nan')
        metrics = {
            'test_auc':    roc_auc,
            'f1_macro':    nan,      # binary method — no per-class multi-hot predictions
            'f1_weighted': nan,
            'mh_acc':      nan,
            'mh_recall':   nan,
            'bin_f1':      float(f_score),
            'bin_acc':     float(accuracy),
            'bin_recall':  float(recall),
            'mcc':         mcc,
            'p_aff':       aff['p_aff'],
            'r_aff':       aff['r_aff'],
            'f_aff':       aff['f_aff'],
            # internal extras kept for checkpoint / logging convenience
            '_n_samples':  int(len(gt)),
            '_precision':  float(precision),
        }
        # Return original unsigned threshold so test_TimesNet.py can apply
        # the same sign-flip logic when loading from checkpoint.
        return threshold, test_energy, metrics


def main(ratio_pollution=None, ratio_known_outlier=None, ratio_known_normal=None, seed=None):
    args = parse_args()
    if ratio_pollution is not None: args.ratio_pollution = ratio_pollution
    if seed            is not None: args.seed            = seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dcfg = DATASET_CONFIGS[args.dataset]
    ratio_pollution = args.ratio_pollution or dcfg['ratio_pollution']

    model_args = {
        'use_gpu': torch.cuda.is_available(),
        'gpu_type': 'cuda',
        'gpu': 0,
        'use_multi_gpu': False,
        'seed': args.seed,
        'train_epochs': args.train_epochs,
        'learning_rate': args.learning_rate,
        'lradj': args.lradj,
        'features': 'M',
        'ratio_pollution': ratio_pollution,
        'seq_len': dcfg['seq_len'],
        'pred_len': 0,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'top_k': args.top_k,
        'num_kernels': args.num_kernels,
        'e_layers': args.e_layers,
        'enc_in': dcfg['enc_in'],
        'c_out': dcfg['c_out'],
        'embed': args.embed,
        'freq': args.freq,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
    }

    exp = Exp_Anomaly_Detection(model_args)
    train_loader, vali_loader, test_loader = exp.get_data(
        args.dataset, args.data_path, ratio_pollution)

    tmp_model_path = './saved_model/TimesNet_tmp'
    exp.train(train_loader, vali_loader, test_loader, tmp_model_path)
    threshold, test_energy, metrics = exp.test(train_loader, test_loader)

    def _fmt(v):
        try:
            return 'N/A' if np.isnan(v) else f'{v:.4f}'
        except (TypeError, ValueError):
            return f'{v:.4f}'

    W = 46
    print('=' * W)
    print('    TimesNet Test Results')
    print('=' * W)
    print(f'  Dataset      : {args.dataset}')
    print(f'  Test samples : {metrics["_n_samples"]}')
    print('-' * W)
    print('  -- Multi-hot --')
    print(f'  F1 macro      : {_fmt(metrics["f1_macro"])}')
    print(f'  F1 weighted   : {_fmt(metrics["f1_weighted"])}')
    print(f'  MH accuracy   : {_fmt(metrics["mh_acc"])}')
    print(f'  MH recall     : {_fmt(metrics["mh_recall"])}')
    print('-' * W)
    print('  -- Binary --')
    print(f'  AUC           : {_fmt(metrics["test_auc"])}')
    print(f'  F1            : {_fmt(metrics["bin_f1"])}')
    print(f'  Accuracy      : {_fmt(metrics["bin_acc"])}')
    print(f'  Recall        : {_fmt(metrics["bin_recall"])}')
    print(f'  MCC           : {_fmt(metrics["mcc"])}')
    print('-' * W)
    print('  -- Affiliation --')
    print(f'  P_aff (UAff)  : {_fmt(metrics["p_aff"])}')
    print(f'  R_aff (NAff)  : {_fmt(metrics["r_aff"])}')
    print(f'  F_aff         : {_fmt(metrics["f_aff"])}')
    print('=' * W)

    wandb.log({
        'test_auc':     metrics['test_auc'],
        'f1_macro':     metrics['f1_macro'],
        'f1_weighted':  metrics['f1_weighted'],
        'mh_acc':       metrics['mh_acc'],
        'mh_recall':    metrics['mh_recall'],
        'bin_f1':       metrics['bin_f1'],
        'bin_acc':      metrics['bin_acc'],
        'bin_recall':   metrics['bin_recall'],
        'mcc':          metrics['mcc'],
        'p_aff':        metrics['p_aff'],
        'r_aff':        metrics['r_aff'],
        'f_aff':        metrics['f_aff'],
    })

    if not args.no_save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        torch.save({
            'net_dict': exp.model.state_dict(),
            'threshold': threshold,
            'args': model_args,
        }, args.save_path)
        print(f"Checkpoint saved to {args.save_path}")


if __name__ == '__main__':
    wandb.login()
    wandb.init(
        project='PIAD_Ext',
        name='TimesNet',
        config={
            'train_epochs': 20,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'd_model': 128,
            'd_ff': 128,
        }
    )
    ratio_pollution, ratio_known_outlier, ratio_known_normal = wandb.config.ratios
    seed = wandb.config.seed
    main(ratio_pollution=ratio_pollution, ratio_known_outlier=ratio_known_outlier,
         ratio_known_normal=ratio_known_normal, seed=seed)
wandb.finish()
