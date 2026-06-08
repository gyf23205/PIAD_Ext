"""
Evaluate a saved TimesNet checkpoint.

Usage:
  python src/test_TimesNet.py --checkpoint ./saved_model/timesnet_checkpoint.pt --dataset ALFA
  python src/test_TimesNet.py --checkpoint ./saved_model/timesnet_checkpoint.pt --dataset Pegasus
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from main_TimesNet import Exp_Anomaly_Detection, DATASET_CONFIGS
from datasets.main import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a TimesNet checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="ALFA", choices=list(DATASET_CONFIGS))
    p.add_argument("--data_path", default="./data")
    p.add_argument("--seed", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_args = ckpt['args']
    threshold = ckpt['threshold']
    # Override seed if different (to match original dataset split)
    model_args['seed'] = args.seed

    exp = Exp_Anomaly_Detection(model_args)
    exp.model.load_state_dict(ckpt['net_dict'])
    exp.model.eval()

    ratio_pollution = model_args['ratio_pollution']
    train_loader, _, test_loader = exp.get_data(args.dataset, args.data_path, ratio_pollution)

    # Run test using saved threshold
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    import torch.nn as nn

    anomaly_criterion = nn.MSELoss(reduce=False)

    attens_energy = []
    test_labels = []
    exp.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            sample = batch[0].float().to(exp.device)
            target = batch[1]
            sample = sample.view(sample.size(0), model_args['seq_len'], model_args['enc_in'])
            outputs = exp.model(sample)
            score = anomaly_criterion(sample, outputs).mean(dim=(-1, -2))
            attens_energy.append(score.detach().cpu().numpy())
            test_labels.append(target.numpy())

    test_energy = np.concatenate(attens_energy).reshape(-1)

    test_labels_np = np.concatenate(test_labels, axis=0)
    if test_labels_np.ndim == 2:
        gt = (test_labels_np.sum(axis=-1) > 0).astype(int)
    else:
        gt = (test_labels_np > 0).astype(int)

    # Auto-detect inverted score direction (same logic as main_TimesNet.py).
    try:
        raw_auc = roc_auc_score(gt, test_energy)
    except ValueError:
        raw_auc = float('nan')

    score_sign = 1.0
    if not np.isnan(raw_auc) and raw_auc < 0.5:
        score_sign = -1.0

    effective_energy    = score_sign * test_energy
    effective_threshold = score_sign * threshold  # threshold loaded from checkpoint

    pred = (effective_energy > effective_threshold).astype(int)
    roc_auc = raw_auc if score_sign == 1.0 else 1.0 - raw_auc

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        gt, pred, average='binary', zero_division=0)

    width = 40
    print("=" * width)
    print("    TimesNet Test Results")
    print("=" * width)
    print(f"  Dataset   : {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Threshold : {threshold:.6f}")
    print("-" * width)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F-score   : {f_score:.4f}")
    print(f"  ROC AUC   : {roc_auc:.4f}")
    print("=" * width)


if __name__ == "__main__":
    main()
