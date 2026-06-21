"""Root-cause analysis: average sample-to-centroid distance per class group.

ALFA and Pegasus contain anomalies that hit the same physical part (e.g. ALFA
classes 2 & 3 are both aileron failures) and hybrid anomalies where several
faults are active at once (e.g. Pegasus spoofing + IMU-HF-noise). The
DeepSAD-Physical model learns one centroid per class in embedding space. This
script selects all samples matching a (possibly hybrid) multi-hot class spec,
runs them through the trained encoder, and reports the average squared-Euclidean
distance from each class group to every stored centroid -- a root-cause /
confusion view in embedding space.

Run from the `src/` directory, e.g.:

    python visualization/rca_centroid_distance.py --dataset ALFA \
        --checkpoint ./saved_model/physical/ALFA/model_01_00/model_physical_best_seed0.tar \
        --classes 2,3 1 4
"""
import os
import sys
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless: save figures, no display needed
import matplotlib.pyplot as plt

# Make the src/ package modules importable when run from src/visualization/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import setting
from datasets.main import load_dataset
from networks.main import build_network_physical
from utils.metrics import centroid_distances, centroid_probabilities

# Reuse the canonical net_name / hyperparameter / known-class config.
from main_all import DATASET_CONFIGS

# Class index -> human-readable name (index = anomaly class number 1..7; 0 = normal).
CLASS_NAMES = {
    'ALFA': {
        0: 'normal', 1: 'engine', 2: 'aileron R', 3: 'aileron L',
        4: 'elevator', 5: 'rudder L', 6: 'rudder R', 7: 'rudder zero',
    },
    'Pegasus': {
        0: 'normal', 1: 'spoofing', 2: 'replay', 3: 'gyro bias',
        4: 'motor fault', 5: 'motor delay', 6: 'GPS denial', 7: 'IMU HF noise',
    },
}

N_ANOMALY_CLASSES = 7  # both datasets: outlier_classes = (1..7)


def parse_class_spec(spec: str) -> np.ndarray:
    """Convert a comma-separated class-number group into a 7-d multi-hot vector.

    e.g. "2,3" -> [0,1,1,0,0,0,0]; "0" or "" -> all zeros (the normal group).
    Column index = class_number - 1.
    """
    v = np.zeros(N_ANOMALY_CLASSES, dtype=np.float32)
    for tok in spec.split(','):
        tok = tok.strip()
        if tok == '' or tok == '0':
            continue
        c = int(tok)
        if not (1 <= c <= N_ANOMALY_CLASSES):
            raise ValueError(f'class number {c} out of range 1..{N_ANOMALY_CLASSES} (spec "{spec}")')
        v[c - 1] = 1.0
    return v


def spec_label(spec: str, v: np.ndarray, names: dict) -> str:
    """Readable label for a class-group spec, e.g. '2,3 (aileron R+aileron L)'."""
    active = [i + 1 for i in range(N_ANOMALY_CLASSES) if v[i] == 1.0]
    if not active:
        return '0 (normal)'
    nums = ','.join(str(c) for c in active)
    return f"{nums} ({'+'.join(names.get(c, str(c)) for c in active)})"


def main():
    p = argparse.ArgumentParser(description='Average class-to-centroid distances (RCA).')
    p.add_argument('--dataset', required=True, choices=list(CLASS_NAMES),
                   help='Dataset name.')
    p.add_argument('--checkpoint', required=True,
                   help='Path to a trained .tar checkpoint containing "centroids" and "net_dict".')
    p.add_argument('--classes', nargs='+', required=True,
                   help='Class-group specs, each comma-separated class numbers, e.g. 2,3 1 1,7. '
                        'Use 0 for the normal group.')
    p.add_argument('--data_path', default='./data', help='Dataset root directory.')
    p.add_argument('--eval_rule', default='probability', choices=['threshold', 'probability'],
                   help="Decision rule (mirrors main_all.py): 'probability' (distance -> "
                        "probability distribution, mean per group) or 'threshold' (squared-"
                        "Euclidean + the checkpoint's per-class Youden thresholds, fraction of "
                        "group samples the model assigns to each class).")
    p.add_argument('--metric', default='cosine', choices=['cosine', 'euclidean'],
                   help='Distance metric for the probability rule (default cosine, matching the '
                        'training dir-loss geometry). Ignored for --eval_rule threshold (always '
                        'squared-Euclidean, to match the stored thresholds).')
    p.add_argument('--batch_size', type=int, default=512, help='Embedding batch size.')
    p.add_argument('--out', default=None,
                   help='Plot output path (.pdf) (default: ./figures/<dataset>_rca_distances.pdf).')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = DATASET_CONFIGS[args.dataset]
    names = CLASS_NAMES[args.dataset]

    # setting.rep is read by build_network_physical, so init before building.
    setting.init(cfg['setting_hypers'])

    # ---- Load all samples (train + val + test merged) ----
    # We use the true multi-hot labels from data_direct(); the semi-supervised
    # ratios / known-class args only affect the (discarded) semi_targets, so all
    # are left at their defaults -- the data here is always fully labeled.
    dataset = load_dataset(args.dataset, args.data_path,
                           normal_class=cfg['normal_class'], known_outlier_class=())
    X_tr, y_tr, _, X_te, y_te, X_va, y_va = dataset.data_direct()
    X_all = np.concatenate([X_tr, X_te, X_va]).astype(np.float32)
    Y_all = np.concatenate([y_tr, y_te, y_va]).astype(np.float32)
    print(f'Loaded {len(X_all)} samples ({args.dataset}, all splits merged).')

    # ---- Load checkpoint: centroids + network weights ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'centroids' not in ckpt or ckpt['centroids'] is None:
        raise SystemExit(
            f'Checkpoint {args.checkpoint} has no "centroids" -- it was trained without '
            f'labeled anomalies (ratio_known_outlier=0). Train with --ratio_known_outlier>0.')
    centroids = ckpt['centroids']

    net = build_network_physical(cfg['net_name'])
    net.load_state_dict(ckpt['net_dict'])
    net.to(device).eval()

    # ---- Centroid order + labels: class_ids = [0] + known_outlier_classes ----
    known = list(cfg['known_outlier_classes'])
    cent_keys = ['c_normal'] + [f'c_outlier_{i + 1}' for i in range(len(known))]
    class_ids = [0] + known
    missing = [k for k in cent_keys if k not in centroids]
    if missing:
        raise SystemExit(f'Checkpoint centroids missing keys {missing}; found {list(centroids)}. '
                         f"It was likely trained with a different known-class set than "
                         f"DATASET_CONFIGS['{args.dataset}'].")
    cent_labels = [f'{k}({cid}:{names.get(cid, cid)})' for k, cid in zip(cent_keys, class_ids)]
    C = torch.stack([centroids[k].to(device).float() for k in cent_keys])  # (n_cent, rep)

    # ---- For the threshold rule, fetch the stored decision thresholds ----
    per_class_thr = bin_thr = None
    if args.eval_rule == 'threshold':
        pct = ckpt.get('per_class_thresholds')
        roc = ckpt.get('roc')
        if not pct or roc is None:
            raise SystemExit(
                "Checkpoint has no per-class thresholds / ROC curve, so --eval_rule threshold "
                "is unavailable (these are only saved for models trained with labeled anomalies "
                "via DeepSAD.save_model). Use --eval_rule probability instead.")
        per_class_thr = pct                                    # one per known outlier class
        fpr, tpr, thr = roc
        bin_thr = float(thr[int(np.argmax(tpr - fpr))])        # binary Youden threshold

    def assign_vectors(emb):
        """(b, K+1) per-sample column values under the chosen rule."""
        if args.eval_rule == 'probability':
            d = centroid_distances(emb, C, metric=args.metric)
            return centroid_probabilities(d)                   # probabilities, rows sum to 1
        # threshold rule: squared-Euclidean dist + stored thresholds -> binary "fires"
        d = centroid_distances(emb, C, metric='euclidean')     # col 0 = dist to c_normal
        anomaly = d[:, 0] > bin_thr                            # binary anomaly gate
        out = torch.zeros_like(d)
        out[:, 0] = (~anomaly).float()                         # predicted normal
        for i in range(len(per_class_thr)):
            out[:, i + 1] = (((-d[:, i + 1]) > per_class_thr[i]) & anomaly).float()
        return out                                             # multi-hot fires (not normalised)

    # ---- Per-group aggregate value per centroid ----
    # probability rule -> mean assignment probability; threshold rule -> fraction of
    # group samples the model assigns to each centroid/class.
    rows = []  # (row_label, n_samples, [value per centroid])
    for spec in args.classes:
        v = parse_class_spec(spec)
        mask = (Y_all == v).all(axis=1)
        n = int(mask.sum())
        label = spec_label(spec, v, names)
        if n == 0:
            print(f'WARNING: no samples exactly match group "{label}".')
            rows.append((label, 0, [float('nan')] * len(cent_keys)))
            continue

        Xg = torch.from_numpy(X_all[mask])
        sums = torch.zeros(len(cent_keys), device=device)
        with torch.no_grad():
            for i in range(0, len(Xg), args.batch_size):
                xb = Xg[i:i + args.batch_size].to(device)
                emb, _ = net(xb)  # MLP_Physical -> (emb, x_next)
                sums += assign_vectors(emb).sum(dim=0)
        means = (sums / n).cpu().tolist()
        rows.append((label, n, means))

    # ---- Print matrix ----
    # probability rule uses a 0.3 assignment threshold; the threshold (fraction)
    # rule uses 0.5 to mark the predominant class.
    thr = 0.3 if args.eval_rule == 'probability' else 0.5
    if args.eval_rule == 'probability':
        value_name = f'Mean assignment probability ({args.metric}, threshold {thr})'
        pred_tag   = f'predicted (p>{thr})'
    else:
        value_name = 'Fraction assigned (euclidean + stored thresholds)'
        pred_tag   = f'predominant (frac>{thr})'
    print('\n' + '=' * 100)
    print(f'  {value_name} to each centroid  ({args.dataset})')
    print('=' * 100)
    lbl_w = max([len('class group')] + [len(r[0]) for r in rows]) + 2
    col_w = max(14, max(len(c) for c in cent_labels) + 2)
    header = 'class group'.ljust(lbl_w) + 'n'.rjust(7) + '  ' + ''.join(c.rjust(col_w) for c in cent_labels)
    print(header)
    print('-' * len(header))
    for label, n, means in rows:
        line = label.ljust(lbl_w) + str(n).rjust(7) + '  '
        if n == 0:
            line += ''.join('nan'.rjust(col_w) for _ in means)
            predicted = '-'
        else:
            line += ''.join(f'{m:.4f}'.rjust(col_w) for m in means)
            j = int(np.argmax(means))
            predicted = cent_labels[j] if means[j] > thr else f'none (max={means[j]:.2f})'
        print(line + f'   -> {pred_tag}: {predicted}')
    print('=' * 100)

    # ---- Plot heatmap ----
    out = args.out or os.path.join('./imgs', f'{args.dataset}_rca_{args.eval_rule}.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plot_heatmap(rows, cent_labels, args.dataset, args.eval_rule, args.metric, thr, out)
    print(f'Saved plot to {out}')


def plot_heatmap(rows, cent_labels, dataset, eval_rule, metric, thr, out_path):
    """Heatmap of the per-group value (in [0,1]) for each centroid.

    Value is mean assignment probability (probability rule) or fraction of group
    samples assigned (threshold rule), on a fixed 0-1 scale; each cell is annotated
    and cells above `thr` (the prediction) are outlined.
    """
    M = np.array([r[2] for r in rows], dtype=float)          # (n_groups, n_cent)
    row_labels = [f'{r[0]}\n(n={r[1]})' for r in rows]
    n_rows, n_cols = M.shape

    if eval_rule == 'probability':
        title = f'Mean assignment probability ({dataset}, {metric} distance, threshold {thr})'
        cbar_label = 'assignment probability (0-1)'
    else:
        title = f'Fraction of samples assigned ({dataset}, euclidean + stored thresholds, {thr})'
        cbar_label = 'fraction of group samples (0-1)'

    fig, ax = plt.subplots(figsize=(1.6 * n_cols + 4, 0.9 * n_rows + 2.5))
    cmap = plt.cm.viridis.copy()     # higher value = brighter
    cmap.set_bad(color='lightgrey')  # NaN (no samples)
    im = ax.imshow(M, cmap=cmap, aspect='auto', vmin=0.0, vmax=1.0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(cent_labels, rotation=30, ha='right', fontsize=20)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=20)
    ax.set_xlabel('stored centroid', size=20)
    # ax.set_ylabel('selected class group', size=20)
    # ax.set_title(title, size=20)

    # Annotate probabilities.
    for i in range(n_rows):
        if np.all(np.isnan(M[i])):
            ax.text(n_cols / 2 - 0.5, i, 'no samples', ha='center', va='center',
                    fontsize=20, color='dimgrey')
            continue
        for j in range(n_cols):
            txt_col = 'white' if M[i, j] < 0.55 else 'black'
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    fontsize=20, color=txt_col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04, aspect=12,
                        ticks=np.linspace(0.0, 1.0, 6))
    cbar.set_label(cbar_label, size=20)
    cbar.ax.tick_params(labelsize=18)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
