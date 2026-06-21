"""Plot the percentage gap of the PCAD variants over the best baseline.

For each metric X, dataset, and label-ratio config this computes

    gap = (X_PCAD_variant - X_best_baseline) / X_best_baseline * 100

where ``X_best_baseline`` is the best value among all methods that do NOT have
"PCAD" in their name (best = maximum, since every metric here is
higher-is-better). A positive bar means the PCAD variant beats the best
baseline; a negative bar means the best baseline wins.

Like plot_results_configs.py the figure has one row per metric and one column
per dataset. Within each subplot the x-axis lists the PCAD variants and bars are
grouped by config.

Usage:
    python src/visualization/plot_results_gap.py --metrics bin_f1
    python src/visualization/plot_results_gap.py --metrics bin_f1 mh_recall test_auc
    python src/visualization/plot_results_gap.py --metrics test_auc --baselines RoSAS TimesNet SimAD
"""
import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from plot_results import METRICS, N_CONFIGS, XLSX_PATH, load_results

CONFIG_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']


def split_methods(methods):
    """Split into (pcad_variants, baselines) by whether 'PCAD' is in the name."""
    pcad = [m for m in methods if 'PCAD' in m.upper()]
    baselines = [m for m in methods if 'PCAD' not in m.upper()]
    if not pcad:
        raise ValueError(f'No PCAD variants found among methods: {methods}')
    if not baselines:
        raise ValueError(f'No baseline methods (non-PCAD) found among methods: {methods}')
    return pcad, baselines


def compute_gaps(results, datasets, metrics, pcad_variants, baselines, config_labels):
    """gaps[metric][dataset][cfg][variant] = percentage gap over best baseline."""
    needed = list(pcad_variants) + list(baselines)
    for metric in metrics:
        for dataset in datasets:
            for cfg in range(N_CONFIGS):
                for method in needed:
                    mean, std = results[dataset][cfg].get(method, {}).get(
                        metric, (float('nan'), float('nan')))
                    if math.isnan(mean):
                        raise ValueError(
                            f'Missing value: dataset={dataset}, method={method}, '
                            f'metric={metric}, config={cfg} ({config_labels[cfg]})'
                        )

    gaps = {}
    for metric in metrics:
        gaps[metric] = {}
        for dataset in datasets:
            gaps[metric][dataset] = {}
            for cfg in range(N_CONFIGS):
                best_method = max(baselines, key=lambda b: results[dataset][cfg][b][metric][0])
                best = results[dataset][cfg][best_method][metric][0]
                if best == 0:
                    raise ValueError(
                        f'Best baseline is 0, cannot take percentage gap: '
                        f'dataset={dataset}, metric={metric}, config={cfg}'
                    )
                print(f'best baseline: metric={metric}, dataset={dataset}, '
                      f'cfg={cfg} -> {best_method} ({best:.4f})')
                gaps[metric][dataset][cfg] = {
                    v: (results[dataset][cfg][v][metric][0] - best) / best * 100.0
                    for v in pcad_variants
                }
    return gaps


def plot_gaps(gaps, datasets, metrics, pcad_variants, config_labels, save_dir,
              show=True, save=True):
    n_rows = len(metrics)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows),
                             squeeze=False)
    x = np.arange(len(pcad_variants))
    bar_w = 0.8 / N_CONFIGS
    offsets = (np.arange(N_CONFIGS) - (N_CONFIGS - 1) / 2) * bar_w

    # Shared y-range per row (per metric), with a small symmetric margin.
    row_ylim = {}
    for metric in metrics:
        vals = [gaps[metric][d][cfg][v]
                for d in datasets
                for cfg in range(N_CONFIGS) for v in pcad_variants]
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        row_ylim[metric] = (lo - pad, hi + pad)

    for r, metric in enumerate(metrics):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            for cfg in range(N_CONFIGS):
                vals = [gaps[metric][dataset][cfg][v] for v in pcad_variants]
                ax.bar(x + offsets[cfg], vals, width=bar_w,
                       color=CONFIG_COLORS[cfg % len(CONFIG_COLORS)],
                       edgecolor='black', linewidth=0.5,
                       label=config_labels[cfg])
            ax.axhline(0, color='black', linewidth=1.0)
            ax.set_ylim(row_ylim[metric])
            ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
            ax.set_xticks(x)
            if r == n_rows - 1:
                ax.set_xticklabels(pcad_variants, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])
            if r == 0:
                ax.set_title(dataset)
            if c == 0:
                ax.set_ylabel(f'{metric} gap (%)')
            ax.grid(axis='y', color='0.85', linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=N_CONFIGS, frameon=False)
    fig.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        safe_metrics = '_'.join(re.sub(r'[^\w.-]+', '_', m).strip('_') for m in metrics)
        out_path = os.path.join(save_dir, f'{safe_metrics}_gap.pdf')
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved figure to {out_path}')
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", required=True, choices=["detection", "classification"], help="Choose setting.")
    parser.add_argument('--xlsx', default=XLSX_PATH, help='Path to results.xlsx')
    parser.add_argument('--save-dir', default='./figures', help='Directory to save the figure')
    parser.add_argument('--no-show', action='store_true', help='Only save the figure, do not open a window')
    parser.add_argument('--no-save', action='store_true', help='Only show the figure, do not save it to disk')
    args = parser.parse_args()

    if args.setting == "detection":
        pcad_variants = ["PCAD_full"]
        metrics = ["test_auc", "bin_f1", "bin_acc", "bin_recall", "mcc", "P_aff (UAff)"]
        baselines = ["PCAD_full", "RoSAS", "SimAD", "CATS", "TimesNet"]
    elif args.setting == "classification":
        pcad_variants = ["PCAD_full"]
        metrics = ["f1_macro", "f1_weighted", "mh_acc", "mh_recall"]
        baselines = ["PCAD_full", "DASO", "CCL", "SimPro", "CATS"]
    else:
        parser.error(f'Unknown metric(s) {args.metrics}. Valid metrics: detection, classification')

    results, datasets, all_methods, config_labels = load_results(args.xlsx)

    print(f'PCAD variants: {pcad_variants}')
    print(f'Baselines:     {baselines}')

    gaps = compute_gaps(results, datasets, metrics, pcad_variants, baselines,
                        config_labels)
    plot_gaps(gaps, datasets, metrics, pcad_variants, config_labels,
              args.save_dir, show=not args.no_show, save=not args.no_save)
