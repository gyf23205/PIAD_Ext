"""Plot mean +/- std of one metric for all label-ratio configs in a single figure.

Like plot_results.py, but instead of picking one config, every config is drawn
in each subplot: mean values appear as dots with a different marker shape (and
color) per config, slightly offset horizontally per method, with +/- std error
bars.

Pass several metrics to stack them as rows: the figure has one row per metric
and one column per dataset.

Usage:
    python src/visualization/plot_results_configs.py --metrics bin_f1
    python src/visualization/plot_results_configs.py --metrics bin_f1 mh_recall test_auc
    python src/visualization/plot_results_configs.py --metrics F_aff --methods PCAD RoSAS SimAD
"""
import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from plot_results import CONFIG_BLOCKS, XLSX_PATH, load_results

MARKERS = ['o', 's', '^', 'D', 'v']
CONFIG_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']


def plot_metrics_all_configs(results, datasets, metrics, methods, config_labels,
                             config_blocks, save_dir, show=True, save=True):
    n_cfg = len(config_blocks)
    # Fail loudly on any missing value for the selected methods/metrics.
    for metric in metrics:
        for dataset in datasets:
            for block in config_blocks:
                for method in methods:
                    mean, std = results[dataset][block].get(method, {}).get(metric, (float('nan'), float('nan')))
                    if math.isnan(mean) or math.isnan(std):
                        raise ValueError(
                            f'Missing value: dataset={dataset}, method={method}, '
                            f'metric={metric}, config={block} ({config_labels[block]})'
                        )

    n_rows = len(metrics)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.7 * n_rows),
                             squeeze=False)
    x = np.arange(len(methods))
    offsets = np.linspace(-0.25, 0.25, n_cfg)
    for r, metric in enumerate(metrics):
        # Shared y-range across all datasets (columns) in this row, computed
        # from the data including the +/- std error bars.
        row_lo, row_hi = float('inf'), float('-inf')
        for dataset in datasets:
            for block in config_blocks:
                for m in methods:
                    mean, std = results[dataset][block][m][metric]
                    row_lo = min(row_lo, mean - std)
                    row_hi = max(row_hi, mean + std)
        pad = 0.05 * (row_hi - row_lo) if row_hi > row_lo else 0.1
        row_ylim = (row_lo - pad, row_hi + pad)
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            for i, block in enumerate(config_blocks):
                means = [results[dataset][block][m][metric][0] for m in methods]
                stds = [results[dataset][block][m][metric][1] for m in methods]
                ax.errorbar(x + offsets[i], means, yerr=stds,
                            fmt=MARKERS[i % len(MARKERS)],
                            color=CONFIG_COLORS[i % len(CONFIG_COLORS)],
                            markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                            linestyle='none', capsize=3, elinewidth=1.2,
                            label=config_labels[block])
            # Vertical separators at the midpoints between adjacent method groups
            for xb in x[:-1] + 0.5:
                ax.axvline(xb, color='0.7', linewidth=0.8, linestyle='--', zorder=0)
            ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
            ax.set_xticks(x)
            # Only label the x-axis methods on the bottom row to reduce clutter.
            if r == n_rows - 1:
                ax.set_xticklabels(methods, rotation=45, ha='right')
            else:
                ax.set_xticklabels([])
            if r == 0:
                ax.set_title(dataset)
            if c == 0:
                ax.set_ylabel(metric)
            ax.set_ylim(row_ylim)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.grid(axis='y', color='0.85', linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=n_cfg, frameon=False)
    fig.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        safe_metrics = '_'.join(re.sub(r'[^\w.-]+', '_', m).strip('_') for m in metrics)
        out_path = os.path.join(save_dir, f'{safe_metrics}_allcfg.pdf')
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved figure to {out_path}')
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('--metrics', nargs='+', required=True,
    #                     help=f'Metric(s) to plot, one row per metric. Choose from: {METRICS}')
    parser.add_argument("--setting", required=True, choices=["detection", "classification"], help="Choose setting.")
    # parser.add_argument('--methods', nargs='+', default=None,
    #                     help='Methods to include (default: all methods found in the file)')
    parser.add_argument('--xlsx', default=XLSX_PATH, help='Path to results.xlsx')
    parser.add_argument('--save-dir', default='./imgs', help='Directory to save the figure')
    parser.add_argument('--no-show', action='store_true', help='Only save the figure, do not open a window')
    parser.add_argument('--no-save', action='store_true', help='Only show the figure, do not save it to disk')
    args = parser.parse_args()

    if args.setting == "detection":
        metrics = ["test_auc", "bin_f1", "bin_acc", "bin_recall", "mcc", "P_aff (UAff)"]
        methods = ["PCAD_full", "PCAD_nngmix", "PCAD_partialSAD", "RoSAS", "SimAD", "CATS", "TimesNet"]
    elif args.setting == "classification":
        metrics = ["f1_macro", "f1_weighted", "mh_acc", "mh_recall"]
        methods = ["PCAD_full", "PCAD_nngmix", "PCAD_partialSAD", "DASO", "CCL", "SimPro", "CATS"]
    else:
        parser.error(f'Unknown metric(s) {args.metrics}. Valid metrics: detection, classification')

    results, datasets, all_methods, config_labels = load_results(args.xlsx)

    config_blocks = CONFIG_BLOCKS[args.setting]
    missing = [b for b in config_blocks if b >= len(config_labels) or config_labels[b] is None]
    if missing:
        parser.error(
            f'Setting "{args.setting}" needs column block(s) {missing}, but the '
            f'spreadsheet only has {len(config_labels)} block(s). '
            f'Did you add the rp:0.01 rko:0.01 rkn:0.01 columns?'
        )

    plot_metrics_all_configs(results, datasets, metrics, methods, config_labels,
                             config_blocks, args.save_dir, show=not args.no_show,
                             save=not args.no_save)
