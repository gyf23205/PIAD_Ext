"""Plot mean +/- std of one metric for all label-ratio configs in a single figure.

Like plot_results.py, but instead of picking one config, every config is drawn
in each subplot: mean values appear as dots with a different marker shape (and
color) per config, slightly offset horizontally per method, with +/- std error
bars.

Usage:
    python src/visualization/plot_results_configs.py --metric bin_f1
    python src/visualization/plot_results_configs.py --metric F_aff --methods PCAD RoSAS SimAD
"""
import argparse
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from plot_results import METRICS, N_CONFIGS, XLSX_PATH, load_results

MARKERS = ['o', 's', '^', 'D', 'v']
CONFIG_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']


def plot_metric_all_configs(results, datasets, metric, methods, config_labels, save_dir,
                            show=True, save=True):
    # Fail loudly on any missing value for the selected methods/metric.
    for dataset in datasets:
        for cfg in range(N_CONFIGS):
            for method in methods:
                mean, std = results[dataset][cfg].get(method, {}).get(metric, (float('nan'), float('nan')))
                if math.isnan(mean) or math.isnan(std):
                    raise ValueError(
                        f'Missing value: dataset={dataset}, method={method}, '
                        f'metric={metric}, config={cfg} ({config_labels[cfg]})'
                    )

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    x = np.arange(len(methods))
    offsets = np.linspace(-0.25, 0.25, N_CONFIGS)
    for ax, dataset in zip(axes, datasets):
        for cfg in range(N_CONFIGS):
            means = [results[dataset][cfg][m][metric][0] for m in methods]
            stds = [results[dataset][cfg][m][metric][1] for m in methods]
            ax.errorbar(x + offsets[cfg], means, yerr=stds,
                        fmt=MARKERS[cfg % len(MARKERS)],
                        color=CONFIG_COLORS[cfg % len(CONFIG_COLORS)],
                        markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                        linestyle='none', capsize=3, elinewidth=1.2,
                        label=config_labels[cfg])
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_title(dataset, fontsize=13)
        ax.grid(axis='y', color='0.85', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(metric)
    fig.suptitle(metric, fontweight='bold', y=1.06)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=N_CONFIGS, frameon=False)
    fig.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        safe_metric = re.sub(r'[^\w.-]+', '_', metric).strip('_')
        out_path = os.path.join(save_dir, f'{safe_metric}_allcfg.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved figure to {out_path}')
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--metric', required=True, help=f'Metric to plot, one of: {METRICS}')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to include (default: all methods found in the file)')
    parser.add_argument('--xlsx', default=XLSX_PATH, help='Path to results.xlsx')
    parser.add_argument('--save-dir', default='./figures', help='Directory to save the figure')
    parser.add_argument('--no-show', action='store_true', help='Only save the figure, do not open a window')
    parser.add_argument('--no-save', action='store_true', help='Only show the figure, do not save it to disk')
    args = parser.parse_args()

    if args.metric not in METRICS:
        parser.error(f'Unknown metric "{args.metric}". Valid metrics: {METRICS}')

    results, datasets, all_methods, config_labels = load_results(args.xlsx)

    methods = args.methods if args.methods else all_methods
    unknown = [m for m in methods if m not in all_methods]
    if unknown:
        parser.error(f'Unknown method(s) {unknown}. Valid methods: {all_methods}')

    plot_metric_all_configs(results, datasets, args.metric, methods, config_labels,
                            args.save_dir, show=not args.no_show, save=not args.no_save)
