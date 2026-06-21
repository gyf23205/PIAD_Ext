"""Plot mean +/- std bar charts of benchmark results from results.xlsx.

The spreadsheet contains one block per dataset (stacked vertically), each with
three label-ratio configs side by side. Within a config block the columns are
[label, seed values..., Mean, Std]. Methods are stacked under each dataset
header, each followed by one row per metric.

Usage:
    python src/visualization/plot_results.py --metric bin_f1 --config 1
    python src/visualization/plot_results.py --metric F_aff --methods PCAD RoSAS SimAD
"""
import argparse
import os
import re

import math
import matplotlib.pyplot as plt
import pandas as pd

# Larger fonts everywhere. Imported by plot_results_configs.py too, so this
# applies to both scripts.
plt.rcParams.update({
    'font.size': 17,
    'axes.titlesize': 19,
    'axes.labelsize': 19,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

XLSX_PATH = r'C:\Users\63218\OneDrive - purdue.edu\Documents\purdue research\PIAD_Ext\results.xlsx'

N_CONFIGS = 3          # number of label-ratio scenarios shown per figure
BLOCK_COL_STRIDE = 8   # config blocks start at columns 0, 8, 16, 24, ...
MEAN_COL_OFFSET = 4    # relative to block start
STD_COL_OFFSET = 5

# Which spreadsheet column block backs each plotted scenario, per setting.
# Detection uses the first three blocks (the first is rp:0.01 rko:0.00 rkn:0.01).
# Classification swaps that first scenario for the extra rp:0.01 rko:0.01
# rkn:0.01 block appended after the detection blocks (block index 3).
CONFIG_BLOCKS = {
    'detection': [0, 1, 2],
    'classification': [3, 1, 2],
}

METRICS = [
    'test_auc', 'f1_macro', 'f1_weighted', 'mh_acc', 'mh_recall',
    'bin_f1', 'bin_acc', 'bin_recall', 'mcc',
    'P_aff (UAff)', 'R_aff (NAff)', 'F_aff',
]

# Okabe-Ito colorblind-safe palette (+ a distinct brown), cycled if there are
# more methods than hues.
PALETTE = [
    '#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7',
    '#56B4E9', '#F0E442', '#999999', '#000000', '#8C564B',
]


def _cell(df, row, col):
    if row >= df.shape[0] or col >= df.shape[1]:
        return None
    v = df.iat[row, col]
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def load_results(xlsx_path):
    """Parse the spreadsheet into results[dataset][cfg][method][metric] = (mean, std).

    Also returns ordered dataset/method lists and per-config labels.
    """
    df = pd.read_excel(xlsx_path, sheet_name=0, header=None)

    # Number of column blocks present in the sheet (each block is one scenario).
    # Detection sheets have 3; classification adds an extra rp:0.01 rko:0.01
    # rkn:0.01 block, giving 4. Detect it from the sheet width so both work.
    n_blocks = max(1, (df.shape[1] - STD_COL_OFFSET - 1) // BLOCK_COL_STRIDE + 1)

    results = {}
    datasets, methods = [], []
    config_labels = [None] * n_blocks

    row = 0
    while row < df.shape[0]:
        name = _cell(df, row, 0)
        # A dataset header is a non-empty cell followed by a 'Seed' row.
        if name is None or _cell(df, row + 1, 0) != 'Seed':
            row += 1
            continue

        dataset = str(name)
        datasets.append(dataset)
        results[dataset] = {cfg: {} for cfg in range(n_blocks)}
        for cfg in range(n_blocks):
            base = cfg * BLOCK_COL_STRIDE
            parts = [str(_cell(df, row, base + c)) for c in (1, 2, 3) if _cell(df, row, base + c) is not None]
            config_labels[cfg] = ' '.join(parts)

        # Walk method/metric rows until a blank row ends the dataset block.
        row += 2
        method = None
        while row < df.shape[0]:
            label = _cell(df, row, 0)
            if label is None:
                break
            label = str(label)
            if label in METRICS:
                if method is None:
                    raise ValueError(f'Metric row "{label}" before any method name (row {row + 1})')
                for cfg in range(n_blocks):
                    base = cfg * BLOCK_COL_STRIDE
                    mean = _cell(df, row, base + MEAN_COL_OFFSET)
                    std = _cell(df, row, base + STD_COL_OFFSET)
                    results[dataset][cfg][method][label] = (
                        float(mean) if mean is not None else float('nan'),
                        float(std) if std is not None else float('nan'),
                    )
            else:
                method = label
                for cfg in range(n_blocks):
                    results[dataset][cfg].setdefault(method, {})
                if method not in methods:
                    methods.append(method)
            row += 1

    if not datasets:
        raise ValueError(f'No dataset blocks found in {xlsx_path}')
    return results, datasets, methods, config_labels


def plot_metric(results, datasets, metric, methods, all_methods, cfg, config_label, save_dir,
                show=True, save=True):
    # Fail loudly on any missing value for the selected methods/metric.
    for dataset in datasets:
        for method in methods:
            mean, std = results[dataset][cfg].get(method, {}).get(metric, (float('nan'), float('nan')))
            if math.isnan(mean) or math.isnan(std):
                raise ValueError(
                    f'Missing value: dataset={dataset}, method={method}, '
                    f'metric={metric}, config={cfg} ({config_label})'
                )

    # Colors are assigned from the full method list so each method keeps the
    # same color regardless of which subset is plotted.
    color_map = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(all_methods)}

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 2.8))
    if n == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        means = [results[dataset][cfg][m][metric][0] for m in methods]
        stds = [results[dataset][cfg][m][metric][1] for m in methods]
        ax.bar(methods, means, yerr=stds, width=0.7,
               color=[color_map[m] for m in methods],
               edgecolor='black', linewidth=0.6,
               capsize=3, error_kw={'ecolor': '0.25', 'elinewidth': 1.2})
        ax.set_title(dataset)
        ax.grid(axis='y', color='0.85', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_ha('right')
    axes[0].set_ylabel(metric)
    fig.suptitle(f'{metric}  ({config_label})', fontweight='bold')
    fig.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        safe_metric = re.sub(r'[^\w.-]+', '_', metric).strip('_')
        out_path = os.path.join(save_dir, f'{safe_metric}_cfg{cfg}.pdf')
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved figure to {out_path}')
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", default="detection", choices=["detection", "classification"],
                        help="Choose setting (selects which label-ratio scenarios are used).")
    parser.add_argument('--metric', required=True, help=f'Metric to plot, one of: {METRICS}')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to include (default: all methods found in the file)')
    parser.add_argument('--config', type=int, default=0, choices=range(N_CONFIGS),
                        help='Label-ratio scenario to plot (0=first scenario for the setting)')
    parser.add_argument('--xlsx', default=XLSX_PATH, help='Path to results.xlsx')
    parser.add_argument('--save-dir', default='./figures', help='Directory to save the figure')
    parser.add_argument('--no-show', action='store_true', help='Only save the figure, do not open a window')
    parser.add_argument('--no-save', action='store_true', help='Only show the figure, do not save it to disk')
    args = parser.parse_args()

    if args.metric not in METRICS:
        parser.error(f'Unknown metric "{args.metric}". Valid metrics: {METRICS}')

    results, datasets, all_methods, config_labels = load_results(args.xlsx)

    block = CONFIG_BLOCKS[args.setting][args.config]
    if block >= len(config_labels) or config_labels[block] is None:
        parser.error(
            f'Scenario {args.config} for setting "{args.setting}" needs column block '
            f'{block}, but the spreadsheet only has {len(config_labels)} block(s). '
            f'Did you add the rp:0.01 rko:0.01 rkn:0.01 columns?'
        )

    methods = args.methods if args.methods else all_methods
    unknown = [m for m in methods if m not in all_methods]
    if unknown:
        parser.error(f'Unknown method(s) {unknown}. Valid methods: {all_methods}')

    plot_metric(results, datasets, args.metric, methods, all_methods, block,
                config_labels[block], args.save_dir, show=not args.no_show,
                save=not args.no_save)
