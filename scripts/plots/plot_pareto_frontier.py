"""
plot_pareto_frontier.py

Pareto Frontier plots: Utility (F1) vs. Fairness metrics.

For each pair (F1, fairness_metric), a point is Pareto-optimal if no other
scenario simultaneously achieves higher F1 **and** better fairness.

Fairness direction conventions:
  - DI  : ideal = 1.0  → better = closer to 1  → score = -|DI  - 1|
  - SPD : ideal = 0.0  → better = closer to 0  → score = -|SPD|
  - EOD : ideal = 0.0  → better = closer to 0  → score = -|EOD|
  - AOD : ideal = 0.0  → better = closer to 0  → score = -|AOD|

Plots saved to  plots/{dataset}/paper/:
  plot_pareto_di.png
  plot_pareto_spd.png
  plot_pareto_eod.png
  plot_pareto_aod.png
  plot_pareto_dashboard.png   ← 2×2 panel with all four

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plots/plot_pareto_frontier.py
    PYTHONPATH=. .venv/bin/python scripts/plots/plot_pareto_frontier.py --dataset compas adult diabetes
"""
import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import seaborn as sns

# ── Palette (same as generate_paper_plots) ────────────────────────────────────
SG_COLORS = {
    'S1':             '#e63946',
    'S2_1 (Reweigh)': '#1d3557',
    'S2_2 (LFR)':     '#457b9d',
    'S2_3 (DIR)':     '#a8dadc',
    'S3 (1.5x)':      '#40916c',
    'S3 (2.0x)':      '#2a9d8f',
    'S3 (3.0x)':      '#52b788',
    'S4':             '#e9c46a',
    'S5':             '#f4a261',
    'S6':             '#6a4c93',
}
MODEL_MARKERS = {'LogisticRegression': 'o', 'SVM': 's', 'CatBoost': 'D'}

# Metrics: (column, x-axis label, ideal value, x-axis direction note)
METRICS = [
    ('disparate_impact',                'Disparate Impact (DI)',                1.0,  'ideal = 1'),
    ('statistical_parity_difference',   'Statistical Parity Difference (SPD)',  0.0,  'ideal = 0'),
    ('equal_opportunity_difference',    'Equal Opportunity Difference (EOD)',   0.0,  'ideal = 0'),
    ('average_absolute_odds_difference','Avg. Absolute Odds Difference (AOD)', 0.0,  'ideal = 0'),
]
METRIC_FNAMES = {
    'disparate_impact':                  'plot_pareto_di.png',
    'statistical_parity_difference':     'plot_pareto_spd.png',
    'equal_opportunity_difference':      'plot_pareto_eod.png',
    'average_absolute_odds_difference':  'plot_pareto_aod.png',
}


# ── Pareto helpers ────────────────────────────────────────────────────────────

def fairness_score(values: np.ndarray, ideal: float) -> np.ndarray:
    """Transform raw metric values to a 'higher = better' fairness score."""
    return -np.abs(values - ideal)


def pareto_mask(f1: np.ndarray, fscore: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of Pareto-optimal points.
    A point i is Pareto-optimal iff no other point j satisfies:
        f1[j] >= f1[i]  AND  fscore[j] >= fscore[i]
        with at least one strict.
    """
    n = len(f1)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (f1[j] >= f1[i] and fscore[j] >= fscore[i] and
                    (f1[j] > f1[i] or fscore[j] > fscore[i])):
                dominated[i] = True
                break
    return ~dominated


def pareto_frontier_line(f1: np.ndarray, metric: np.ndarray, ideal: float):
    """
    Return (x, y) arrays for drawing the Pareto frontier step-line.
    Sorted by metric (ascending toward ideal if ideal=0; or ascending raw if ideal=1).
    """
    fscore = fairness_score(metric, ideal)
    mask   = pareto_mask(f1, fscore)
    pf1    = f1[mask]
    pm     = metric[mask]

    if len(pf1) == 0:
        return np.array([]), np.array([])

    # Sort by metric x-axis for clean frontier line
    order = np.argsort(pm)
    return pm[order], pf1[order]


# ── Label / grouping ──────────────────────────────────────────────────────────

def build_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'sg' (scenario-group label) column used for colouring."""
    def _sg(row):
        s = row['scenario']
        m = row.get('mitigator', None)
        m = m if pd.notna(m) else 'None'
        if s == 'S1': return 'S1'
        if s == 'S2' and m == 'Reweighing': return 'S2_1 (Reweigh)'
        if s == 'S2' and m == 'LFR':        return 'S2_2 (LFR)'
        if s == 'S2' and m == 'DIRemover':  return 'S2_3 (DIR)'
        if s == 'S3_1.5': return 'S3 (1.5x)'
        if s == 'S3_2.0': return 'S3 (2.0x)'
        if s == 'S3_3.0': return 'S3 (3.0x)'
        if s == 'S4': return 'S4'
        if s == 'S5': return 'S5'
        if s == 'S6': return 'S6'
        return s
    df = df.copy()
    df['sg'] = df.apply(_sg, axis=1)
    return df


# ── Per-metric single plot ────────────────────────────────────────────────────

def plot_pareto_single(ax, means_full: pd.DataFrame,
                       metric_col: str, xlabel: str,
                       ideal: float, ideal_label: str,
                       dataset_name: str,
                       show_legend: bool = True):
    """Draw one Pareto-frontier scatter on *ax*."""

    f1_all     = means_full['f1'].values
    metric_all = means_full[metric_col].values
    fscore_all = fairness_score(metric_all, ideal)
    mask       = pareto_mask(f1_all, fscore_all)

    # ── 1. Reference lines / zones ─────────────────────────────────────────
    if metric_col == 'disparate_impact':
        ax.axvspan(0.8, 1.25, alpha=0.07, color='#2dc653', zorder=0)
        ax.axvline(1.0, color='#2dc653', lw=1.2, ls='--', alpha=0.7,
                   label='DI = 1 (fair)')
        ax.axvline(0.8, color='#e63946', lw=0.9, ls=':',  alpha=0.5)
        ax.axvline(1.25,color='#e63946', lw=0.9, ls=':',  alpha=0.5)
    else:
        ax.axvspan(-0.1, 0.1, alpha=0.07, color='#2dc653', zorder=0)
        ax.axvline(0.0, color='#2dc653', lw=1.2, ls='--', alpha=0.7,
                   label=f'{ideal_label}')

    # ── 2. All points (faded) ──────────────────────────────────────────────
    for sg, grp in means_full.groupby('sg'):
        for model, mgrp in grp.groupby('model'):
            ax.scatter(
                mgrp[metric_col], mgrp['f1'],
                color=SG_COLORS.get(sg, '#888888'),
                marker=MODEL_MARKERS.get(model, 'o'),
                s=55, alpha=0.30, edgecolors='none', zorder=2,
            )

    # ── 3. Pareto-optimal points (highlighted) ────────────────────────────
    pareto_pts = means_full[mask]
    for sg, grp in pareto_pts.groupby('sg'):
        for model, mgrp in grp.groupby('model'):
            ax.scatter(
                mgrp[metric_col], mgrp['f1'],
                color=SG_COLORS.get(sg, '#888888'),
                marker=MODEL_MARKERS.get(model, 'o'),
                s=130, alpha=1.0, edgecolors='white', linewidths=1.2,
                zorder=5,
                path_effects=[
                    pe.withStroke(linewidth=2.5,
                                  foreground=SG_COLORS.get(sg, '#888888'),
                                  alpha=0.35)
                ],
            )

    # ── 4. Pareto frontier step-line ──────────────────────────────────────
    px, py = pareto_frontier_line(f1_all, metric_all, ideal)
    if len(px) > 1:
        ax.plot(px, py,
                color='#222222', lw=1.8, ls='-', zorder=6, alpha=0.75,
                label='Pareto frontier',
                marker='', drawstyle='default')
        # Convex hull–like smooth line through pareto points
        ax.plot(px, py,
                color='#222222', lw=0, marker='', zorder=6,
                markersize=0)
        # Shaded region below frontier toward bad-fairness corner
        ax.fill_betweenx(py, ax.get_xlim()[0], px,
                         alpha=0.04, color='#222222', zorder=1)

    # ── 5. Labels for Pareto points ───────────────────────────────────────
    already = set()
    for _, row in pareto_pts.iterrows():
        label = row['sg'].replace(' ', '\n')
        if label in already:
            continue
        already.add(label)
        ax.annotate(
            label,
            xy=(row[metric_col], row['f1']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=7, color=SG_COLORS.get(row['sg'], '#333'),
            fontweight='bold', zorder=7,
        )

    # ── 6. Axes cosmetics ─────────────────────────────────────────────────
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel('F1-Score  →  Utility', fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)

    if show_legend:
        legend_sg  = [mpatches.Patch(color=c, label=sg)
                      for sg, c in SG_COLORS.items()
                      if sg in means_full['sg'].values]
        legend_mod = [plt.Line2D([0], [0], marker=m, color='gray',
                                 linestyle='None', markersize=7, label=mod)
                      for mod, m in MODEL_MARKERS.items()]
        legend_pf  = [plt.Line2D([0], [0], color='#222222', lw=1.8,
                                 label='Pareto frontier')]
        ax.legend(
            handles=legend_sg + legend_mod + legend_pf,
            fontsize=7, loc='lower right', ncol=2,
            framealpha=0.9,
        )


# ── Per-dataset runner ────────────────────────────────────────────────────────

def run_for_dataset(dataset_name: str) -> None:
    print(f'\n{"═"*65}')
    print(f'  plot_pareto_frontier.py  ·  {dataset_name.upper()}')
    print(f'{"═"*65}\n')

    results_path = f'outputs/{dataset_name}_results.csv'
    if not os.path.exists(results_path):
        print(f'  [SKIP] {results_path} not found.')
        return

    out = os.path.join('plots', dataset_name, 'paper')
    os.makedirs(out, exist_ok=True)

    df = pd.read_csv(results_path)
    df = build_groups(df)

    # Aggregate: mean per (scenario, mitigator, generator, model)
    means_s2 = (
        df[df['scenario'] == 'S2']
        .groupby(['scenario', 'mitigator', 'generator', 'model'])[
            ['f1', 'disparate_impact', 'statistical_parity_difference',
             'equal_opportunity_difference', 'average_absolute_odds_difference']
        ].mean().reset_index()
    )
    _s2_map = {'Reweighing': 'S2_1 (Reweigh)', 'LFR': 'S2_2 (LFR)',
               'DIRemover': 'S2_3 (DIR)'}
    means_s2['sg'] = means_s2['mitigator'].map(_s2_map).fillna('S2')

    means_other = (
        df[df['scenario'] != 'S2']
        .groupby(['scenario', 'generator', 'model'])[
            ['f1', 'disparate_impact', 'statistical_parity_difference',
             'equal_opportunity_difference', 'average_absolute_odds_difference']
        ].mean().reset_index()
    )
    means_other['mitigator'] = None

    def _sg_other(row):
        s = row['scenario']
        if s == 'S1': return 'S1'
        if s == 'S4': return 'S4'
        if s == 'S5': return 'S5'
        if s == 'S6': return 'S6'
        if s.startswith('S3'):
            a = s.split('_')[1] if '_' in s else '?'
            return f'S3 ({a}x)'
        return s

    means_other['sg'] = means_other.apply(_sg_other, axis=1)
    means_full = pd.concat([means_other, means_s2], ignore_index=True)

    # ── Individual plots ──────────────────────────────────────────────────
    for metric_col, xlabel, ideal, ideal_label in METRICS:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_pareto_single(ax, means_full, metric_col, xlabel,
                           ideal, ideal_label, dataset_name,
                           show_legend=True)
        fig.tight_layout()
        fname = METRIC_FNAMES[metric_col]
        fpath = os.path.join(out, fname)
        fig.savefig(fpath, dpi=200, bbox_inches='tight')
        fig.savefig(fpath.replace('.png', '.eps'), format='eps', bbox_inches='tight')
        plt.close(fig)
        print(f'  saved → {fpath} and .eps')

    # ── 2×2 Dashboard ─────────────────────────────────────────────────────
    print('  Generating 2×2 dashboard...')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    for ax, (metric_col, xlabel, ideal, ideal_label) in zip(axes.flat, METRICS):
        plot_pareto_single(ax, means_full, metric_col, xlabel,
                           ideal, ideal_label, dataset_name,
                           show_legend=False)

    # Shared legend below the grid
    legend_sg  = [mpatches.Patch(color=c, label=sg)
                  for sg, c in SG_COLORS.items()
                  if sg in means_full['sg'].values]
    legend_mod = [plt.Line2D([0], [0], marker=m, color='gray',
                             linestyle='None', markersize=8, label=mod)
                  for mod, m in MODEL_MARKERS.items()]
    legend_extra = [
        plt.Line2D([0], [0], color='#222222', lw=2, label='Pareto frontier'),
        mpatches.Patch(color='#222222', alpha=0.1, label='Non-dominated region'),
    ]
    fig.legend(
        handles=legend_sg + legend_mod + legend_extra,
        loc='lower center', ncol=7, fontsize=8.5,
        bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
    )

    fig.tight_layout()
    dash_path = os.path.join(out, 'plot_pareto_dashboard.png')
    fig.savefig(dash_path, dpi=200, bbox_inches='tight')
    fig.savefig(dash_path.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {dash_path} and .eps')

    print(f'\n  ✔ Done! All Pareto plots saved to {out}/\n')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_datasets = ['compas', 'adult', 'diabetes']
    parser = argparse.ArgumentParser(
        description='Pareto frontier plots for bias2fair-synth.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--dataset', nargs='+', default=['compas'],
        choices=all_datasets,
        metavar='DATASET',
        help=(
            'One or more datasets to plot.\n'
            f'Choices: {all_datasets}\n'
            'Example: --dataset compas adult diabetes'
        ),
    )
    args = parser.parse_args()
    datasets = list(dict.fromkeys(d.lower() for d in args.dataset))

    for i, ds in enumerate(datasets, 1):
        print(f'\n  [{i}/{len(datasets)}] {ds.upper()}')
        run_for_dataset(ds)

    print(f'\n  All done: {", ".join(d.upper() for d in datasets)}')


if __name__ == '__main__':
    main()
