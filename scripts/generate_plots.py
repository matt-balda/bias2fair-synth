"""
generate_plots.py
Generates all mandatory visualizations as per .agent-context.md §11 + §14:
  - Per-metric boxplots / violin plots (across seeds)
  - Grouped bar plots (scenario × generator)
  - Fairness vs F1 trade-off (Pareto frontier)
  - Real vs Synthetic distributions per feature
  - Correlation heatmaps (Real, Synthetic, Difference)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import wilcoxon

# ── Style ──────────────────────────────────────────────────────────────────
PALETTE = 'Set2'
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

PERF_METRICS    = ['f1', 'auc_roc', 'auc_pr', 'recall', 'precision', 'accuracy']
FAIRNESS_METRICS = ['disparate_impact', 'statistical_parity_difference',
                    'equal_opportunity_difference', 'average_absolute_odds_difference']
ALL_METRICS      = PERF_METRICS + FAIRNESS_METRICS

SCENARIO_ORDER  = ['S1', 'S2', 'S3', 'S4', 'S5']
GENERATOR_ORDER = ['Baseline', 'GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']

TARGET    = 'two_year_recid'
SENSITIVE = 'race'
DATASET   = 'compas'


# ── Helpers ────────────────────────────────────────────────────────────────
def makedirs(*parts):
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def save(fig, *parts, tight=True):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  METRICS PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_metric_boxplots(df, base):
    """Violin + box per metric, one PNG per metric."""
    out = makedirs(base, 'metrics', 'boxplots')
    for metric in tqdm(ALL_METRICS, desc='  Boxplots/Violin', leave=False):
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        order = [g for g in GENERATOR_ORDER if g in df['generator'].unique()]
        sns.violinplot(data=df, x='scenario', y=metric, hue='generator',
                       order=SCENARIO_ORDER, hue_order=order,
                       palette=PALETTE, inner='quart', ax=ax)
        if metric == 'disparate_impact':
            ax.axhline(0.8, color='red', linestyle='--', lw=1, label='80% Rule')
        ax.set_title(f'{metric.replace("_", " ").title()} — Distribution across Seeds')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        save(fig, out, f'{metric}.png')


def plot_grouped_bars(df, base):
    """Grouped bar (mean ± std) per metric."""
    out = makedirs(base, 'metrics', 'grouped_bars')
    summary = df.groupby(['scenario', 'generator'])[ALL_METRICS].agg(['mean', 'std'])
    for metric in tqdm(ALL_METRICS, desc='  Grouped bars', leave=False):
        if metric not in df.columns:
            continue
        means = summary[metric]['mean'].unstack('generator')
        stds  = summary[metric]['std'].unstack('generator')
        fig, ax = plt.subplots(figsize=(12, 5))
        means.loc[[s for s in SCENARIO_ORDER if s in means.index]].plot(
            kind='bar', yerr=stds, ax=ax, colormap=PALETTE, capsize=3, rot=0
        )
        if metric == 'disparate_impact':
            ax.axhline(0.8, color='red', linestyle='--', lw=1)
        ax.set_title(f'{metric.replace("_", " ").title()} (Mean ± Std)')
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        save(fig, out, f'{metric}.png')


def plot_tradeoff(df, base):
    """Fairness vs Performance Pareto plot."""
    out = makedirs(base, 'metrics')
    means = (df.groupby(['scenario', 'generator', 'model'])
               [['f1', 'disparate_impact']].mean().reset_index())
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=means, x='disparate_impact', y='f1',
                    hue='scenario', style='generator',
                    s=90, alpha=0.85, palette='viridis', ax=ax)
    ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='80% Rule')
    ax.set_title('Fairness–Utility Trade-off (Pareto View)')
    ax.set_xlabel('Fairness — Disparate Impact (↑ better)')
    ax.set_ylabel('Utility — F1-Score (↑ better)')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    save(fig, out, 'pareto_frontier.png')


def plot_wilcoxon_heatmap(df, base):
    """Wilcoxon p-values: S1 vs all for F1 and Disparate Impact."""
    out = makedirs(base, 'metrics')
    models  = df['model'].unique()
    comparisons = [('S1', 'S2'), ('S1', 'S3'), ('S1', 'S4'), ('S1', 'S5'), ('S2', 'S5')]
    results = []
    for s_a, s_b in comparisons:
        for model in models:
            gens_b = df[df['scenario'] == s_b]['generator'].unique()
            for gen_b in gens_b:
                a = df[(df['scenario'] == s_a) & (df['model'] == model)
                       & (df['generator'] == 'Baseline')].sort_values('seed')
                b = df[(df['scenario'] == s_b) & (df['model'] == model)
                       & (df['generator'] == gen_b)].sort_values('seed')
                shared = set(a['seed']) & set(b['seed'])
                a = a[a['seed'].isin(shared)].sort_values('seed')
                b = b[b['seed'].isin(shared)].sort_values('seed')
                if len(a) < 3:
                    continue
                try:
                    _, p_f1 = wilcoxon(a['f1'], b['f1'])
                    _, p_di = wilcoxon(a['disparate_impact'], b['disparate_impact'])
                    results.append({'comparison': f'{s_a}→{s_b}', 'generator': gen_b,
                                    'model': model, 'p_f1': p_f1, 'p_di': p_di})
                except Exception:
                    pass
    if results:
        pd.DataFrame(results).to_csv(os.path.join(base, 'metrics', 'wilcoxon.csv'), index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DISTRIBUTION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def plot_distributions(real_df, base):
    """For each synthetic_data CSV, compare distributions per feature."""
    synth_root = os.path.join('synthetic_data', DATASET)
    if not os.path.exists(synth_root):
        return

    num_cols = real_df.select_dtypes(include='number').columns.tolist()
    cat_cols = real_df.select_dtypes(exclude='number').columns.tolist()

    for scenario in tqdm(os.listdir(synth_root), desc='  Distributions', leave=False):
        sc_path = os.path.join(synth_root, scenario)
        for generator in os.listdir(sc_path):
            gen_path = os.path.join(sc_path, generator)
            csvs = [f for f in os.listdir(gen_path) if f.endswith('.csv')]
            if not csvs:
                continue

            # Concatenate all seeds
            synth_dfs = [pd.read_csv(os.path.join(gen_path, f)) for f in csvs]
            synth = pd.concat(synth_dfs, ignore_index=True)
            out   = makedirs(base, 'distributions', DATASET, scenario, generator)

            # Numerical — KDE
            for col in num_cols:
                if col not in synth.columns:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.kdeplot(real_df[col].dropna(), ax=ax, label='Real',
                            fill=True, alpha=0.35, color='steelblue')
                sns.kdeplot(synth[col].dropna(), ax=ax, label='Synthetic',
                            fill=True, alpha=0.35, color='tomato', linestyle='--')
                ax.set_title(f'{col} — Real vs Synthetic ({generator})')
                ax.legend()
                save(fig, out, f'{col}_kde.png')

            # Categorical — bar
            for col in cat_cols:
                if col not in synth.columns:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                real_df[col].value_counts(normalize=True).plot(
                    kind='bar', ax=axes[0], color='steelblue', rot=30)
                axes[0].set_title(f'Real — {col}')
                synth[col].value_counts(normalize=True).plot(
                    kind='bar', ax=axes[1], color='tomato', rot=30)
                axes[1].set_title(f'Synthetic — {col}')
                save(fig, out, f'{col}_bar.png')


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlations(real_df, base):
    """Heatmaps: real corr, synth corr, |difference|."""
    synth_root = os.path.join('synthetic_data', DATASET)
    if not os.path.exists(synth_root):
        return

    real_corr = real_df.corr(numeric_only=True)

    for scenario in tqdm(os.listdir(synth_root), desc='  Correlations', leave=False):
        sc_path = os.path.join(synth_root, scenario)
        for generator in os.listdir(sc_path):
            gen_path = os.path.join(sc_path, generator)
            csvs = [f for f in os.listdir(gen_path) if f.endswith('.csv')]
            if not csvs:
                continue

            synth_dfs  = [pd.read_csv(os.path.join(gen_path, f)) for f in csvs]
            synth      = pd.concat(synth_dfs, ignore_index=True)
            synth_corr = synth.corr(numeric_only=True)
            diff_corr  = (real_corr - synth_corr).abs()

            out = makedirs(base, 'correlation', DATASET, scenario, generator)

            for title, corr, fname in [
                ('Real Correlation',      real_corr,  'real.png'),
                ('Synthetic Correlation', synth_corr, 'synthetic.png'),
                ('|Real − Synthetic|',    diff_corr,  'difference.png'),
            ]:
                fig, ax = plt.subplots(figsize=(8, 7))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                            center=0, ax=ax, annot_kws={'size': 6})
                ax.set_title(f'{title} ({generator} / {scenario})')
                save(fig, out, fname)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    from utils.data_loader import load_compas

    csv_path = 'outputs/compas_results.csv'
    if not os.path.exists(csv_path):
        print('✘ No results file found. Run scripts/run_experiment.py first.')
        return

    df   = pd.read_csv(csv_path)
    base = 'plots'

    print('\n' + '═' * 60)
    print('  bias2fair-synth  ·  Visualization Generator v2')
    print('═' * 60 + '\n')

    real_data = load_compas()

    steps = [
        ('Boxplot/Violin plots',       lambda: plot_metric_boxplots(df, base)),
        ('Grouped bar plots',          lambda: plot_grouped_bars(df, base)),
        ('Pareto trade-off plot',      lambda: plot_tradeoff(df, base)),
        ('Wilcoxon significance',      lambda: plot_wilcoxon_heatmap(df, base)),
        ('Real vs Synthetic dists',    lambda: plot_distributions(real_data, base)),
        ('Correlation heatmaps',       lambda: plot_correlations(real_data, base)),
    ]

    for label, fn in tqdm(steps, desc='Generating plots', unit='step',
                          bar_format='  {l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}',
                          colour='cyan'):
        tqdm.write(f'  → {label}...')
        fn()

    print('\n  ✔ All plots saved to plots/')
    print('═' * 60 + '\n')


if __name__ == '__main__':
    main()
