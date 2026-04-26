"""
generate_plots.py:

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
# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

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
# 2.  DISTRIBUTION COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
def plot_distributions(real_df, base):
    """For each synthetic_data CSV, compare distributions per feature."""
    synth_root = os.path.join('synthetic_data', DATASET)
    if not os.path.exists(synth_root):
        return

    num_cols = real_df.select_dtypes(include='number').columns.tolist()
    cat_cols = real_df.select_dtypes(exclude='number').columns.tolist()

    for scenario in tqdm(sorted(os.listdir(synth_root)), desc='  Distributions', leave=False):
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

    for scenario in tqdm(sorted(os.listdir(synth_root)), desc='  Correlations', leave=False):
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

    base = 'plots'

    print('\n' + '═' * 60)
    print('  bias2fair-synth  ·  Distribuições e Correlações dos Dados Sintéticos')
    print('═' * 60 + '\n')

    real_data = load_compas()

    steps = [
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
