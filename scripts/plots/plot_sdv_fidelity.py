"""
plot_sdv_fidelity.py

Generates two statistical fidelity plots comparing real vs synthetic data.

Logic:
  - Reference scenario: S3_2.0 (fixed, mid-point augmentation ratio)
  - For each generator × seed: sample n_real rows from synthetic, compute JSD per feature
  - Aggregate JSD as mean across seeds → robust per-generator fidelity estimate
  - Best generator = lowest mean JSD averaged across all features
  - KDE overlay uses seed=42 of the best generator (canonical representative run)

Plots saved to plots/paper/:
  - plot_sdv_heatmap_jsd.png  : mean JSD heatmap  (Generator × Feature, averaged over seeds)
  - plot_sdv_kde_overlay.png  : histogram overlay  (best generator, seed=42 vs real)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plots/plot_sdv_fidelity.py
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import jensenshannon

from utils.data_loader import load_dataset, DATASET_CONFIGS

# ── CLI args ─────────────────────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser(description='SDV fidelity plots.')
parser.add_argument('--dataset', type=str, default='compas',
                    choices=['compas', 'adult', 'diabetes'],
                    help='Dataset to analyse (default: compas)')
parser.add_argument('--scenario', type=str, default='S3_2.0',
                    help='Reference scenario for fidelity (default: S3_2.0)')
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
SYNTH_ROOT         = os.path.join('synthetic_data', args.dataset)
OUT_DIR            = os.path.join('plots', args.dataset, 'paper')
REFERENCE_SCENARIO = args.scenario
GENERATORS         = ['CTGAN', 'GaussianCopula', 'TVAE', 'TabDDPM']
CANONICAL_SEED     = 42   # seed used for the KDE overlay visual

# Columns present in both real (load_compas) and synthetic CSVs
COL_LABELS = {
    'age':                      'Age',
    'priors_count':             'Prior Crimes',
    'juv_fel_count':            'Juv. Felonies',
    'juv_misd_count':           'Juv. Misdemeanors',
    'juv_other_count':          'Juv. Other',
    'two_year_recid':           'Recidivism (Target)',
    'sex':                      'Sex',
    'c_charge_degree':          'Charge Degree',
    'race':                     'Race',
    'age_cat_Greater than 45':  'Age > 45',
    'age_cat_Less than 25':     'Age < 25',
}

# Features shown in the KDE overlay (most informative)
KDE_FEATURES = ['age', 'priors_count', 'race', 'sex', 'two_year_recid', 'juv_fel_count']

os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def jsd_score(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    """Jensen-Shannon Divergence² (histogram-based) in [0, 1]."""
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(a, bins=edges, density=True)[0].astype(float) + 1e-10
    q = np.histogram(b, bins=edges, density=True)[0].astype(float) + 1e-10
    p /= p.sum(); q /= q.sum()
    return float(jensenshannon(p, q) ** 2)


def load_synth_seed(generator: str, seed_file: str) -> pd.DataFrame:
    """Load one synthetic seed CSV from the reference scenario."""
    path = os.path.join(SYNTH_ROOT, REFERENCE_SCENARIO, generator, seed_file)
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def list_seed_files(generator: str) -> list[str]:
    """Return sorted list of seed CSV filenames for a generator."""
    gen_path = os.path.join(SYNTH_ROOT, REFERENCE_SCENARIO, generator)
    if not os.path.isdir(gen_path):
        return []
    return sorted(f for f in os.listdir(gen_path) if f.endswith('.csv'))


# ── JSD computation across seeds ─────────────────────────────────────────────
def compute_jsd_per_seed(real_df: pd.DataFrame, generator: str) -> pd.DataFrame:
    """
    For each seed of `generator`, sample n_real rows from the synthetic data
    and compute JSD per feature.

    Returns a DataFrame with one row per seed and one column per feature.
    """
    features  = [c for c in COL_LABELS if c in real_df.columns]
    n_real    = len(real_df)
    seed_files = list_seed_files(generator)

    rows = []
    for fname in seed_files:
        synth = load_synth_seed(generator, fname)
        if synth.empty:
            continue
        # Down-sample to n_real for a fair like-for-like comparison
        if len(synth) > n_real:
            synth = synth.sample(n=n_real, random_state=CANONICAL_SEED)

        row = {'seed_file': fname}
        for feat in features:
            if feat in synth.columns:
                row[feat] = jsd_score(real_df[feat].values, synth[feat].values)
            else:
                row[feat] = np.nan
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('seed_file')


# ── Best generator selection ──────────────────────────────────────────────────
def select_best_generator(mean_jsd: pd.DataFrame) -> str:
    """
    Rank generators by mean JSD averaged across all features.
    Lower is better (more similar to real data).
    """
    ranking = mean_jsd.mean(axis=1).sort_values()
    print('\n  Generator ranking by mean JSD across features (↓ better):')
    for gen, score in ranking.items():
        marker = '  ← best' if gen == ranking.index[0] else ''
        print(f'    {gen:<20} {score:.4f}{marker}')
    best = ranking.index[0]
    print(f'\n  Selected: {best}\n')
    return best


# ── Plot 1: JSD Heatmap (mean across seeds) ───────────────────────────────────
def plot_heatmap_jsd(real_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each generator, compute mean JSD across all seeds (S3_2.0).
    Plot heatmap and return mean JSD DataFrame (index=generator, cols=features).
    """
    print(f'Computing JSD per generator × seed ({REFERENCE_SCENARIO})...')
    features = [c for c in COL_LABELS if c in real_df.columns]

    mean_rows = []
    for gen in GENERATORS:
        seed_jsd = compute_jsd_per_seed(real_df, gen)
        if seed_jsd.empty:
            print(f'  [WARN] No data for {gen}')
            continue
        n_seeds = len(seed_jsd)
        mean_row = seed_jsd[features].mean()
        mean_row.name = gen
        mean_rows.append(mean_row)
        print(f'  {gen:<20} computed over {n_seeds} seeds')

    if not mean_rows:
        print('  [SKIP] No data available.')
        return pd.DataFrame()

    mean_jsd = pd.DataFrame(mean_rows)[features]  # (generator × feature)

    # Pretty names for display
    jsd_plot = mean_jsd.copy()
    jsd_plot.columns = [COL_LABELS[c] for c in features]

    n_cols = len(jsd_plot.columns)
    fig, ax = plt.subplots(figsize=(max(11, n_cols * 1.1 + 2), len(jsd_plot) * 1.3 + 2))
    sns.heatmap(
        jsd_plot, annot=True, fmt='.3f',
        cmap='YlOrRd', vmin=0, vmax=0.35,
        linewidths=1.5, linecolor='white',
        annot_kws={'size': 11, 'weight': 'bold'},
        cbar_kws={'label': 'JSD médio entre seeds (↓ = mais similar ao real)'},
        ax=ax,
    )
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Gerador Sintético', fontsize=12)
    ax.tick_params(axis='x', rotation=35, labelsize=10)
    ax.tick_params(axis='y', rotation=0,  labelsize=10)
    ax.set_title(
        f'Similaridade Estatística: Dados Reais × Dados Sintéticos (COMPAS) — {REFERENCE_SCENARIO}\n'
        f'Jensen-Shannon Divergence médio por Gerador × Feature  '
        f'(média de {len(mean_rows[0].index) if mean_rows else "?"} seeds por gerador)',
        fontsize=12, fontweight='bold', pad=14,
    )
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'plot_sdv_heatmap_jsd.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out}')

    return mean_jsd


# ── Plot 2: KDE / Histogram Overlay (best generator, seed=42) ────────────────
def plot_kde_overlay(real_df: pd.DataFrame, gen: str):
    """
    Compare real vs synthetic distributions for the best generator.
    Uses seed=42 as the canonical representative run.
    """
    seed_fname = f'seed_{CANONICAL_SEED}.csv'
    print(f'Generating KDE overlay ({gen}, {seed_fname})...')

    synth = load_synth_seed(gen, seed_fname)
    if synth.empty:
        print(f'  [SKIP] {seed_fname} not found for {gen}.')
        return

    n_real = len(real_df)
    if len(synth) > n_real:
        synth = synth.sample(n=n_real, random_state=CANONICAL_SEED)

    features = [f for f in KDE_FEATURES if f in real_df.columns and f in synth.columns]
    n_rows, n_cols = 2, 3

    jsd_vals = {
        feat: jsd_score(real_df[feat].dropna().values, synth[feat].dropna().values)
        for feat in features
    }

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.35)

    for idx, feat in enumerate(features):
        ax    = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        label = COL_LABELS.get(feat, feat)
        jsd   = jsd_vals[feat]

        r_vals = real_df[feat].dropna().values
        s_vals = synth[feat].dropna().values

        # Binary / low-cardinality → aligned histogram; continuous → free bins
        unique_r = np.unique(r_vals)
        if len(unique_r) <= 12:
            lo   = min(r_vals.min(), s_vals.min())
            hi   = max(r_vals.max(), s_vals.max())
            bins = np.linspace(lo - 0.5, hi + 0.5, len(unique_r) + 2)
        else:
            bins = 40

        ax.hist(r_vals, bins=bins, density=True, alpha=0.65,
                color='#4a5568', label='Real')
        ax.hist(s_vals, bins=bins, density=True, alpha=0.50,
                color='#63b3ed', label=f'Sintético ({gen})')

        ax.set_title(f'{label}  [JSD={jsd:.3f}]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Densidade', fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    mean_jsd = np.nanmean(list(jsd_vals.values()))
    fig.suptitle(
        f'Sobreposição de Distribuições: Real vs Sintético — {REFERENCE_SCENARIO}\n'
        f'Melhor gerador: {gen}  |  seed={CANONICAL_SEED}  |  JSD médio = {mean_jsd:.3f}  '
        f'(n_real = n_synth = {n_real})',
        fontsize=12, fontweight='bold', y=1.02,
    )
    out = os.path.join(OUT_DIR, 'plot_sdv_kde_overlay.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '═' * 65)
    print(f'  plot_sdv_fidelity.py  ·  Fidelity Analysis  [{REFERENCE_SCENARIO}]')
    print(f'  Dataset: {args.dataset.upper()}')
    print('═' * 65 + '\n')

    cfg = DATASET_CONFIGS[args.dataset]
    real_df = cfg['loader']()
    print(f'  Real data  : {len(real_df)} rows × {real_df.shape[1]} cols')
    print(f'  Scenario   : {REFERENCE_SCENARIO}')
    print(f'  Generators : {", ".join(GENERATORS)}')
    print(f'  Sampling   : n_synth → n_real = {len(real_df)} (per seed)\n')

    mean_jsd = plot_heatmap_jsd(real_df)
    best_gen = select_best_generator(mean_jsd)
    plot_kde_overlay(real_df, gen=best_gen)

    print('\n  ✔ Done! Plots saved to', OUT_DIR)
    print('═' * 65 + '\n')


if __name__ == '__main__':
    main()
