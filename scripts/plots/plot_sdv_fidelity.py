"""
plot_sdv_fidelity.py

Generates two statistical fidelity plots comparing real vs synthetic data.

Logic:
  - Reference scenario: S3_2.0 (default, overridable with --scenario)
  - For each generator × seed: sample n_real rows from synthetic, compute JSD per feature
  - Aggregate JSD as mean across seeds → robust per-generator fidelity estimate
  - Best generator = lowest mean JSD averaged across all features
  - KDE overlay uses seed=42 of the best generator (canonical representative run)

Plots saved to plots/{dataset}/paper/:
  - plot_sdv_heatmap_jsd.png  : mean JSD heatmap  (Generator × Feature, averaged over seeds)
  - plot_sdv_kde_overlay.png  : histogram overlay  (best generator, seed=42 vs real)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plots/plot_sdv_fidelity.py
    PYTHONPATH=. .venv/bin/python scripts/plots/plot_sdv_fidelity.py --dataset adult diabetes
    PYTHONPATH=. .venv/bin/python scripts/plots/plot_sdv_fidelity.py --dataset compas --scenario S3_1.5
"""
import warnings
warnings.filterwarnings('ignore')

import argparse
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

from utils.data_loader import DATASET_CONFIGS

GENERATORS    = ['CTGAN', 'GaussianCopula', 'TVAE', 'TabDDPM']
CANONICAL_SEED = 42


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


def load_synth_seed(synth_root: str, scenario: str, generator: str, seed_file: str) -> pd.DataFrame:
    """Load one synthetic seed CSV from the reference scenario."""
    path = os.path.join(synth_root, scenario, generator, seed_file)
    if not os.path.isfile(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def list_seed_files(synth_root: str, scenario: str, generator: str) -> list:
    """Return sorted list of seed CSV filenames for a generator."""
    gen_path = os.path.join(synth_root, scenario, generator)
    if not os.path.isdir(gen_path):
        return []
    return sorted(f for f in os.listdir(gen_path) if f.endswith('.csv'))


# ── JSD computation across seeds ─────────────────────────────────────────────
def compute_jsd_per_seed(real_df: pd.DataFrame, generator: str,
                         col_labels: dict, synth_root: str, scenario: str) -> pd.DataFrame:
    features   = [c for c in col_labels if c in real_df.columns]
    n_real     = len(real_df)
    seed_files = list_seed_files(synth_root, scenario, generator)

    rows = []
    for fname in seed_files:
        synth = load_synth_seed(synth_root, scenario, generator, fname)
        if synth.empty:
            continue
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


def select_best_generator(mean_jsd: pd.DataFrame) -> str:
    ranking = mean_jsd.mean(axis=1).sort_values()
    print('\n  Generator ranking by mean JSD across features (↓ better):')
    for gen, score in ranking.items():
        marker = '  ← best' if gen == ranking.index[0] else ''
        print(f'    {gen:<20} {score:.4f}{marker}')
    best = ranking.index[0]
    print(f'\n  Selected: {best}\n')
    return best


# ── Plot 1: JSD Heatmap ───────────────────────────────────────────────────────
def plot_heatmap_jsd(real_df: pd.DataFrame, col_labels: dict,
                     out_dir: str, synth_root: str, scenario: str,
                     dataset_name: str) -> pd.DataFrame:
    print(f'Computing JSD per generator × seed ({scenario})...')
    features = [c for c in col_labels if c in real_df.columns]

    mean_rows = []
    for gen in GENERATORS:
        seed_jsd = compute_jsd_per_seed(real_df, gen, col_labels, synth_root, scenario)
        if seed_jsd.empty:
            print(f'  [WARN] No data for {gen}')
            continue
        n_seeds  = len(seed_jsd)
        mean_row = seed_jsd[features].mean()
        mean_row.name = gen
        mean_rows.append(mean_row)
        print(f'  {gen:<20} computed over {n_seeds} seeds')

    if not mean_rows:
        print('  [SKIP] No data available.')
        return pd.DataFrame()

    mean_jsd = pd.DataFrame(mean_rows)[features]
    jsd_plot = mean_jsd.copy()
    jsd_plot.columns = [col_labels[c] for c in features]

    n_cols = len(jsd_plot.columns)
    fig, ax = plt.subplots(figsize=(max(11, n_cols * 1.1 + 2), len(jsd_plot) * 1.3 + 2))
    sns.heatmap(
        jsd_plot, annot=True, fmt='.3f',
        cmap='YlOrRd', vmin=0, vmax=0.35,
        linewidths=1.5, linecolor='white',
        annot_kws={'size': 11, 'weight': 'bold'},
        cbar_kws={'label': 'Mean JSD across seeds (↓ = closer to real)'},
        ax=ax,
    )
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Synthetic Generator', fontsize=12)
    ax.tick_params(axis='x', rotation=35, labelsize=10)
    ax.tick_params(axis='y', rotation=0,  labelsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, 'plot_sdv_heatmap_jsd.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out} and .eps')
    return mean_jsd


# ── Plot 2: KDE Overlay ───────────────────────────────────────────────────────
def plot_kde_overlay(real_df: pd.DataFrame, gen: str, col_labels: dict,
                     kde_features: list, out_dir: str, synth_root: str, scenario: str):
    seed_fname = f'seed_{CANONICAL_SEED}.csv'
    print(f'Generating KDE overlay ({gen}, {seed_fname})...')

    synth = load_synth_seed(synth_root, scenario, gen, seed_fname)
    if synth.empty:
        print(f'  [SKIP] {seed_fname} not found for {gen}.')
        return

    n_real = len(real_df)
    if len(synth) > n_real:
        synth = synth.sample(n=n_real, random_state=CANONICAL_SEED)

    features = [f for f in kde_features if f in real_df.columns and f in synth.columns]
    n_rows, n_cols = 2, 3

    jsd_vals = {
        feat: jsd_score(real_df[feat].dropna().values, synth[feat].dropna().values)
        for feat in features
    }

    fig = plt.figure(figsize=(15, 8))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.35)

    for idx, feat in enumerate(features):
        ax    = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        label = col_labels.get(feat, feat)
        jsd   = jsd_vals[feat]

        r_vals = real_df[feat].dropna().values
        s_vals = synth[feat].dropna().values

        unique_r = np.unique(r_vals)
        if len(unique_r) <= 12:
            lo   = min(r_vals.min(), s_vals.min())
            hi   = max(r_vals.max(), s_vals.max())
            bins = np.linspace(lo - 0.5, hi + 0.5, len(unique_r) + 2)
        else:
            bins = 40

        ax.hist(r_vals, bins=bins, density=True, alpha=0.65, color='#4a5568', label='Real')
        ax.hist(s_vals, bins=bins, density=True, alpha=0.50, color='#63b3ed',
                label=f'Synthetic ({gen})')
        ax.set_title(f'{label}  [JSD={jsd:.3f}]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    out = os.path.join(out_dir, 'plot_sdv_kde_overlay.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out} and .eps')


# ── Per-dataset runner ────────────────────────────────────────────────────────
def run_for_dataset(dataset_name: str, scenario: str) -> None:
    """Run fidelity analysis for a single dataset."""
    synth_root = os.path.join('synthetic_data', dataset_name)
    out_dir    = os.path.join('plots', dataset_name, 'paper')
    os.makedirs(out_dir, exist_ok=True)

    print('\n' + '═' * 65)
    print(f'  plot_sdv_fidelity.py  ·  Fidelity Analysis  [{scenario}]')
    print(f'  Dataset: {dataset_name.upper()}')
    print('═' * 65 + '\n')

    cfg       = DATASET_CONFIGS[dataset_name]
    real_df   = cfg['loader']()
    target    = cfg['target']
    sensitive = cfg['sensitive']

    # Apply the same 10k stratified cut used during the experiment,
    # so JSD is computed against the same distribution the generators trained on.
    if dataset_name in ['adult', 'diabetes'] and len(real_df) > 10000:
        from sklearn.model_selection import train_test_split as _tts
        strat = real_df[target].astype(str) + '_' + real_df[sensitive].astype(str)
        _, real_df = _tts(real_df, test_size=10000, random_state=42, stratify=strat)
        real_df = real_df.reset_index(drop=True)

    print(f'  Real data  : {len(real_df)} rows × {real_df.shape[1]} cols')
    print(f'  Scenario   : {scenario}')
    print(f'  Generators : {", ".join(GENERATORS)}\n')

    # Build column labels dynamically from real_df columns
    col_labels = {c: c.replace('_', ' ').title() for c in real_df.columns}

    # KDE overlay: pick up to 6 numeric columns (prefer sensitive + target + high-variance)
    numeric_cols = real_df.select_dtypes(include='number').columns.tolist()
    priority = [sensitive, target]
    rest = [c for c in numeric_cols if c not in priority]
    kde_features = (priority + rest)[:6]

    mean_jsd = plot_heatmap_jsd(real_df, col_labels, out_dir, synth_root,
                                scenario, dataset_name)
    if mean_jsd.empty:
        print('  [SKIP] No synthetic data found — skipping KDE overlay.')
        return

    best_gen = select_best_generator(mean_jsd)
    plot_kde_overlay(real_df, best_gen, col_labels, kde_features,
                     out_dir, synth_root, scenario)

    print(f'\n  ✔ Done! Plots saved to {out_dir}')
    print('═' * 65 + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_datasets = list(DATASET_CONFIGS.keys())
    parser = argparse.ArgumentParser(
        description='SDV fidelity plots for bias2fair-synth.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--dataset', nargs='+', default=['compas'],
        choices=all_datasets,
        metavar='DATASET',
        help=(
            'One or more datasets to analyse.\n'
            f'Choices: {all_datasets}\n'
            'Examples:\n'
            '  --dataset compas\n'
            '  --dataset compas adult diabetes'
        )
    )
    parser.add_argument(
        '--scenario', type=str, default='S3_2.0',
        help='Reference scenario for fidelity (default: S3_2.0)'
    )
    args = parser.parse_args()

    datasets = list(dict.fromkeys(d.lower() for d in args.dataset))
    for i, dataset_name in enumerate(datasets, 1):
        print(f'\n  [{i}/{len(datasets)}] {dataset_name.upper()}')
        run_for_dataset(dataset_name, args.scenario)

    print(f'\n  All done: {", ".join(d.upper() for d in datasets)}')


if __name__ == '__main__':
    main()
