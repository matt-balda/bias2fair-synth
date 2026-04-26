import warnings
warnings.filterwarnings('ignore')
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import wilcoxon

sns.set_theme(style='whitegrid', font_scale=1.1)
PALETTE = {'S1':'#e63946','S2':'#457b9d','S3':'#2a9d8f','S4':'#e9c46a','S5':'#f4a261','S6':'#6a4c93'}
MODEL_MARKERS = {'LogisticRegression':'o','SVM':'s','CatBoost':'D'}
GENERATOR_PALETTE = {'GaussianCopula':'#4cc9f0','CTGAN':'#f72585','TVAE':'#7209b7','TabDDPM':'#3a0ca3'}

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

WILCOXON_COMPARISONS = [
    ('S1',  'None',       'S3_1.5', 'None',       'S1→S3 (1.5x)'),
    ('S1',  'None',       'S3_2.0', 'None',       'S1→S3 (2.0x)'),
    ('S1',  'None',       'S3_3.0', 'None',       'S1→S3 (3.0x)'),
    ('S1',  'None',       'S4',     'None',       'S1→S4'),
    ('S1',  'None',       'S5',     'None',       'S1→S5'),
    ('S1',  'None',       'S6',     'Reweighing', 'S1→S6'),
    ('S2',  'Reweighing', 'S4',     'None',       'S2Rew→S4'),
    ('S2',  'Reweighing', 'S5',     'None',       'S2Rew→S5'),
    ('S2',  'Reweighing', 'S6',     'Reweighing', 'S2Rew→S6'),
]


def run_for_dataset(dataset_name: str) -> None:
    """Generate all paper plots for a single dataset."""

    out = os.path.join('plots', dataset_name, 'paper')
    os.makedirs(out, exist_ok=True)

    results_path = f'outputs/{dataset_name}_results.csv'
    if not os.path.exists(results_path):
        print(f'  [SKIP] {results_path} not found — run the experiment first.')
        return

    print(f'\n{"═"*60}')
    print(f'  generate_paper_plots  ·  {dataset_name.upper()}')
    print(f'{"═"*60}\n')

    df = pd.read_csv(results_path)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out, name), dpi=180, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved {name}')

    # ── Scenario label mapping ─────────────────────────────────────────────
    def simplify_scenario(row):
        s = row['scenario']
        m = row['mitigator'] if pd.notna(row.get('mitigator')) else 'None'
        if s == 'S1': return 'S1\nBaseline'
        if s == 'S2' and m == 'Reweighing': return 'S2\nReweigh'
        if s == 'S2' and m == 'LFR': return 'S2\nLFR'
        if s == 'S2' and m == 'DIRemover': return 'S2\nDIR'
        if s == 'S3_1.5': return 'S3\n1.5x'
        if s == 'S3_2.0': return 'S3\n2.0x'
        if s == 'S3_3.0': return 'S3\n3.0x'
        if s == 'S4': return 'S4\nMinority'
        if s == 'S5': return 'S5\nCombined'
        if s == 'S6': return 'S6\nRew+Synth'
        return s

    df['scenario_label'] = df.apply(simplify_scenario, axis=1)

    # ── Build sub-group labels (for S2 and S3 variants) ───────────────────
    def get_scenario_group(row):
        s = row['scenario']
        if s == 'S1': return 'S1'
        if s == 'S4': return 'S4'
        if s == 'S5': return 'S5'
        if s == 'S6': return 'S6'
        if s.startswith('S3'):
            alpha = s.split('_')[1] if '_' in s else '?'
            return f'S3 ({alpha}x)'
        return s

    means = df.groupby(['scenario', 'generator', 'model'])[
        ['f1', 'disparate_impact', 'statistical_parity_difference',
         'equal_opportunity_difference', 'average_absolute_odds_difference']
    ].mean().reset_index()

    means_s2 = df[df['scenario'] == 'S2'].groupby(
        ['scenario', 'mitigator', 'generator', 'model']
    )[['f1', 'disparate_impact', 'statistical_parity_difference',
       'equal_opportunity_difference', 'average_absolute_odds_difference']].mean().reset_index()
    _s2_label_map = {'Reweighing': 'S2_1 (Reweigh)', 'LFR': 'S2_2 (LFR)', 'DIRemover': 'S2_3 (DIR)'}
    means_s2['sg'] = means_s2['mitigator'].map(_s2_label_map).fillna('S2')

    means_other = means[means['scenario'] != 'S2'].copy()
    means_other['mitigator'] = None
    means_other['sg'] = means_other.apply(get_scenario_group, axis=1)

    means_full = pd.concat([means_other, means_s2], ignore_index=True)

    # Shared legend handles
    legend_sc  = [mpatches.Patch(color=c, label=sg) for sg, c in SG_COLORS.items()]
    legend_mod = [plt.Line2D([0],[0], marker=m, color='gray', linestyle='None',
                             markersize=8, label=mod) for mod, m in MODEL_MARKERS.items()]

    # ── PLOT 2.1: Pareto Scatter F1 vs DI ────────────────────────────────
    print('Plot 2.1 – Pareto scatter')
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvspan(0.8, 1.25, alpha=0.08, color='green', label='Zona de Fairness (80% rule)')
    ax.axvline(1.0, color='green', linestyle='--', lw=1.2, alpha=0.6)
    ax.axvline(0.8, color='red', linestyle='--', lw=1, alpha=0.5)
    ax.axvline(1.25, color='red', linestyle='--', lw=1, alpha=0.5)
    for sg, grp in means_full.groupby('sg'):
        for model, mgrp in grp.groupby('model'):
            ax.scatter(mgrp['disparate_impact'], mgrp['f1'],
                       color=SG_COLORS.get(sg, 'gray'),
                       marker=MODEL_MARKERS.get(model, 'o'),
                       s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.legend(handles=legend_sc + legend_mod, loc='upper right', fontsize=8, ncol=2)
    ax.set_xlabel('Disparate Impact (DI) →  Fairness', fontsize=11)
    ax.set_ylabel('F1-Score →  Utilidade', fontsize=11)
    ax.set_title(f'Trade-off Fairness × Utilidade — {dataset_name.upper()}', fontsize=13, fontweight='bold')
    ax.set_xlim(0.5, 5.5)
    save(fig, 'plot2_1_pareto_scatter.png')

    # ── PLOT 2.2: F1 vs SPD ───────────────────────────────────────────────
    print('Plot 2.2 – Scatter F1 vs SPD')
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvline(0.0, color='green', linestyle='--', lw=1.5, alpha=0.7, label='SPD ideal=0')
    ax.axvspan(-0.1, 0.1, alpha=0.08, color='green')
    for sg, grp in means_full.groupby('sg'):
        for model, mgrp in grp.groupby('model'):
            ax.scatter(mgrp['statistical_parity_difference'], mgrp['f1'],
                       color=SG_COLORS.get(sg, 'gray'),
                       marker=MODEL_MARKERS.get(model, 'o'),
                       s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.legend(handles=legend_sc + legend_mod + [
        plt.Line2D([0],[0], color='green', linestyle='--', label='SPD=0')],
        loc='upper right', fontsize=8, ncol=2)
    ax.set_xlabel('Statistical Parity Difference (SPD)', fontsize=11)
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title(f'Trade-off F1 × SPD — {dataset_name.upper()}', fontsize=13, fontweight='bold')
    save(fig, 'plot2_2_scatter_f1_spd.png')

    # ── PLOT 2.3: F1 vs EOD ───────────────────────────────────────────────
    print('Plot 2.3 – Scatter F1 vs EOD')
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvline(0.0, color='green', linestyle='--', lw=1.5, alpha=0.7, label='EOD ideal=0')
    ax.axvspan(-0.1, 0.1, alpha=0.08, color='green')
    for sg, grp in means_full.groupby('sg'):
        for model, mgrp in grp.groupby('model'):
            ax.scatter(mgrp['equal_opportunity_difference'], mgrp['f1'],
                       color=SG_COLORS.get(sg, 'gray'),
                       marker=MODEL_MARKERS.get(model, 'o'),
                       s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.legend(handles=legend_sc + legend_mod + [
        plt.Line2D([0],[0], color='green', linestyle='--', label='EOD=0')],
        loc='upper right', fontsize=8, ncol=2)
    ax.set_xlabel('Equal Opportunity Difference (EOD)', fontsize=11)
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title(f'Trade-off F1 × EOD — {dataset_name.upper()}', fontsize=13, fontweight='bold')
    save(fig, 'plot2_3_scatter_f1_eod.png')

    # ── PLOT 2.4: F1 vs AOD ───────────────────────────────────────────────
    print('Plot 2.4 – Scatter F1 vs AOD')
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvline(0.0, color='green', linestyle='--', lw=1.5, alpha=0.7, label='AOD ideal=0')
    ax.axvspan(-0.1, 0.1, alpha=0.08, color='green')
    for sg, grp in means_full.groupby('sg'):
        for model, mgrp in grp.groupby('model'):
            ax.scatter(mgrp['average_absolute_odds_difference'], mgrp['f1'],
                       color=SG_COLORS.get(sg, 'gray'),
                       marker=MODEL_MARKERS.get(model, 'o'),
                       s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.legend(handles=legend_sc + legend_mod + [
        plt.Line2D([0],[0], color='green', linestyle='--', label='AOD=0')],
        loc='upper right', fontsize=8, ncol=2)
    ax.set_xlabel('Average Absolute Odds Difference (AOD)', fontsize=11)
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title(f'Trade-off F1 × AOD — {dataset_name.upper()}', fontsize=13, fontweight='bold')
    save(fig, 'plot2_4_scatter_f1_aod.png')

    # ── PLOT 3.1: Violin SPD ──────────────────────────────────────────────
    print('Plot 3.1 – Violin SPD')
    order = ['S1\nBaseline','S2\nReweigh','S2\nLFR','S2\nDIR',
             'S3\n1.5x','S3\n2.0x','S3\n3.0x','S4\nMinority','S5\nCombined','S6\nRew+Synth']
    present = [o for o in order if o in df['scenario_label'].values]
    pal = {
        'S1\nBaseline': '#e63946', 'S2\nReweigh': '#457b9d',
        'S2\nLFR': '#1d3557', 'S2\nDIR': '#a8dadc',
        'S3\n1.5x': '#2a9d8f', 'S3\n2.0x': '#52b788', 'S3\n3.0x': '#40916c',
        'S4\nMinority': '#e9c46a', 'S5\nCombined': '#f4a261', 'S6\nRew+Synth': '#6a4c93',
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.violinplot(data=df, x='scenario_label', y='statistical_parity_difference',
                   order=present, palette=pal, inner='box', ax=ax, cut=0)
    ax.axhline(0, color='green', linestyle='--', lw=1.5, label='SPD ideal = 0')
    ax.set_xlabel('')
    ax.set_ylabel('Statistical Parity Difference (SPD)', fontsize=11)
    ax.set_title(f'SPD Distribution by Scenario — {dataset_name.upper()}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    save(fig, 'plot3_1_violin_spd_by_scenario.png')

    # ── PLOT 4.1: Augmentation ratio ─────────────────────────────────────
    print('Plot 4.1 – Augmentation ratio')
    s3 = df[df['scenario'].isin(['S3_1.5','S3_2.0','S3_3.0'])].copy()
    s3['ratio'] = s3['scenario'].map({'S3_1.5':1.5,'S3_2.0':2.0,'S3_3.0':3.0})
    grp = s3.groupby(['ratio','generator'])[['f1','disparate_impact']].agg(['mean','std']).reset_index()
    grp.columns = ['ratio','generator','f1_mean','f1_std','di_mean','di_std']
    s1_di = df[df['scenario']=='S1']['disparate_impact'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for gen, gdf in grp.groupby('generator'):
        c = GENERATOR_PALETTE.get(gen, 'gray')
        gdf = gdf.sort_values('ratio')
        axes[0].plot(gdf['ratio'], gdf['f1_mean'], marker='o', color=c, label=gen)
        axes[0].fill_between(gdf['ratio'],
                              gdf['f1_mean']-gdf['f1_std'],
                              gdf['f1_mean']+gdf['f1_std'], alpha=0.15, color=c)
        axes[1].plot(gdf['ratio'], gdf['di_mean'], marker='o', color=c, label=gen)
        axes[1].fill_between(gdf['ratio'],
                              gdf['di_mean']-gdf['di_std'],
                              gdf['di_mean']+gdf['di_std'], alpha=0.15, color=c)
    axes[1].axhline(s1_di, color='#e63946', linestyle='--', lw=1.5, label=f'S1 DI={s1_di:.2f}')
    axes[1].axhline(1.0, color='green', linestyle=':', lw=1.5, label='DI ideal=1')
    for ax, ylabel in [(axes[0],'F1-Score'),(axes[1],'Disparate Impact')]:
        ax.set_xlabel('Augmentation Ratio (α)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks([1.5,2.0,3.0])
        ax.legend(fontsize=8)
    axes[0].set_title(f'F1 vs Augmentation Ratio (S3) — {dataset_name.upper()}', fontweight='bold')
    axes[1].set_title(f'DI vs Augmentation Ratio (S3) — {dataset_name.upper()}', fontweight='bold')
    save(fig, 'plot4_1_augmentation_ratio.png')

    # ── PLOT 6.1: Wilcoxon heatmap ────────────────────────────────────────
    print('Plot 6.1 – Wilcoxon heatmap')
    models_in_df    = df['model'].unique()
    generators_synth = [g for g in df['generator'].unique() if g != 'Baseline']

    wil_results = []
    for s_a, m_a, s_b, m_b, label in WILCOXON_COMPARISONS:
        mask_a = (
            (df['scenario'] == s_a) &
            (df['mitigator'].fillna('None') == m_a) &
            (df['generator'] == 'Baseline')
        )
        for model in models_in_df:
            a_base = df[mask_a & (df['model'] == model)].sort_values('seed')
            for gen in generators_synth:
                mask_b = (
                    (df['scenario'] == s_b) &
                    (df['mitigator'].fillna('None') == m_b) &
                    (df['generator'] == gen) &
                    (df['model'] == model)
                )
                b = df[mask_b].sort_values('seed')
                shared = set(a_base['seed']) & set(b['seed'])
                a = a_base[a_base['seed'].isin(shared)].sort_values('seed')
                b = b[b['seed'].isin(shared)].sort_values('seed')
                if len(a) < 3 or len(b) < 3:
                    continue
                try:
                    _, p_f1 = wilcoxon(a['f1'].values, b['f1'].values)
                except Exception:
                    p_f1 = float('nan')
                try:
                    _, p_di = wilcoxon(a['disparate_impact'].values, b['disparate_impact'].values)
                except Exception:
                    p_di = float('nan')
                wil_results.append({
                    'comparison': label, 'generator': gen, 'model': model,
                    'p_f1': p_f1, 'p_di': p_di,
                })

    wil_df = pd.DataFrame(wil_results)
    metrics_dir = os.path.join('plots', dataset_name, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    wil_df.to_csv(os.path.join(metrics_dir, 'wilcoxon_full.csv'), index=False)
    print(f'  computed {len(wil_df)} Wilcoxon tests')

    model_abbr = {'LogisticRegression': 'LR', 'SVM': 'SVM', 'CatBoost': 'CB'}
    wil_df['config'] = wil_df['generator'] + '|' + wil_df['model'].map(model_abbr)
    comp_order = [c[4] for c in WILCOXON_COMPARISONS]

    for metric_label, p_col in [('F1', 'p_f1'), ('DI', 'p_di')]:
        piv = wil_df.pivot_table(index='comparison', columns='config', values=p_col, aggfunc='mean')
        piv = piv.reindex([c for c in comp_order if c in piv.index])
        if piv.empty:
            continue
        n_cols_piv = len(piv.columns)
        fig, ax = plt.subplots(figsize=(max(10, n_cols_piv * 0.85 + 2), len(piv) * 0.75 + 2))
        sns.heatmap(
            piv, annot=True, fmt='.3f', cmap='RdYlGn_r',
            vmin=0, vmax=0.1, ax=ax, linewidths=0.4,
            annot_kws={'size': 7}, cbar_kws={'label': 'p-value'},
        )
        ax.set_title(
            f'Wilcoxon p-values – {metric_label} — {dataset_name.upper()}\n'
            f'(green = p<0.05 = significant)',
            fontweight='bold', fontsize=11,
        )
        ax.set_xlabel('Generator | Model', fontsize=10)
        ax.set_ylabel('Scenario Comparison', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
        save(fig, f'plot6_1_wilcoxon_{metric_label.lower()}.png')

    print(f'\n  ✔ Done! Plots saved to {out}\n')


def main():
    all_datasets = ['compas', 'adult', 'diabetes']
    parser = argparse.ArgumentParser(
        description='Generate paper plots for bias2fair-synth.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--dataset', nargs='+', default=['compas'],
        choices=all_datasets,
        metavar='DATASET',
        help=(
            'One or more datasets to plot.\n'
            f'Choices: {all_datasets}\n'
            'Examples:\n'
            '  --dataset compas\n'
            '  --dataset compas adult diabetes'
        )
    )
    args = parser.parse_args()

    datasets = list(dict.fromkeys(d.lower() for d in args.dataset))  # deduplicate, keep order
    for i, dataset_name in enumerate(datasets, 1):
        print(f'\n  [{i}/{len(datasets)}] {dataset_name.upper()}')
        run_for_dataset(dataset_name)

    print(f'\n  All done: {", ".join(d.upper() for d in datasets)}')


if __name__ == '__main__':
    main()
