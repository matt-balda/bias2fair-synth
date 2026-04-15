"""
análise_conclusões.py
─────────────────────
Gráficos orientados a conclusões do experimento bias2fair-synth.
Responde diretamente às comparações do .agent-context.md §11:
  • S1 vs S2  → efeito da mitigação
  • S1 vs S3/S4 → efeito dos dados sintéticos
  • S2 vs S5  → valor adicional do sintético sobre a mitigação
"""
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import wilcoxon

# ── Setup ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 10,
})

OUT = 'reports/analysis'
os.makedirs(OUT, exist_ok=True)

SCENARIO_LABELS = {
    'S1': 'S1\nBaseline',
    'S2': 'S2\nMitigação\n(Reweighing)',
    'S3': 'S3\nAugmentation\n(Real+Synth)',
    'S4': 'S4\nMinority\nOversamp.',
    'S5': 'S5\nCombinado\n(Mitig.+Synth)',
}

COLORS = {
    'Baseline':       '#7f8c8d',
    'GaussianCopula': '#2980b9',
    'CTGAN':          '#27ae60',
    'TVAE':           '#e67e22',
    'TabDDPM':        '#8e44ad',
}

df = pd.read_csv('outputs/compas_results.csv')

# For S1/S2 rename generator for cleaner plots
df['gen_label'] = df['generator']


# ══════════════════════════════════════════════════════════════════════════
# 1. VISÃO GERAL — Todas as métricas-chave por cenário (CatBoost focus)
# ══════════════════════════════════════════════════════════════════════════
def plot_overview():
    metrics = {
        'F1-Score':         ('f1', '↑ melhor',  True),
        'AUC-ROC':          ('auc_roc', '↑ melhor', True),
        'Disparate Impact': ('disparate_impact', '↑ melhor (alvo ≥ 0.80)', False),
        'Stat. Parity Diff':('statistical_parity_difference', '↓ melhor (alvo → 0)', False),
        'Eq. Opp. Diff':    ('equal_opportunity_difference', '↓ melhor (alvo → 0)', False),
        'AAOD':             ('average_absolute_odds_difference', '↓ melhor (alvo → 0)', False),
    }

    gens = ['Baseline', 'GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']
    scenarios = ['S1', 'S2', 'S3', 'S4', 'S5']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (label, (col, subtitle, higher)) in zip(axes, metrics.items()):
        # mean per scenario/generator
        means = df.groupby(['scenario', 'generator'])[col].mean().unstack('generator')
        stds  = df.groupby(['scenario', 'generator'])[col].std().unstack('generator')

        x = np.arange(len(scenarios))
        width = 0.15
        for i, gen in enumerate(gens):
            if gen not in means.columns:
                continue
            vals = [means.loc[s, gen] if s in means.index else np.nan for s in scenarios]
            errs = [stds.loc[s, gen]  if s in stds.index  else 0      for s in scenarios]
            ax.bar(x + i*width - 2*width, vals, width,
                   label=gen, color=COLORS[gen], alpha=0.85,
                   yerr=errs, capsize=2, error_kw={'linewidth': 0.8})

        if col == 'disparate_impact':
            ax.axhline(0.8, color='red', linestyle='--', lw=1, label='80% Rule', alpha=0.8)
        if col == 'statistical_parity_difference':
            ax.axhline(0.0, color='green', linestyle=':', lw=1, alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios], fontsize=7)
        ax.set_title(f'{label}\n{subtitle}', fontsize=9, fontweight='bold')
        ax.set_ylabel(label, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.suptitle('Visão Geral — Todas as Métricas por Cenário e Gerador\n(médias ± DP, 10 sementes, 3 modelos)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(f'{OUT}/01_overview_all_metrics.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✔ 01_overview_all_metrics.png')


# ══════════════════════════════════════════════════════════════════════════
# 2. COMPARAÇÃO S1 vs S2 — Efeito da Mitigação
# ══════════════════════════════════════════════════════════════════════════
def plot_s1_vs_s2():
    s12 = df[df['scenario'].isin(['S1', 'S2'])].copy()
    metrics = ['f1', 'auc_roc', 'disparate_impact', 'statistical_parity_difference',
               'equal_opportunity_difference', 'average_absolute_odds_difference']
    labels  = ['F1-Score', 'AUC-ROC', 'Disp. Impact', 'Stat. Parity Diff',
                'Eq. Opp. Diff', 'AAOD']

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, col, label in zip(axes, metrics, labels):
        sns.boxplot(data=s12, x='scenario', y=col, hue='model',
                    palette='Set2', ax=ax, order=['S1', 'S2'],
                    linewidth=0.8)
        if col == 'disparate_impact':
            ax.axhline(0.8, color='red', ls='--', lw=1)

        # Wilcoxon p-value annotation
        for model in s12['model'].unique():
            a = s12[(s12['scenario']=='S1')&(s12['model']==model)][col].values
            b = s12[(s12['scenario']=='S2')&(s12['model']==model)][col].values
            if len(a) > 2 and len(b) > 2:
                try:
                    _, p = wilcoxon(a, b)
                    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
                except:
                    sig = '?'

        ax.set_title(f'{label}', fontsize=9, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticklabels(['S1 — Baseline', 'S2 — Mitigação'], fontsize=8)
        ax.legend(fontsize=7, title='Modelo', title_fontsize=7)

    fig.suptitle('Comparação S1 vs S2: Efeito do Reweighing (Mitigação de Viés)\n'
                 'Wilcoxon signed-rank — * p<0.05  ** p<0.01  *** p<0.001',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/02_s1_vs_s2_mitigation.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✔ 02_s1_vs_s2_mitigation.png')


# ══════════════════════════════════════════════════════════════════════════
# 3. COMPARAÇÃO S1 vs S3/S4 — Efeito dos Dados Sintéticos
# ══════════════════════════════════════════════════════════════════════════
def plot_synthetic_effect():
    synth = df[df['scenario'].isin(['S1', 'S3', 'S4'])].copy()
    # Simplify: show best model only (CatBoost) and keep generator info

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = [('f1', 'F1-Score', True), ('disparate_impact', 'Disparate Impact', True),
               ('equal_opportunity_difference', 'Eq. Opp. Diff', False)]

    for ax, (col, label, higher) in zip(axes, metrics):
        gen_order = ['Baseline', 'GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']
        sc_order  = ['S1', 'S3', 'S4']
        means = synth.groupby(['scenario', 'generator'])[col].mean().unstack('generator')
        stds  = synth.groupby(['scenario', 'generator'])[col].std().unstack('generator')

        x = np.arange(len(sc_order))
        width = 0.15
        for i, gen in enumerate(gen_order):
            if gen not in means.columns:
                continue
            vals = [means.loc[s, gen] if s in means.index else np.nan for s in sc_order]
            errs = [stds.loc[s, gen]  if s in stds.index  else 0      for s in sc_order]
            ax.bar(x + i*width - 2*width, vals, width,
                   label=gen, color=COLORS[gen], alpha=0.85,
                   yerr=errs, capsize=2, error_kw={'linewidth': 0.8})

        if col == 'disparate_impact':
            ax.axhline(0.8, color='red', ls='--', lw=1, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(['S1\nBaseline', 'S3\nAugmentation', 'S4\nMinority OS'], fontsize=9)
        ax.set_title(label, fontweight='bold')
        ax.legend(fontsize=7)

    fig.suptitle('Efeito dos Dados Sintéticos: S1 vs S3 (Augmentation) vs S4 (Minority Oversampling)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/03_s1_vs_synthetic.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✔ 03_s1_vs_synthetic.png')


# ══════════════════════════════════════════════════════════════════════════
# 4. COMPARAÇÃO S2 vs S5 — Valor adicional do sintético sobre mitigação
# ══════════════════════════════════════════════════════════════════════════
def plot_s2_vs_s5():
    sub = df[df['scenario'].isin(['S2', 'S5'])].copy()
    metrics = ['f1', 'auc_roc', 'disparate_impact', 'equal_opportunity_difference']
    labels  = ['F1-Score', 'AUC-ROC', 'Disparate Impact', 'Eq. Opp. Diff']

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    gens = ['Baseline', 'GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']

    for ax, col, label in zip(axes, metrics, labels):
        means = sub.groupby(['scenario', 'generator'])[col].mean().unstack('generator')
        stds  = sub.groupby(['scenario', 'generator'])[col].std().unstack('generator')

        for i, gen in enumerate(gens):
            if gen not in means.columns:
                continue
            sc_present = [s for s in ['S2', 'S5'] if s in means.index]
            vals = [means.loc[s, gen] if s in means.index else np.nan for s in ['S2', 'S5']]
            errs = [stds.loc[s, gen]  if s in stds.index  else 0      for s in ['S2', 'S5']]
            x = np.arange(2)
            ax.bar(x + i*0.15 - 2*0.15, vals, 0.15,
                   label=gen, color=COLORS[gen], alpha=0.85,
                   yerr=errs, capsize=2, error_kw={'linewidth': 0.8})

        if col == 'disparate_impact':
            ax.axhline(0.8, color='red', ls='--', lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['S2\nMitigação', 'S5\nCombinado'], fontsize=9)
        ax.set_title(label, fontweight='bold')
        ax.legend(fontsize=7)

    fig.suptitle('Valor Adicional do Sintético sobre a Mitigação: S2 vs S5',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/04_s2_vs_s5_combined.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✔ 04_s2_vs_s5_combined.png')


# ══════════════════════════════════════════════════════════════════════════
# 5. PARETO FRONTIER — Trade-off Fairness vs Performance
# ══════════════════════════════════════════════════════════════════════════
def plot_pareto():
    means = df.groupby(['scenario', 'generator'])[['f1', 'disparate_impact']].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 8))

    sc_markers = {'S1': 'o', 'S2': 's', 'S3': '^', 'S4': 'D', 'S5': 'P'}
    sc_colors  = {'S1': '#e74c3c', 'S2': '#2ecc71', 'S3': '#3498db',
                  'S4': '#f39c12', 'S5': '#9b59b6'}

    for _, row in means.iterrows():
        sc, gen = row['scenario'], row['generator']
        ax.scatter(row['disparate_impact'], row['f1'],
                   marker=sc_markers[sc], color=sc_colors[sc],
                   s=140, edgecolors='white', linewidths=0.8,
                   zorder=5, alpha=0.9)
        ax.annotate(f'{sc}\n{gen[:4]}', (row['disparate_impact'], row['f1']),
                    textcoords='offset points', xytext=(5, 4), fontsize=6.5)

    ax.axvline(0.8, color='red', ls='--', lw=1.2, alpha=0.7, label='80% Rule (DI ≥ 0.8)')
    ax.axvspan(0.8, ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 1.1,
               alpha=0.05, color='green')

    # Legend for scenarios
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker=sc_markers[s], color='w',
                       markerfacecolor=sc_colors[s], markersize=10,
                       label=SCENARIO_LABELS[s].replace('\n',' '))
               for s in sc_markers]
    ax.legend(handles=handles, fontsize=8, loc='lower right')

    ax.set_xlabel('Fairness — Disparate Impact (↑ melhor, alvo ≥ 0.80)', fontsize=10)
    ax.set_ylabel('Utilidade — F1-Score (↑ melhor)', fontsize=10)
    ax.set_title('Fronteira de Pareto: Trade-off Fairness × Performance\npor Cenário e Gerador',
                 fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(f'{OUT}/05_pareto_frontier.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✔ 05_pareto_frontier.png')


# ══════════════════════════════════════════════════════════════════════════
# 6. GENERATOR RANKING — Por cenário, qual gerador é melhor?
# ══════════════════════════════════════════════════════════════════════════
def plot_generator_ranking():
    synth_scenarios = ['S3', 'S4', 'S5']
    sub = df[df['scenario'].isin(synth_scenarios)]

    # Composite score: normalize F1 and DI to [0,1] and average
    # DI: capped at 1.0, higher=better
    # F1: higher=better
    means = sub.groupby(['scenario', 'generator'])[['f1', 'disparate_impact']].mean()
    means['di_capped'] = means['disparate_impact'].clip(upper=1.0)
    f1_min, f1_max = means['f1'].min(), means['f1'].max()
    di_min, di_max = means['di_capped'].min(), means['di_capped'].max()
    means['f1_norm'] = (means['f1'] - f1_min) / (f1_max - f1_min + 1e-9)
    means['di_norm'] = (means['di_capped'] - di_min) / (di_max - di_min + 1e-9)
    means['composite'] = 0.5 * means['f1_norm'] + 0.5 * means['di_norm']
    means = means.reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    for ax, sc in zip(axes, synth_scenarios):
        sc_data = means[means['scenario'] == sc].sort_values('composite', ascending=False)
        colors_bar = [COLORS.get(g, '#95a5a6') for g in sc_data['generator']]
        bars = ax.barh(sc_data['generator'], sc_data['composite'],
                       color=colors_bar, alpha=0.85, edgecolor='white')
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.set_title(f'{sc}\n{SCENARIO_LABELS[sc].strip()}', fontweight='bold')
        ax.set_xlabel('Score composto (F1 + DI normalizados)')
        ax.set_xlim(0, 1.15)
        ax.invert_yaxis()

    fig.suptitle('Ranking de Geradores por Cenário\n(Score Composto = 0.5×F1 + 0.5×DI, normalizados)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/06_generator_ranking.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✔ 06_generator_ranking.png')


# ══════════════════════════════════════════════════════════════════════════
# 7. VIOLIN — Estabilidade das métricas por gerador
# ══════════════════════════════════════════════════════════════════════════
def plot_stability():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (col, label) in zip(axes, [('f1', 'F1-Score'), ('disparate_impact', 'Disparate Impact')]):
        gen_order = ['Baseline', 'GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']
        palette   = [COLORS[g] for g in gen_order]
        sns.violinplot(data=df, x='generator', y=col, hue='scenario',
                       order=gen_order, palette='Set2', inner='quart', ax=ax)
        if col == 'disparate_impact':
            ax.axhline(0.8, color='red', ls='--', lw=1, label='80% Rule')
        ax.set_title(f'Estabilidade de {label} por Gerador', fontweight='bold')
        ax.set_xlabel('Gerador')
        ax.set_ylabel(label)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
        ax.tick_params(axis='x', rotation=15)

    fig.suptitle('Estabilidade das Métricas — Distribuição por Gerador e Cenário\n(10 sementes × 3 modelos)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/07_stability_violin.png', bbox_inches='tight')
    plt.close(fig)
    print('  ✔ 07_stability_violin.png')


# ══════════════════════════════════════════════════════════════════════════
# 8. TABELA RESUMO — Mean ± Std para todos os cenários
# ══════════════════════════════════════════════════════════════════════════
def export_summary_table():
    metrics_of_interest = ['f1', 'auc_roc', 'auc_pr',
                            'disparate_impact', 'statistical_parity_difference',
                            'equal_opportunity_difference', 'average_absolute_odds_difference']

    summary = df.groupby(['scenario', 'generator'])[metrics_of_interest].agg(['mean', 'std'])
    summary.columns = [f'{col}_{stat}' for col, stat in summary.columns]
    summary = summary.round(4).reset_index()
    summary.to_csv(f'{OUT}/summary_table.csv', index=False)
    print('  ✔ summary_table.csv')

    # Pretty table for each metric
    for col in ['f1', 'disparate_impact']:
        tbl = df.groupby(['scenario', 'generator'])[col].agg(['mean', 'std']).round(4)
        tbl['mean±std'] = tbl.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
        tbl[['mean±std']].unstack('generator').to_csv(f'{OUT}/table_{col}.csv')
    print('  ✔ table_f1.csv + table_disparate_impact.csv')


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('\n' + '═'*55)
    print('  Análise e Conclusões — bias2fair-synth')
    print('═'*55 + '\n')

    steps = [
        ('Visão geral — todas as métricas',           plot_overview),
        ('S1 vs S2 — efeito da mitigação',            plot_s1_vs_s2),
        ('S1 vs S3/S4 — efeito do sintético',         plot_synthetic_effect),
        ('S2 vs S5 — valor adicional combinado',       plot_s2_vs_s5),
        ('Pareto frontier F1 × Fairness',              plot_pareto),
        ('Ranking de geradores por cenário',           plot_generator_ranking),
        ('Estabilidade (Violin) por gerador',          plot_stability),
        ('Tabela resumo CSV',                          export_summary_table),
    ]

    for label, fn in steps:
        print(f'  → {label}...')
        fn()

    print(f'\n  ✔ Todos os gráficos em {OUT}/')
    print('═'*55 + '\n')
