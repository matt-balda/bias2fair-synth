import warnings
warnings.filterwarnings('ignore')
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

OUT = 'plots/paper'
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}')

df = pd.read_csv('outputs/compas_results.csv')

# Map scenario labels
def simplify_scenario(row):
    s = row['scenario']
    m = row['mitigator'] if pd.notna(row['mitigator']) else 'None'
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

# ── PLOT 1.1: Baseline Bias ──────────────────────────────────────────────────
print('Plot 1.1 – Baseline bias')
s1 = df[df['scenario']=='S1']
metrics_baseline = s1.groupby('model')[['disparate_impact','statistical_parity_difference']].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = ['#e63946','#457b9d','#2a9d8f']
for ax, col, title, ref, reflabel in [
    (axes[0], 'disparate_impact', 'Disparate Impact (DI)', 1.0, 'Ideal DI=1'),
    (axes[1], 'statistical_parity_difference', 'Statistical Parity Diff. (SPD)', 0.0, 'Ideal SPD=0'),
]:
    bars = ax.barh(metrics_baseline['model'], metrics_baseline[col], color=colors, edgecolor='white', height=0.5)
    ax.axvline(ref, color='black', linestyle='--', lw=1.5, label=reflabel)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(col.replace('_',' ').title())
    for bar, val in zip(bars, metrics_baseline[col]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9)
    ax.legend(fontsize=8)
fig.suptitle('Bias no Baseline (S1) – Motivação do Paper', fontsize=13, fontweight='bold')
save(fig, 'plot1_1_baseline_bias.png')

# ── PLOT 1.2: Prediction rates by group ──────────────────────────────────────
print('Plot 1.2 – Prediction rates by group')
try:
    pred_files = []
    pred_dir = 'outputs/predictions'
    if os.path.exists(pred_dir):
        for f in os.listdir(pred_dir):
            if f.endswith('.csv'):
                pred_files.append(pd.read_csv(os.path.join(pred_dir, f)))
    if pred_files:
        preds = pd.concat(pred_files)
        if 'race' in preds.columns and 'predicted' in preds.columns:
            rates = preds.groupby(['scenario','race'])['predicted'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8,4))
            for i, sc in enumerate(['S1','S4','S5','S6']):
                d = rates[rates['scenario']==sc]
                if len(d) > 0:
                    ax.bar([i-0.15, i+0.15],
                           [d[d['race']==0]['predicted'].values[0] if len(d[d['race']==0])>0 else 0,
                            d[d['race']==1]['predicted'].values[0] if len(d[d['race']==1])>0 else 0],
                           width=0.28, color=['#e63946','#457b9d'])
            ax.set_xticks(range(4)); ax.set_xticklabels(['S1','S4','S5','S6'])
            ax.set_ylabel('Taxa de Predição Positiva')
            ax.set_title('Predições por Grupo Racial por Cenário')
            red_p = mpatches.Patch(color='#e63946', label='African-American')
            blue_p = mpatches.Patch(color='#457b9d', label='Caucasian')
            ax.legend(handles=[red_p, blue_p])
            save(fig, 'plot1_2_group_prediction_rates.png')
except Exception as e:
    print(f'  skip 1.2: {e}')

# ── PLOT 2.1: Pareto Scatter F1 vs DI ────────────────────────────────────────
print('Plot 2.1 – Pareto scatter')
means = df.groupby(['scenario','generator','model'])[['f1','disparate_impact','statistical_parity_difference']].mean().reset_index()

def get_scenario_group(s):
    if s=='S1': return 'S1'
    if s.startswith('S2'): return 'S2'
    if s.startswith('S3'): return 'S3'
    if s=='S4': return 'S4'
    if s=='S5': return 'S5'
    if s=='S6': return 'S6'
    return s

means['sg'] = means['scenario'].apply(get_scenario_group)
sg_colors = {'S1':'#e63946','S2':'#457b9d','S3':'#2a9d8f','S4':'#e9c46a','S5':'#f4a261','S6':'#6a4c93'}

fig, ax = plt.subplots(figsize=(9, 6))
ax.axvspan(0.8, 1.25, alpha=0.08, color='green', label='Zona de Fairness (80% rule)')
ax.axvline(1.0, color='green', linestyle='--', lw=1.2, alpha=0.6)
ax.axvline(0.8, color='red', linestyle='--', lw=1, alpha=0.5)
ax.axvline(1.25, color='red', linestyle='--', lw=1, alpha=0.5)
for sg, grp in means.groupby('sg'):
    for model, mgrp in grp.groupby('model'):
        ax.scatter(mgrp['disparate_impact'], mgrp['f1'],
                   color=sg_colors.get(sg,'gray'),
                   marker=MODEL_MARKERS.get(model,'o'),
                   s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
legend_sc = [mpatches.Patch(color=c, label=sg) for sg, c in sg_colors.items()]
legend_mod = [plt.Line2D([0],[0], marker=m, color='gray', linestyle='None', markersize=8, label=mod)
              for mod, m in MODEL_MARKERS.items()]
ax.legend(handles=legend_sc+legend_mod, loc='upper right', fontsize=8, ncol=2)
ax.set_xlabel('Disparate Impact (DI) →  Fairness', fontsize=11)
ax.set_ylabel('F1-Score →  Utilidade', fontsize=11)
ax.set_title('Trade-off Fairness × Utilidade (Visão Pareto)', fontsize=13, fontweight='bold')
ax.set_xlim(0.5, 5.5)
save(fig, 'plot2_1_pareto_scatter.png')

# ── PLOT 2.2: Scatter F1 vs SPD ──────────────────────────────────────────────
print('Plot 2.2 – Scatter F1 vs SPD')
fig, ax = plt.subplots(figsize=(9, 6))
ax.axvline(0.0, color='green', linestyle='--', lw=1.5, alpha=0.7, label='SPD ideal=0')
ax.axvspan(-0.1, 0.1, alpha=0.08, color='green')
for sg, grp in means.groupby('sg'):
    for model, mgrp in grp.groupby('model'):
        ax.scatter(mgrp['statistical_parity_difference'], mgrp['f1'],
                   color=sg_colors.get(sg,'gray'),
                   marker=MODEL_MARKERS.get(model,'o'),
                   s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
ax.legend(handles=legend_sc+legend_mod+[plt.Line2D([0],[0], color='green', linestyle='--', label='SPD=0')],
          loc='upper right', fontsize=8, ncol=2)
ax.set_xlabel('Statistical Parity Difference (SPD)', fontsize=11)
ax.set_ylabel('F1-Score', fontsize=11)
ax.set_title('Trade-off F1 × SPD por Cenário', fontsize=13, fontweight='bold')
save(fig, 'plot2_2_scatter_f1_spd.png')

# ── PLOT 3.1: Violin SPD by Scenario ─────────────────────────────────────────
print('Plot 3.1 – Violin SPD')
order = ['S1\nBaseline','S2\nReweigh','S2\nLFR','S2\nDIR','S3\n1.5x','S3\n2.0x','S3\n3.0x','S4\nMinority','S5\nCombined','S6\nRew+Synth']
present = [o for o in order if o in df['scenario_label'].values]
pal = {'S1\nBaseline':'#e63946','S2\nReweigh':'#457b9d','S2\nLFR':'#1d3557','S2\nDIR':'#a8dadc',
       'S3\n1.5x':'#2a9d8f','S3\n2.0x':'#52b788','S3\n3.0x':'#40916c',
       'S4\nMinority':'#e9c46a','S5\nCombined':'#f4a261','S6\nRew+Synth':'#6a4c93'}
fig, ax = plt.subplots(figsize=(13, 5))
sns.violinplot(data=df, x='scenario_label', y='statistical_parity_difference',
               order=present, palette=pal, inner='box', ax=ax, cut=0)
ax.axhline(0, color='green', linestyle='--', lw=1.5, label='SPD ideal = 0')
ax.set_xlabel('')
ax.set_ylabel('Statistical Parity Difference (SPD)', fontsize=11)
ax.set_title('Distribuição de SPD por Cenário (10 seeds)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
save(fig, 'plot3_1_violin_spd_by_scenario.png')

# ── PLOT 3.2: Heatmap Scenario x Model ───────────────────────────────────────
print('Plot 3.2 – Heatmap scenario × model')
pivot_di = df.groupby(['scenario_label','model'])['disparate_impact'].mean().unstack()
pivot_f1 = df.groupby(['scenario_label','model'])['f1'].mean().unstack()
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, pivot, title, cmap, center in [
    (axes[0], pivot_di, 'Disparate Impact Médio', 'RdYlGn_r', 1.0),
    (axes[1], pivot_f1, 'F1-Score Médio', 'YlGn', None),
]:
    idx = [i for i in present if i in pivot.index]
    sns.heatmap(pivot.loc[idx], annot=True, fmt='.3f', cmap=cmap,
                center=center, ax=ax, linewidths=0.5, annot_kws={'size':8})
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
save(fig, 'plot3_2_heatmap_scenario_model.png')

# ── PLOT 4.1: Line augmentation ratio ────────────────────────────────────────
print('Plot 4.1 – Augmentation ratio')
s3 = df[df['scenario'].isin(['S3_1.5','S3_2.0','S3_3.0'])].copy()
s3['ratio'] = s3['scenario'].map({'S3_1.5':1.5,'S3_2.0':2.0,'S3_3.0':3.0})
grp = s3.groupby(['ratio','generator'])[['f1','disparate_impact']].agg(['mean','std']).reset_index()
grp.columns = ['ratio','generator','f1_mean','f1_std','di_mean','di_std']

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

s1_di = df[df['scenario']=='S1']['disparate_impact'].mean()
axes[1].axhline(s1_di, color='#e63946', linestyle='--', lw=1.5, label=f'S1 DI={s1_di:.2f}')
axes[1].axhline(1.0, color='green', linestyle=':', lw=1.5, label='DI ideal=1')
for ax, ylabel in [(axes[0],'F1-Score'),(axes[1],'Disparate Impact')]:
    ax.set_xlabel('Razão de Augmentação Sintética', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks([1.5,2.0,3.0])
    ax.legend(fontsize=8)
axes[0].set_title('F1 vs. Razão de Augmentação (S3)', fontweight='bold')
axes[1].set_title('DI vs. Razão de Augmentação (S3)', fontweight='bold')
save(fig, 'plot4_1_augmentation_ratio.png')

# ── PLOT 4.2: Heatmap DI generator × ratio ───────────────────────────────────
print('Plot 4.2 – Heatmap DI generator × ratio')
pivot_s3 = s3.groupby(['generator','ratio'])['disparate_impact'].mean().unstack()
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot_s3, annot=True, fmt='.2f', cmap='RdYlGn_r', center=1.0, ax=ax,
            linewidths=0.5, annot_kws={'size':9})
ax.set_title('DI Médio por Gerador × Razão Augmentação (S3)', fontweight='bold')
ax.set_xlabel('Razão de Augmentação')
ax.set_ylabel('Gerador')
save(fig, 'plot4_2_heatmap_di_generator_ratio.png')

# ── PLOT 6.1: Wilcoxon heatmap (recalculated from raw data) ─────────────────
print('Plot 6.1 – Wilcoxon heatmap (full scenarios)')

# All comparisons: (baseline_scenario, baseline_mitigator, target_scenario, label)
WILCOXON_COMPARISONS = [
    ('S1',  'None',       'S3_1.5', 'None', 'S1→S3 (1.5x)'),
    ('S1',  'None',       'S3_2.0', 'None', 'S1→S3 (2.0x)'),
    ('S1',  'None',       'S3_3.0', 'None', 'S1→S3 (3.0x)'),
    ('S1',  'None',       'S4',     'None', 'S1→S4'),
    ('S1',  'None',       'S5',     'None', 'S1→S5'),
    ('S1',  'None',       'S6',     'Reweighing', 'S1→S6'),
    ('S2',  'Reweighing', 'S4',     'None', 'S2Rew→S4'),
    ('S2',  'Reweighing', 'S5',     'None', 'S2Rew→S5'),
    ('S2',  'Reweighing', 'S6',     'Reweighing', 'S2Rew→S6'),
]

models    = df['model'].unique()
generators_synth = [g for g in df['generator'].unique() if g != 'Baseline']

wil_results = []
for s_a, m_a, s_b, m_b, label in WILCOXON_COMPARISONS:
    mask_a = (df['scenario'] == s_a) & (df['mitigator'].fillna('None') == m_a) & (df['generator'] == 'Baseline')
    for model in models:
        a_base = df[mask_a & (df['model'] == model)].sort_values('seed')
        for gen in generators_synth:
            mask_b = (df['scenario'] == s_b) & (df['mitigator'].fillna('None') == m_b) & \
                     (df['generator'] == gen) & (df['model'] == model)
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
                'p_f1': p_f1, 'p_di': p_di
            })

wil_df = pd.DataFrame(wil_results)
os.makedirs('plots/metrics', exist_ok=True)
wil_df.to_csv('plots/metrics/wilcoxon_full.csv', index=False)
print(f'  computed {len(wil_df)} Wilcoxon tests')

# Short model labels
model_abbr = {'LogisticRegression': 'LR', 'SVM': 'SVM', 'CatBoost': 'CB'}
wil_df['config'] = wil_df['generator'] + '|' + wil_df['model'].map(model_abbr)

# Desired row order
comp_order = [c[4] for c in WILCOXON_COMPARISONS]

for metric_label, p_col in [('F1', 'p_f1'), ('DI', 'p_di')]:
    piv = wil_df.pivot_table(index='comparison', columns='config', values=p_col, aggfunc='mean')
    # reorder rows
    piv = piv.reindex([c for c in comp_order if c in piv.index])

    n_cols = len(piv.columns)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.85 + 2), len(piv) * 0.75 + 2))
    sns.heatmap(
        piv, annot=True, fmt='.3f', cmap='RdYlGn_r',
        vmin=0, vmax=0.1, ax=ax, linewidths=0.4,
        annot_kws={'size': 7}, cbar_kws={'label': 'p-value'}
    )
    # Mark significant cells with border
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            if not np.isnan(val) and val < 0.05:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                             edgecolor='black', lw=1.5))
    ax.set_title(
        f'Wilcoxon p-valores – {metric_label}\n'
        f'(verde = p<0.05 = significativo  |  borda preta = significativo)',
        fontweight='bold', fontsize=11
    )
    ax.set_xlabel('Gerador | Modelo', fontsize=10)
    ax.set_ylabel('Comparação de Cenários', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    save(fig, f'plot6_1_wilcoxon_{metric_label.lower()}.png')

print('\nDone! Plots saved to', OUT)

def add_extra_plots():
    # ── PLOT 3.3: Grouped bars F1 and DI by scenario × model ──────────────
    print('Plot 3.3 – Grouped bars')
    key_scenarios = ['S1\nBaseline','S2\nReweigh','S4\nMinority','S5\nCombined','S6\nRew+Synth']
    sub = df[df['scenario_label'].isin(key_scenarios)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel, title in [
        (axes[0], 'f1', 'F1-Score', 'F1-Score por Cenário × Modelo'),
        (axes[1], 'disparate_impact', 'Disparate Impact', 'DI por Cenário × Modelo'),
    ]:
        agg = sub.groupby(['scenario_label','model'])[metric].mean().reset_index()
        models = agg['model'].unique()
        x = np.arange(len(key_scenarios))
        w = 0.25
        mc = {'LogisticRegression':'#4cc9f0','SVM':'#f72585','CatBoost':'#7209b7'}
        for i, model in enumerate(models):
            vals = [agg[(agg['scenario_label']==sc)&(agg['model']==model)][metric].values
                    for sc in key_scenarios]
            vals = [v[0] if len(v)>0 else 0 for v in vals]
            ax.bar(x + i*w - w, vals, width=w, label=model, color=mc.get(model,'gray'), alpha=0.85)
        if metric == 'disparate_impact':
            ax.axhline(1.0, color='green', linestyle='--', lw=1.2, label='DI=1')
            ax.axhline(0.8, color='red', linestyle=':', lw=1, label='Limite 80%')
        ax.set_xticks(x)
        ax.set_xticklabels(key_scenarios, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
    save(fig, 'plot3_3_grouped_bars_f1_di.png')

    # ── PLOT 5.3: Radar chart ─────────────────────────────────────────────
    print('Plot 5.3 – Radar quality')
    gens = ['GaussianCopula','CTGAN','TVAE','TabDDPM']
    s4_df = df[df['scenario']=='S4']
    radar_data = s4_df.groupby('generator').agg(
        f1=('f1','mean'),
        di=('disparate_impact','mean'),
        spd=('statistical_parity_difference','mean'),
        auc=('auc_roc','mean')
    ).reindex(gens)
    radar_data['fairness_score'] = 1 - radar_data['di'].sub(1).abs().clip(0,1)
    radar_data['spd_score'] = 1 - radar_data['spd'].abs().clip(0,1)
    cols = ['f1','auc','fairness_score','spd_score']
    labels = ['F1-Score','AUC-ROC','Fairness (DI)','Fairness (SPD)']
    N = len(cols)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    for gen in gens:
        vals = radar_data.loc[gen, cols].tolist()
        vals_norm = [(v - 0.5) / 0.5 for v in vals]
        vals_norm += vals_norm[:1]
        ax.plot(angles, vals_norm, linewidth=2, label=gen, color=GENERATOR_PALETTE.get(gen,'gray'))
        ax.fill(angles, vals_norm, alpha=0.08, color=GENERATOR_PALETTE.get(gen,'gray'))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title('Qualidade dos Geradores – S4 (Normalizado)', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_ylim(-0.5, 0.5)
    save(fig, 'plot5_3_radar_generator_quality.png')

add_extra_plots()
print('All extra plots done.')
