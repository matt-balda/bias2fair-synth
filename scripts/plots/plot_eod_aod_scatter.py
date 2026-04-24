import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def plot_eod_aod():
    df = pd.read_csv('outputs/compas_results.csv')
    
    sns.set_theme(style='whitegrid', font_scale=1.1)
    MODEL_MARKERS = {'LogisticRegression':'o','SVM':'s','CatBoost':'D'}
    sg_colors = {'S1':'#e63946','S2':'#457b9d','S3':'#2a9d8f','S4':'#e9c46a','S5':'#f4a261','S6':'#6a4c93'}
    
    means = df.groupby(['scenario','generator','model'])[['f1','equal_opportunity_difference','average_absolute_odds_difference']].mean().reset_index()
    
    def get_scenario_group(s):
        if s=='S1': return 'S1'
        if s.startswith('S2'): return 'S2'
        if s.startswith('S3'): return 'S3'
        if s=='S4': return 'S4'
        if s=='S5': return 'S5'
        if s=='S6': return 'S6'
        return s
    
    means['sg'] = means['scenario'].apply(get_scenario_group)
    
    legend_sc = [mpatches.Patch(color=c, label=sg) for sg, c in sg_colors.items()]
    legend_mod = [plt.Line2D([0],[0], marker=m, color='gray', linestyle='None', markersize=8, label=mod)
                  for mod, m in MODEL_MARKERS.items()]
    
    OUT = 'plots/paper'
    os.makedirs(OUT, exist_ok=True)
    
    # ── PLOT 1: Scatter F1 vs EOD ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvline(0.0, color='green', linestyle='--', lw=1.5, alpha=0.7, label='EOD ideal=0')
    ax.axvspan(-0.1, 0.1, alpha=0.08, color='green')
    for sg, grp in means.groupby('sg'):
        for model, mgrp in grp.groupby('model'):
            ax.scatter(mgrp['equal_opportunity_difference'], mgrp['f1'],
                       color=sg_colors.get(sg,'gray'),
                       marker=MODEL_MARKERS.get(model,'o'),
                       s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.legend(handles=legend_sc+legend_mod+[plt.Line2D([0],[0], color='green', linestyle='--', label='EOD=0')],
              loc='upper right', fontsize=8, ncol=2)
    ax.set_xlabel('Equal Opportunity Difference (EOD)', fontsize=11)
    ax.set_ylabel('F1-Score → Utilidade', fontsize=11)
    ax.set_title('Trade-off F1 × EOD por Cenário', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_path = os.path.join(OUT, 'plot2_3_scatter_f1_eod.png')
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path}")
    
    # ── PLOT 2: Scatter F1 vs AOD ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvline(0.0, color='green', linestyle='--', lw=1.5, alpha=0.7, label='AOD ideal=0')
    ax.axvspan(-0.1, 0.1, alpha=0.08, color='green') 
    for sg, grp in means.groupby('sg'):
        for model, mgrp in grp.groupby('model'):
            ax.scatter(mgrp['average_absolute_odds_difference'], mgrp['f1'],
                       color=sg_colors.get(sg,'gray'),
                       marker=MODEL_MARKERS.get(model,'o'),
                       s=70, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.legend(handles=legend_sc+legend_mod+[plt.Line2D([0],[0], color='green', linestyle='--', label='AOD=0')],
              loc='upper right', fontsize=8, ncol=2)
    ax.set_xlabel('Average Absolute Odds Difference (AOD)', fontsize=11)
    ax.set_ylabel('F1-Score → Utilidade', fontsize=11)
    ax.set_title('Trade-off F1 × AOD por Cenário', fontsize=13, fontweight='bold')
    fig.tight_layout()
    save_path = os.path.join(OUT, 'plot2_4_scatter_f1_aod.png')
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_eod_aod()
