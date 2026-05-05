import warnings
warnings.filterwarnings('ignore')

import sys, os
# Adicionar o diretório raiz do projeto ao PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.data_loader import DATASET_CONFIGS

# ── Paleta ───────────────────────────────────────────────────────────────────
BG     = '#ffffff'
PANEL  = '#ffffff'
TEXT   = '#111111'
SUBTEXT = '#555555'
GRID   = '#e0e0e0'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': PANEL,
    'axes.edgecolor':  GRID, 'axes.labelcolor': TEXT,
    'xtick.color':  SUBTEXT, 'ytick.color':    SUBTEXT,
    'text.color':      TEXT, 'grid.color':      GRID,
    'grid.linewidth':  0.6,  'font.family': 'DejaVu Sans',
})

def main():
    datasets = ['compas', 'adult', 'diabetes']
    
    out_dir = os.path.join(PROJECT_ROOT, 'plots', 'individual_crosstabs')
    os.makedirs(out_dir, exist_ok=True)
    
    for dataset_name in datasets:
        print(f"Processando {dataset_name.upper()}...")
        cfg = DATASET_CONFIGS[dataset_name]
        data = cfg['loader']()
        TARGET    = cfg['target']
        SENSITIVE = cfg['sensitive']
        
        y = data[TARGET].values
        s = data[SENSITIVE].values
        N = len(data)
        
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
        
        cross = pd.crosstab(s, y)
        matrix = cross.values
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Class 0 (-)', 'Class 1 (+)'], color=TEXT)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Group 0', 'Group 1'], color=TEXT)
        
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                col_txt = 'white' if val > matrix.max() * 0.55 else TEXT
                ax.text(j, i, f'{val}\n({val/N*100:.1f}%)', ha='center', va='center', fontsize=16, color=col_txt, fontweight='bold')
        
        plt.colorbar(im, ax=ax, pad=0.03)
        
        plt.tight_layout()
        
        out_png = os.path.join(out_dir, f'crosstab_{dataset_name}.png')
        out_eps = os.path.join(out_dir, f'crosstab_{dataset_name}.eps')
        
        fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor=BG)
        fig.savefig(out_eps, format='eps', bbox_inches='tight', facecolor=BG)
        plt.close(fig)
        
    print(f"Salvos em {out_dir}/")

if __name__ == "__main__":
    main()
