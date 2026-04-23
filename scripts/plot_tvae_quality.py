import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import load_compas

def plot_tvae_quality():
    print("1. Carregando dados...")
    real_data = load_compas()
    
    # Pegamos a seed 42 do TVAE (Cenário S4) para demonstração visual
    tvae_path = 'synthetic_data/compas/S4/TVAE/seed_42.csv'
    synth_data = pd.read_csv(tvae_path)
    synth_data = synth_data[real_data.columns] # Garantir alinhamento
    
    out_dir = 'plots/paper'
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')
    
    # ─── 1. COMPARAÇÃO DE DISTRIBUIÇÕES ─────────────────────────────────────
    print("2. Gerando gráficos de distribuição (KDE)...")
    cols_to_plot = ['age', 'priors_count']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, col in enumerate(cols_to_plot):
        sns.kdeplot(real_data[col], fill=True, label='Real (Original)', color='#457b9d', ax=axes[i], alpha=0.5, linewidth=2)
        sns.kdeplot(synth_data[col], fill=True, label='Sintético (TVAE)', color='#e63946', ax=axes[i], alpha=0.5, linewidth=2, linestyle='--')
        
        titulo = 'Idade' if col == 'age' else 'Contagem de Crimes Anteriores'
        axes[i].set_title(f'Distribuição: {titulo}', fontweight='bold')
        axes[i].set_ylabel('Densidade')
        axes[i].legend()
    
    fig.suptitle('Qualidade do TVAE: Fidelidade das Distribuições (Marginais)', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    dist_path = os.path.join(out_dir, 'plot_tvae_distributions.png')
    plt.savefig(dist_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    # ─── 2. COMPARAÇÃO DE CORRELAÇÕES ───────────────────────────────────────
    print("3. Gerando heatmaps de correlação lado a lado...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    corr_real = real_data.corr()
    corr_synth = synth_data.corr()
    
    # Matriz Real
    sns.heatmap(corr_real, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                annot=False, square=True, ax=axes[0], cbar_kws={"shrink": .8})
    axes[0].set_title('Correlações - Dados Reais', fontweight='bold', fontsize=13)
    
    # Matriz Sintética
    sns.heatmap(corr_synth, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                annot=False, square=True, ax=axes[1], cbar_kws={"shrink": .8})
    axes[1].set_title('Correlações - Sintético (TVAE)', fontweight='bold', fontsize=13)
    
    fig.suptitle('Qualidade do TVAE: Manutenção da Estrutura de Correlação', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    corr_path = os.path.join(out_dir, 'plot_tvae_correlations.png')
    plt.savefig(corr_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nPronto! Gráficos salvos em:\n - {dist_path}\n - {corr_path}")

if __name__ == '__main__':
    plot_tvae_quality()
