import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pr_change():
    df = pd.read_csv('outputs/compas_results.csv')
    
    # Configurações de publicação
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # Criar uma figura com 2 painéis (lado a lado)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==========================
    # Painel 1: RECALL
    # ==========================
    sns.lineplot(
        ax=axes[0],
        data=df,
        x='scenario', 
        y='recall',
        hue='model',
        style='model',
        markers=['o', 's', 'D'],
        dashes=False,
        linewidth=3,
        markersize=10,
        errorbar=None, 
        palette='Dark2',
        legend=False # Esconde a legenda no primeiro gráfico
    )
    axes[0].set_title('Evolução do Recall (Revocação)', fontsize=16, fontweight='bold', pad=15)
    axes[0].set_ylabel('Recall Médio', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Cenários (S1 ao S6)', fontsize=14, fontweight='bold')
    
    # Ajuste dinâmico fino para Recall
    r_means = df.groupby(['scenario', 'model'])['recall'].mean()
    r_min = np.floor(r_means.min() * 100) / 100
    r_max = np.ceil(r_means.max() * 100) / 100
    axes[0].set_ylim(r_min - 0.02, r_max + 0.02)
    
    # ==========================
    # Painel 2: PRECISION
    # ==========================
    sns.lineplot(
        ax=axes[1],
        data=df,
        x='scenario', 
        y='precision',
        hue='model',
        style='model',
        markers=['o', 's', 'D'],
        dashes=False,
        linewidth=3,
        markersize=10,
        errorbar=None, 
        palette='Dark2'
    )
    axes[1].set_title('Evolução da Precision (Precisão)', fontsize=16, fontweight='bold', pad=15)
    axes[1].set_ylabel('Precisão Média', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cenários (S1 ao S6)', fontsize=14, fontweight='bold')
    
    # Ajuste dinâmico fino para Precision
    p_means = df.groupby(['scenario', 'model'])['precision'].mean()
    p_min = np.floor(p_means.min() * 100) / 100
    p_max = np.ceil(p_means.max() * 100) / 100
    axes[1].set_ylim(p_min - 0.02, p_max + 0.02)
    
    sns.despine()
    
    # Ajustar legenda apenas no lado direito
    axes[1].legend(title='Modelo Preditivo', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    
    plt.suptitle('Análise Detalhada: Mudança de Recall e Precision por Cenário', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    out_dir = 'plots/compas'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'precision_recall_change.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_pr_change()
