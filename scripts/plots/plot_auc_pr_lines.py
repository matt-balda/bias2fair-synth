import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_auc_pr_lines():
    df = pd.read_csv('outputs/compas_results.csv')
    
    # Publication quality settings
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x='scenario', 
        y='auc_pr',
        hue='model',
        style='model',
        markers=['o', 's', 'D'],
        dashes=False,
        linewidth=3,
        markersize=10,
        errorbar=None, 
        palette='Dark2'
    )
    
    # Dynamically find the min and max means to set tight Y-limits
    means = df.groupby(['scenario', 'model'])['auc_pr'].mean()
    min_mean = np.floor(means.min() * 100) / 100
    max_mean = np.ceil(means.max() * 100) / 100
    
    plt.ylim(min_mean - 0.01, max_mean + 0.01)
    plt.yticks(np.arange(min_mean - 0.01, max_mean + 0.02, 0.01))
    
    plt.title('Impacto das Estratégias de Mitigação na AUC-PR', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Cenários Experimentais (S1 ao S6)', fontsize=14, fontweight='bold')
    plt.ylabel('AUC-PR Média', fontsize=14, fontweight='bold')
    
    sns.despine()
    
    plt.legend(title='Modelo Preditivo', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    
    out_dir = 'plots/compas'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'auc_pr_lines.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_auc_pr_lines()
