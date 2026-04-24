import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_lines():
    df = pd.read_csv('outputs/compas_results.csv')
    
    # Publication quality settings
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    
    # Voltando para o tamanho horizontal de artigo
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=df,
        x='scenario', 
        y='accuracy',
        hue='model',
        style='model',
        markers=['o', 's', 'D'],
        dashes=False,
        linewidth=3,
        markersize=10,
        errorbar=None, 
        palette='Dark2'
    )
    
    # Dando "zoom" no eixo Y para o intervalo numérico exato onde os dados ocorrem (fino/granulado)
    # Isso vai "espaçar" as linhas visualmente
    plt.ylim(0.63, 0.69)
    plt.yticks(np.arange(0.63, 0.70, 0.01)) # Ticks a cada 0.01
    
    plt.title('Impacto das Estratégias de Mitigação na Acurácia', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Cenários Experimentais (S1 ao S6)', fontsize=14, fontweight='bold')
    plt.ylabel('Acurácia Média', fontsize=14, fontweight='bold')
    
    sns.despine()
    
    plt.legend(title='Modelo Preditivo', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    plt.tight_layout()
    
    out_dir = 'plots/compas'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'accuracy_lines.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_accuracy_lines()
