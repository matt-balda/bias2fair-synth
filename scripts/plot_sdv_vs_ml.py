import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality

# Import local modules
from utils.data_loader import load_compas
from utils.diffusion_wrapper import TabDDPMWrapper

OUT_DIR = 'plots/paper'
os.makedirs(OUT_DIR, exist_ok=True)

def run():
    print("1. Obtendo F1-Score médio do CSV existente...")
    df_res = pd.read_csv('outputs/compas_results.csv')
    
    # Vamos usar os cenários sintéticos puros (S3 ou S4) para avaliar o poder base do gerador
    synth_df = df_res[df_res['scenario'] == 'S4']
    f1_scores = synth_df.groupby('generator')['f1'].mean().reset_index()
    f1_scores.rename(columns={'f1': 'ML_Efficacy_F1'}, inplace=True)
    
    print("2. Carregando dados originais do COMPAS...")
    real_data = load_compas()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    gens = ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']
    sdv_scores = []
    
    print("3. Gerando amostras sintéticas e calculando SDV Quality Score (KS-Test/CS-Test)...")
    for gen_name in gens:
        print(f"   -> Treinando e avaliando {gen_name}...")
        try:
            if gen_name == 'GaussianCopula':
                model = GaussianCopulaSynthesizer(metadata)
            elif gen_name == 'CTGAN':
                model = CTGANSynthesizer(metadata, epochs=50, enforce_rounding=False)
            elif gen_name == 'TVAE':
                model = TVAESynthesizer(metadata, epochs=50)
            elif gen_name == 'TabDDPM':
                # Usando menos passos apenas para extrair a métrica estatística em tempo razoável
                model = TabDDPMWrapper(metadata, device='cuda', steps=250)
            
            # Ajuste e geração
            model.fit(real_data)
            synth_data = model.sample(len(real_data))
            
            # Avaliação Oficial do SDV
            quality_report = evaluate_quality(real_data, synth_data, metadata, verbose=False)
            score = quality_report.get_score()
            print(f"      SDV Score para {gen_name}: {score:.4f}")
            sdv_scores.append({'generator': gen_name, 'SDV_Statistical_Similarity': score})
            
        except Exception as e:
            print(f"      Erro no {gen_name}: {e}")
            sdv_scores.append({'generator': gen_name, 'SDV_Statistical_Similarity': np.nan})
            
    df_sdv = pd.DataFrame(sdv_scores)
    
    print("4. Mesclando resultados e gerando gráfico...")
    merged = pd.merge(f1_scores, df_sdv, on='generator')
    
    sns.set_theme(style='whitegrid', font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    PALETTE = {'GaussianCopula':'#4cc9f0','CTGAN':'#f72585','TVAE':'#7209b7','TabDDPM':'#3a0ca3'}
    
    for _, row in merged.iterrows():
        ax.scatter(row['SDV_Statistical_Similarity'], row['ML_Efficacy_F1'], 
                   color=PALETTE.get(row['generator'], 'gray'), s=200, 
                   edgecolors='white', linewidths=1.5, label=row['generator'])
        # Anotar o nome ao lado do ponto
        ax.text(row['SDV_Statistical_Similarity'] + 0.005, row['ML_Efficacy_F1'], 
                row['generator'], fontsize=11, weight='bold')

    # Quadrantes teóricos (ideal = Topo Direito)
    ax.axvline(merged['SDV_Statistical_Similarity'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.axhline(merged['ML_Efficacy_F1'].mean(), color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Trade-off Real: Similaridade Estatística (SDV) vs Utilidade de ML (F1)', fontsize=13, weight='bold')
    ax.set_xlabel('SDV Quality Score (Fidelidade Distribuicional -> Maior é melhor)')
    ax.set_ylabel('ML Efficacy: F1-Score Médio (Utilidade -> Maior é melhor)')
    
    ax.legend(title='Gerador Sintético', loc='lower right')
    
    out_path = os.path.join(OUT_DIR, 'plot_sdv_vs_ml_scatter.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"\n✅ Gráfico gerado com sucesso em: {out_path}")

if __name__ == '__main__':
    run()
