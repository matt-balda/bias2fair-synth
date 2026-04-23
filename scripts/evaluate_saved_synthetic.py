import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

from utils.data_loader import load_compas

def run_evaluation():
    print("1. Carregando dados originais (Real)...")
    real_data = load_compas()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    # Vamos usar o S4 porque ele é a augmentação pura da minoria
    scenario = 'S4'
    base_dir = f'synthetic_data/compas/{scenario}'
    
    generators = ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']
    seeds = [42, 123, 999, 2024, 7, 88, 101, 13, 555, 777]
    
    results = []
    
    print("\n2. Avaliando datasets sintéticos (10 seeds por gerador)...")
    for gen in generators:
        print(f"\n--- Gerador: {gen} ---")
        scores = []
        for seed in seeds:
            file_path = os.path.join(base_dir, gen, f'seed_{seed}.csv')
            if os.path.exists(file_path):
                synth_data = pd.read_csv(file_path)
                
                # Para uma avaliação justa, cortamos o dado sintético para ter
                # o mesmo formato do dado real (removendo colunas extras se houver)
                synth_data = synth_data[real_data.columns]
                
                # SDV avalia KS-Test e CS-Test
                report = evaluate_quality(real_data, synth_data, metadata, verbose=False)
                score = report.get_score()
                scores.append(score)
                print(f"  Seed {seed} -> SDV Quality Score: {score:.4f}")
            else:
                print(f"  Aviso: Seed {seed} não encontrado em {file_path}")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            results.append({'Gerador': gen, 'SDV_Score_Medio': avg_score})
            print(f"  >>> MÉDIA FINAL ({gen}): {avg_score:.4f}")
            
    df_res = pd.DataFrame(results)
    os.makedirs('reports', exist_ok=True)
    out_csv = 'reports/synthetic_quality_sdv_all_seeds.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\nAvaliação concluída com sucesso! Resumo salvo em: {out_csv}")

if __name__ == '__main__':
    run_evaluation()
