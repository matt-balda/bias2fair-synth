import pandas as pd

def analyze_features():
    # Caminho do dataset COMPAS Bruto
    path = "data/compas-scores-two-years.csv"
    
    print("\n" + "="*80)
    print("ANÁLISE DE FEATURE SELECTION - COMPAS DATASET (PROPUBLICA)")
    print("="*80 + "\n")
    
    try:
        df_raw = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Erro: Arquivo bruto não encontrado em {path}.")
        return

    # 1. Obter todas as features originais
    all_raw_cols = set(df_raw.columns)
    
    # 2. Definir as features aprovadas na nossa modelagem (Baseline Paper)
    kept_cols = {
        'sex', 'age', 'age_cat', 'race', 
        'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
        'priors_count', 'c_charge_degree', 'two_year_recid'
    }

    # 3. Interseção matemática 
    dropped_cols = all_raw_cols - kept_cols

    print(f"Total de colunas brutas     : {len(all_raw_cols)}")
    print(f"Total de colunas selecionadas: {len(kept_cols)}")
    print(f"Total de colunas descartadas : {len(dropped_cols)}\n")

    # 4. Mapeamento heurístico (Agrupando Dropadas por Motivo Acadêmico)
    groups = {
        "Vazamento Multi-Alvo (Data Leakage)": [
            'is_recid', 'is_violent_recid', 'violent_recid', 
            'r_case_number', 'r_charge_degree', 'r_days_from_arrest', 
            'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
            'vr_case_number', 'vr_charge_degree', 'vr_offense_date', 'vr_charge_desc'
        ],
        "Scores do Algoritmo proprietário COMPAS (Target Leaking)": [
            'decile_score', 'score_text', 'type_of_assessment', 'decile_score.1', 
            'screening_date', 'v_type_of_assessment', 'v_decile_score', 
            'v_score_text', 'v_screening_date'
        ],
        "Dados Pessoais (ID/Nomes) e Ruídos de Séries Temporais Brutas": [
            'id', 'name', 'first', 'last', 'dob', 'c_case_number', 
            'compas_screening_date', 'c_jail_in', 'c_jail_out', 'c_offense_date', 
            'c_arrest_date', 'days_b_screening_arrest', 'c_days_from_compas', 
            'in_custody', 'out_custody'
        ],
        "Análise de Sobrevivência (Não se aplica a classificador binário)": [
            'start', 'end', 'event'
        ],
        "Textos Livres (High Cardinality) ou Duplicatas Estruturais": [
            'c_charge_desc', 'priors_count.1'
        ]
    }

    print("-" * 60)
    print(" AS 10 FEATURES MANTIDAS NAS EXPERIÊNCIAS (INPUT DA IA)")
    print("-" * 60)
    print(sorted(list(kept_cols)))
    print("\n")

    print("-" * 60)
    print(" ANÁLISE SISTEMÁTICA DAS 43 FEATURES DROPADAS")
    print("-" * 60)
    
    # Contabilização e agrupamento em tempo real
    unmapped = list(dropped_cols)
    
    for group_name, cols_in_group in groups.items():
        print(f"\n>> MOTIVO: {group_name}")
        intersect = []
        for col in cols_in_group:
            if col in dropped_cols:
                intersect.append(col)
                if col in unmapped:
                    unmapped.remove(col)
        
        print(f"   => Excluídas ({len(intersect)} colunas):")
        print(f"      {intersect}")

    if unmapped:
        print("\n>> OUTROS / NÃO MAPEADOS:")
        print(f"   => Excluídas ({len(unmapped)} colunas):")
        print(f"      {unmapped}")

    print("\n" + "="*80)
    print("CONCLUSÃO METODOLÓGICA GERADA:")
    print("A exclusão em massa blinda o classificador contra Future Leakage e")
    print("garante um benchmark comparativo perfeito com os pipelines CTGAN da literatura.")
    print("="*80 + "\n")

if __name__ == '__main__':
    analyze_features()
