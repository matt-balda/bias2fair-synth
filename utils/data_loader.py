import pandas as pd
import numpy as np

def load_compas(path="data/compas-scores-two-years.csv"):
    """
    Loads and cleans the COMPAS dataset as per standard ProPublica guidelines.
    """
    df = pd.read_csv(path)
    
    # Standard filtering criteria
    df = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['score_text'] != 'N/A')
    ]
    
    # Selection of relevant columns
    cols = [
        'sex', 'age', 'age_cat', 'race', 
        'juv_fel_count', 'juv_misd_count', 'juv_other_count', 
        'priors_count', 'c_charge_degree', 'two_year_recid'
    ]
    df = df[cols]
    
    # Binary race: African-American vs Caucasian
    df = df[df['race'].isin(['African-American', 'Caucasian'])]
    
    # Binary encoding
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    df['race'] = df['race'].map({'Caucasian': 1, 'African-American': 0}) # 1: Privileged, 0: Unprivileged
    df['c_charge_degree'] = df['c_charge_degree'].map({'F': 1, 'M': 0})
    
    # One-hot encoding for categorical cat categories if needed, but COMPAS is simple
    df = pd.get_dummies(df, columns=['age_cat'], drop_first=True)
    
    return df.dropna()

if __name__ == "__main__":
    data = load_compas()
    print(f"Dataset loaded: {data.shape}")
    print(f"Target distribution:\n{data['two_year_recid'].value_counts(normalize=True)}")
    print(f"Race distribution:\n{data['race'].value_counts(normalize=True)}")
