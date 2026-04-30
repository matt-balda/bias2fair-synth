import pandas as pd
from utils.data_loader import load_compas

def check_demographic_parity(df, target_col='two_year_recid', sensitive_col='race', priv_val=1, unpriv_val=0):
    """Calculates and displays demographic parity metrics for a given dataset."""
    counts = df.groupby(sensitive_col).size()
    pos_counts = df[df[target_col] == 1].groupby(sensitive_col).size()
    
    p_priv = pos_counts.get(priv_val, 0) / counts.get(priv_val, 1)
    p_unpriv = pos_counts.get(unpriv_val, 0) / counts.get(unpriv_val, 1)
    diff = p_unpriv - p_priv
    
    print("\n[Demographic Parity]")
    print(f" P(Y=1 | Privileged={priv_val}):   {p_priv:.4f} ({pos_counts.get(priv_val, 0)}/{counts.get(priv_val, 0)})")
    print(f" P(Y=1 | Unprivileged={unpriv_val}): {p_unpriv:.4f} ({pos_counts.get(unpriv_val, 0)}/{counts.get(unpriv_val, 0)})")
    print(f" Difference (Unpriv-Priv): {diff:.4f} | Gap: {abs(diff):.4f}\n")
    
    return {'p_priv': p_priv, 'p_unpriv': p_unpriv, 'diff': diff, 'gap': abs(diff)}

if __name__ == '__main__':
    print("Loading COMPAS dataset...")
    check_demographic_parity(load_compas())
