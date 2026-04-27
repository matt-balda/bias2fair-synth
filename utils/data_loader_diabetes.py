import os
import pandas as pd
import numpy as np

DIABETES_CONFIG = {
    'target':    'readmitted',
    'sensitive': 'race',
    # 0 = AfricanAmerican (unprivileged), 1 = Caucasian (privileged)
}

_CACHE_PATH = "data/diabetes_130.csv"

_MED_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone',
]
_MED_MAP = {'No': 0, 'Down': 1, 'Steady': 2, 'Up': 3}

_AGE_MAP = {
    '[0-10)': 5,  '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95,
}

# High-missingness / identifier columns to drop before modelling
_DROP_COLS = [
    'encounter_id', 'patient_nbr',
    'weight', 'payer_code', 'medical_specialty',
    'examide', 'citoglipton',
]


def _fetch_and_cache(path: str) -> pd.DataFrame:
    """Download Diabetes 130-US Hospitals dataset via ucimlrepo (id=296) and cache."""
    from ucimlrepo import fetch_ucirepo
    print(f"  Downloading Diabetes 130-US dataset via ucimlrepo (id=296)...")
    dataset = fetch_ucirepo(id=296)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Cached to {path}  ({len(df)} rows).")
    return df


def load_diabetes(path: str = _CACHE_PATH) -> pd.DataFrame:
    """
    Loads and preprocesses the Diabetes 130-US Hospitals dataset (UCI id=296, 1999-2008).
    Downloads via ucimlrepo on first call; subsequent calls use the local cache.

    Sensitive attribute : race       (0=AfricanAmerican, 1=Caucasian)
    Target              : readmitted (0=NO, 1=readmitted within <30 or >30 days)
    """
    if os.path.exists(path):
        df = pd.read_csv(path, low_memory=False)
    else:
        df = _fetch_and_cache(path)

    df = df.copy()

    # ── Sensitive: keep only African-American vs Caucasian ───────────────────
    df = df[df['race'].isin(['AfricanAmerican', 'Caucasian'])]
    df['race'] = df['race'].map({'Caucasian': 1, 'AfricanAmerican': 0})

    # ── Target: binary readmission ───────────────────────────────────────────
    df['readmitted'] = df['readmitted'].map({'NO': 0, '<30': 1, '>30': 1})

    # ── Drop irrelevant / high-missingness columns ────────────────────────────
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # ── Gender ───────────────────────────────────────────────────────────────
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Unknown/Invalid': np.nan})

    # ── Age brackets → ordinal midpoints ─────────────────────────────────────
    if 'age' in df.columns:
        df['age'] = df['age'].map(_AGE_MAP)

    # ── Medication columns → ordinal (0–3) ────────────────────────────────────
    for col in _MED_COLS:
        if col in df.columns:
            df[col] = df[col].map(_MED_MAP)

    # ── Binary flags ──────────────────────────────────────────────────────────
    if 'change' in df.columns:
        df['change'] = df['change'].map({'No': 0, 'Ch': 1})
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1})

    # ── Lab result ordinals ───────────────────────────────────────────────────
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum'] = df['max_glu_serum'].fillna('None').map(
            {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
        )
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = df['A1Cresult'].fillna('None').map(
            {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
        )

    # ── Diagnosis codes — drop (high cardinality, leakage risk) ──────────────
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'], errors='ignore')

    return df.dropna().reset_index(drop=True)


if __name__ == "__main__":
    data = load_diabetes()
    print(f"Dataset loaded: {data.shape}")
    print(f"Target distribution:\n{data['readmitted'].value_counts(normalize=True)}")
    print(f"Race distribution:\n{data['race'].value_counts(normalize=True)}")
