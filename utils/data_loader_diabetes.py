import os
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np

DIABETES_CONFIG = {
    'target':    'readmitted',
    'sensitive': 'race',
    # 0 = AfricanAmerican (unprivileged), 1 = Caucasian (privileged)
}

_DIABETES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00296/dataset_diabetes.zip"
)

# Columns to drop before modelling
_DROP_COLS = [
    'encounter_id', 'patient_nbr',
    # High missingness
    'weight', 'payer_code', 'medical_specialty',
    # Constant / near-zero variance in cleaned subset
    'examide', 'citoglipton',
]

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


def _download_diabetes(path: str) -> None:
    """Download the Diabetes 130-US Hospitals ZIP and extract diabetic_data.csv."""
    print(f"  Downloading Diabetes 130-US dataset to {path}...")
    with urllib.request.urlopen(_DIABETES_URL) as resp:
        content = resp.read()
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = next(n for n in zf.namelist() if 'diabetic_data' in n)
        raw = zf.read(csv_name)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        f.write(raw)
    print(f"  Saved to {path}.")


def load_diabetes(path: str = "data/diabetes_130.csv") -> pd.DataFrame:
    """
    Loads and preprocesses the Diabetes 130-US Hospitals dataset (UCI, 1999-2008).
    Sensitive attribute: race  (0=AfricanAmerican, 1=Caucasian)
    Target:             readmitted (0=NO, 1=readmitted within <30 or >30 days)
    """
    if not os.path.exists(path):
        _download_diabetes(path)

    df = pd.read_csv(path, na_values='?', low_memory=False)

    # ── Sensitive: keep only African-American vs Caucasian ───────────────────
    df = df[df['race'].isin(['AfricanAmerican', 'Caucasian'])].copy()
    df['race'] = df['race'].map({'Caucasian': 1, 'AfricanAmerican': 0})

    # ── Target: binary readmission ───────────────────────────────────────────
    df['readmitted'] = df['readmitted'].map({'NO': 0, '<30': 1, '>30': 1})

    # ── Drop irrelevant / high-missingness columns ────────────────────────────
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # ── Gender ───────────────────────────────────────────────────────────────
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Unknown/Invalid': np.nan})

    # ── Age brackets → ordinal ────────────────────────────────────────────────
    df['age'] = df['age'].map(_AGE_MAP)

    # ── Medication columns → ordinal (0-3) ───────────────────────────────────
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
        df['max_glu_serum'] = df['max_glu_serum'].map(
            {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
        )
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = df['A1Cresult'].map(
            {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
        )

    # ── Diagnosis codes — drop (high cardinality, leakage risk) ──────────────
    df = df.drop(columns=['diag_1', 'diag_2', 'diag_3'], errors='ignore')

    # ── Admission / discharge type → keep as numeric (already int) ───────────
    # admission_type_id, discharge_disposition_id, admission_source_id are
    # integer-coded categoricals; treat as ordinal numerics for simplicity.

    return df.dropna().reset_index(drop=True)


if __name__ == "__main__":
    data = load_diabetes()
    print(f"Dataset loaded: {data.shape}")
    print(f"Target distribution:\n{data['readmitted'].value_counts(normalize=True)}")
    print(f"Race distribution:\n{data['race'].value_counts(normalize=True)}")
