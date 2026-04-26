import os
import urllib.request
import pandas as pd
import numpy as np

ADULT_CONFIG = {
    'target':    'income',
    'sensitive': 'sex',
    # 0 = Female (unprivileged), 1 = Male (privileged)
}

_ADULT_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
_ADULT_URL_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]


def _download_adult(path: str) -> None:
    """Download and concatenate Adult train + test splits into a single CSV."""
    print(f"  Downloading Adult dataset to {path}...")
    frames = []
    for url, skip in [(_ADULT_URL, 0), (_ADULT_URL_TEST, 1)]:
        with urllib.request.urlopen(url) as resp:
            lines = resp.read().decode('utf-8').splitlines()
        rows = [l.strip().rstrip('.') for l in lines[skip:] if l.strip()]
        from io import StringIO
        df = pd.read_csv(StringIO('\n'.join(rows)), header=None,
                         names=_COLUMNS, na_values=' ?', skipinitialspace=True)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    combined.to_csv(path, index=False)
    print(f"  Saved: {combined.shape[0]} rows.")


def load_adult(path: str = "data/adult.csv") -> pd.DataFrame:
    """
    Loads and preprocesses the UCI Adult (Census Income) dataset.
    Sensitive attribute: sex (0=Female, 1=Male)
    Target: income (0=<=50K, 1=>50K)
    """
    if not os.path.exists(path):
        _download_adult(path)

    df = pd.read_csv(path, na_values=' ?', skipinitialspace=True)
    # Make sure column names are normalised (file may already have header)
    if list(df.columns) == _COLUMNS:
        pass  # already has header
    else:
        df.columns = _COLUMNS

    df = df.dropna()

    # ── Target ────────────────────────────────────────────────────────────────
    df['income'] = (
        df['income'].astype(str).str.strip().str.rstrip('.')
        .map({'<=50K': 0, '>50K': 1})
    )

    # ── Sensitive: sex ────────────────────────────────────────────────────────
    df['sex'] = df['sex'].astype(str).str.strip().map({'Male': 1, 'Female': 0})

    # ── Drop non-predictive / other sensitive cols ────────────────────────────
    # fnlwgt: survey weight (not a meaningful feature)
    # race / native_country: alternative sensitive attrs — excluded to avoid
    #   conflation with the chosen sensitive attribute (sex)
    df = df.drop(columns=['fnlwgt', 'race', 'native_country'], errors='ignore')

    # ── Encode remaining categoricals ─────────────────────────────────────────
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df.dropna().reset_index(drop=True)


if __name__ == "__main__":
    data = load_adult()
    print(f"Dataset loaded: {data.shape}")
    print(f"Target distribution:\n{data['income'].value_counts(normalize=True)}")
    print(f"Sex distribution:\n{data['sex'].value_counts(normalize=True)}")
