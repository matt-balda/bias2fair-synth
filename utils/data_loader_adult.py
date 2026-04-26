import os
import pandas as pd
import numpy as np

ADULT_CONFIG = {
    'target':    'income',
    'sensitive': 'sex',
    # 0 = Female (unprivileged), 1 = Male (privileged)
}

_CACHE_PATH = "data/adult.csv"


def _fetch_and_cache(path: str) -> pd.DataFrame:
    """Download Adult dataset via ucimlrepo (id=2) and cache as CSV."""
    from ucimlrepo import fetch_ucirepo
    print(f"  Downloading Adult dataset via ucimlrepo (id=2)...")
    dataset = fetch_ucirepo(id=2)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Cached to {path}  ({len(df)} rows).")
    return df


def load_adult(path: str = _CACHE_PATH) -> pd.DataFrame:
    """
    Loads and preprocesses the UCI Adult (Census Income) dataset.
    Downloads via ucimlrepo (id=2) on first call; subsequent calls use the local cache.

    Sensitive attribute : sex    (0=Female, 1=Male)
    Target              : income (0=<=50K, 1=>50K)
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = _fetch_and_cache(path)

    df = df.copy()

    # ── Target ────────────────────────────────────────────────────────────────
    # ucimlrepo returns column 'income' with values '<=50K' / '>50K'
    income_col = next((c for c in df.columns if 'income' in c.lower()), 'income')
    df['income'] = (
        df[income_col].astype(str).str.strip().str.rstrip('.')
        .map({'<=50K': 0, '>50K': 1})
    )
    if income_col != 'income':
        df = df.drop(columns=[income_col])

    # ── Sensitive: sex ────────────────────────────────────────────────────────
    df['sex'] = df['sex'].astype(str).str.strip().map({'Male': 1, 'Female': 0})

    # ── Drop non-predictive / alternate sensitive columns ────────────────────
    # fnlwgt: survey sampling weight — not a meaningful feature
    # race / native-country: excluded to avoid conflation with sex
    drop_cols = ['fnlwgt', 'race', 'native-country', 'native_country']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Encode remaining categoricals ─────────────────────────────────────────
    cat_cols = [c for c in df.columns
                if df[c].dtype == object and c not in ('income', 'sex')]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    return df.dropna().reset_index(drop=True)


if __name__ == "__main__":
    data = load_adult()
    print(f"Dataset loaded: {data.shape}")
    print(f"Target distribution:\n{data['income'].value_counts(normalize=True)}")
    print(f"Sex distribution:\n{data['sex'].value_counts(normalize=True)}")
