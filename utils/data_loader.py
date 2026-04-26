"""
Unified data loader — factory entry point.

Usage:
    from utils.data_loader import load_dataset, DATASET_CONFIGS
    data = load_dataset('adult')

Individual loaders are also importable directly:
    from utils.data_loader import load_compas, load_adult, load_diabetes
"""

from utils.data_loader_compas   import load_compas,   COMPAS_CONFIG
from utils.data_loader_adult    import load_adult,    ADULT_CONFIG
from utils.data_loader_diabetes import load_diabetes, DIABETES_CONFIG

# Mapping: dataset name → {loader, target, sensitive}
DATASET_CONFIGS = {
    'compas': {
        'loader':    load_compas,
        'target':    COMPAS_CONFIG['target'],
        'sensitive': COMPAS_CONFIG['sensitive'],
    },
    'adult': {
        'loader':    load_adult,
        'target':    ADULT_CONFIG['target'],
        'sensitive': ADULT_CONFIG['sensitive'],
    },
    'diabetes': {
        'loader':    load_diabetes,
        'target':    DIABETES_CONFIG['target'],
        'sensitive': DIABETES_CONFIG['sensitive'],
    },
}


def load_dataset(name: str, **kwargs):
    """
    Load a dataset by name.

    Args:
        name:   One of 'compas', 'adult', 'diabetes'.
        **kwargs: Forwarded to the individual loader (e.g. path='data/...').

    Returns:
        pd.DataFrame with binary target and binary sensitive attribute columns.
    """
    name = name.lower()
    if name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[name]['loader'](**kwargs)


__all__ = [
    'load_dataset', 'DATASET_CONFIGS',
    'load_compas', 'load_adult', 'load_diabetes',
    'COMPAS_CONFIG', 'ADULT_CONFIG', 'DIABETES_CONFIG',
]
