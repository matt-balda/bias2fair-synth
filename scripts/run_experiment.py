import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from aif360.sklearn.preprocessing import Reweighing
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from tqdm import tqdm
import logging

# Silence everything
logging.disable(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.data_loader import load_compas
from utils.metrics import calculate_metrics
from scripts.seed_manager import get_fixed_seeds, set_seed
from utils.diffusion_wrapper import TabDDPMWrapper

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TARGET    = 'two_year_recid'
SENSITIVE = 'race'
DATASET   = 'compas'

SCENARIOS = ['S1', 'S2', 'S3', 'S4', 'S5']

# S1/S2 don't iterate over generators
GENERATORS_FOR = {
    'S1': ['Baseline'],
    'S2': ['Baseline'],
    'S3': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S4': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S5': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
}

SCENARIO_LABELS = {
    'S1': 'S1 — Baseline (No Intervention)',
    'S2': 'S2 — Bias Mitigation Only (Reweighing)',
    'S3': 'S3 — Full Augmentation (Real + Synthetic)',
    'S4': 'S4 — Minority Oversampling (Synthetic for Race==0)',
    'S5': 'S5 — Combined (Bias Mitigation + Synthetic)',
}
# ─────────────────────────────────────────────────────────────────────────────


def get_models(seed):
    return {
        'LogisticRegression': LogisticRegression(random_state=seed, max_iter=1000),
        'RandomForest':       RandomForestClassifier(random_state=seed, max_depth=7),
        'CatBoost':           CatBoostClassifier(random_state=seed, verbose=0,
                                                 iterations=500, task_type='GPU',
                                                 allow_writing_files=False),
    }


def build_generator(name, metadata):
    if name == 'GaussianCopula':
        return GaussianCopulaSynthesizer(metadata)
    elif name == 'CTGAN':
        return CTGANSynthesizer(metadata, enforce_min_max_values=True,
                                enable_gpu=True, verbose=False)
    elif name == 'TVAE':
        return TVAESynthesizer(metadata, enforce_min_max_values=True,
                               enable_gpu=True, verbose=False)
    elif name == 'TabDDPM':
        return TabDDPMWrapper(metadata, device='cuda', steps=1000)
    raise ValueError(f'Unknown generator: {name}')


def save_synthetic(synth_df, scenario, generator, seed, metadata_info):
    path = os.path.join('synthetic_data', DATASET, scenario, generator)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f'seed_{seed}.csv')
    synth_df.to_csv(filepath, index=False)


def run_single(scenario, generator_name, seed, data, pbar=None):
    set_seed(seed)

    # ── Step 1: Stratified split ───────────────────────────────────────────
    train, test = train_test_split(
        data, test_size=0.2, random_state=seed,
        stratify=data[[TARGET, SENSITIVE]]
    )

    weights = None

    # ── Step 2: Reweighing (only S2 and S5) ───────────────────────────────
    if scenario in ['S2', 'S5']:
        X_rw = train.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
        y_rw = train[TARGET]
        rw = Reweighing(prot_attr=SENSITIVE)
        _, w = rw.fit_transform(X_rw, y_rw)
        weights = w.values if hasattr(w, 'values') else np.asarray(w)

    # ── Step 3: Synthetic data (S3, S4, S5) ───────────────────────────────
    if scenario in ['S3', 'S4', 'S5']:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train)

        gen = build_generator(generator_name, metadata)
        gen.fit(train)

        if scenario == 'S3':  # Full Augmentation
            synth = gen.sample(len(train))
            train_final = pd.concat([train, synth], ignore_index=True)

        elif scenario == 'S4':  # Minority Group Only (race == 0 = African-American)
            minority = train[train[SENSITIVE] == 0]
            synth = gen.sample(len(minority))
            train_final = pd.concat([train, synth], ignore_index=True)

        elif scenario == 'S5':  # Combined — synth from already-reweighed train
            synth = gen.sample(len(train))
            train_final = pd.concat([train, synth], ignore_index=True)
            # Recompute weights on expanded set
            X_rw2 = train_final.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
            y_rw2 = train_final[TARGET]
            rw2 = Reweighing(prot_attr=SENSITIVE)
            _, w2 = rw2.fit_transform(X_rw2, y_rw2)
            weights = w2.values if hasattr(w2, 'values') else np.asarray(w2)

        # Save synthetic data
        save_synthetic(synth, scenario, generator_name, seed,
                       metadata_info={'generator': generator_name,
                                      'scenario': scenario, 'seed': seed,
                                      'n_samples': len(synth),
                                      'strategy': 'full' if scenario in ['S3', 'S5'] else 'minority'})
    else:
        train_final = train

    # ── Step 4 & 5: Train and Evaluate ────────────────────────────────────
    X_train = train_final.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
    y_train = train_final[TARGET]
    X_test  = test.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
    y_test  = test[TARGET]
    s_test  = test[SENSITIVE]

    results = []
    models  = get_models(seed)

    for model_name, model in models.items():
        if pbar:
            pbar.set_postfix_str(f'{model_name}', refresh=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if weights is not None:
                model.fit(X_train, y_train, sample_weight=weights)
            else:
                model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_prob, s_test)
        metrics.update({
            'dataset':   DATASET,
            'scenario':  scenario,
            'generator': generator_name,
            'model':     model_name,
            'seed':      seed,
        })
        results.append(metrics)

    return results


def main():
    print('\n' + '═' * 60)
    print('  bias2fair-synth  ·  Fairness Experiment Pipeline v2')
    print('═' * 60)

    data = load_compas()

    print(f'\n  ✔ Dataset loaded: {data.shape[0]} records × {data.shape[1]} features')
    print(f'  ✔ Target  : {TARGET}  |  distribution: '
          f"{dict(data[TARGET].value_counts().items())}")
    print(f'  ✔ Sensitive: {SENSITIVE}  '
          f'(0=African-Amer: {(data[SENSITIVE]==0).sum()}, '
          f'1=Caucasian: {(data[SENSITIVE]==1).sum()})')
    print()

    seeds = get_fixed_seeds()
    os.makedirs('outputs', exist_ok=True)

    # ── Resume logic ───────────────────────────────────────────────────────
    processed = set()
    all_results = []
    csv_path = 'outputs/compas_results.csv'
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        for _, row in df_old.iterrows():
            processed.add((row['scenario'], row['generator'], int(row['seed'])))
        all_results = df_old.to_dict('records')

    # ── Count total tasks ──────────────────────────────────────────────────
    total_tasks = sum(
        len(GENERATORS_FOR[sc]) * len(seeds)
        for sc in SCENARIOS
    )
    done_tasks = len(processed)

    print(f'  → Resuming: {done_tasks}/{total_tasks} tasks already done.\n')

    with tqdm(total=total_tasks, initial=done_tasks, unit='run',
              bar_format='  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
              colour='green') as pbar:

        for scenario in SCENARIOS:
            generators = GENERATORS_FOR[scenario]
            pbar.set_description(f'{SCENARIO_LABELS[scenario][:42]}')

            for gen in generators:
                for seed in seeds:
                    if (scenario, gen, seed) in processed:
                        pbar.update(0)  # Already counted in initial
                        continue

                    pbar.set_postfix_str(f'gen={gen}  seed={seed}', refresh=True)

                    try:
                        res = run_single(scenario, gen, seed, data, pbar=pbar)
                        all_results.extend(res)
                        processed.add((scenario, gen, seed))

                        # Incremental save
                        pd.DataFrame(all_results).to_csv(csv_path, index=False)
                    except Exception as e:
                        tqdm.write(f'  ⚠ Error [{scenario}/{gen}/seed={seed}]: {e}')

                    pbar.update(1)

    # ── Summary ────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    summary = df.groupby(['scenario', 'generator', 'model']).agg({
        'f1':                          ['mean', 'std'],
        'auc_roc':                     ['mean', 'std'],
        'disparate_impact':            ['mean', 'std'],
        'statistical_parity_difference': ['mean', 'std'],
    }).round(4)
    summary.to_csv('outputs/summary_metrics.csv')

    print('\n' + '─' * 60)
    print('  ✔ All done!  Results in outputs/compas_results.csv')
    print('─' * 60 + '\n')


if __name__ == '__main__':
    main()
