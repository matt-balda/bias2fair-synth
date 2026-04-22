import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from aif360.sklearn.preprocessing import Reweighing, LearnedFairRepresentations
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import BinaryLabelDataset
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

SCENARIOS = ['S1', 'S2', 'S3_1.5', 'S3_2.0', 'S3_3.0', 'S4', 'S5', 'S6']
MITIGATORS = ['Reweighing', 'DIRemover', 'LFR']

# S1/S2 don't iterate over generators
GENERATORS_FOR = {
    'S1': ['Baseline'],
    'S2': ['Baseline'],
    'S3_1.5': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S3_2.0': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S3_3.0': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S4': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S5': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S6': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
}

SCENARIO_LABELS = {
    'S1': 'S1 — Baseline (No Intervention)',
    'S2': 'S2 — Bias Mitigation Only',
    'S3_1.5': 'S3_1.5 — Full Augmentation (alpha=1.5)',
    'S3_2.0': 'S3_2.0 — Full Augmentation (alpha=2.0)',
    'S3_3.0': 'S3_3.0 — Full Augmentation (alpha=3.0)',
    'S4': 'S4 — Minority Oversampling (Sensitive x Target)',
    'S5': 'S5 — Conditional Generation by Outcome',
    'S6': 'S6 — Hybrid (Mitigation + Conditional Gen)',
}
# ─────────────────────────────────────────────────────────────────────────────


def get_models(seed):
    return {
        'LogisticRegression': LogisticRegression(random_state=seed, max_iter=1000),
        'SVM':                SVC(random_state=seed, probability=True),
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


def run_single(scenario, generator_name, mitigator_name, seed, data, pbar=None):
    set_seed(seed)

    # ── Step 1: Stratified split ───────────────────────────────────────────
    train, test = train_test_split(
        data, test_size=0.2, random_state=seed,
        stratify=data[[TARGET, SENSITIVE]]
    )

    weights = None
    dir_remover = None
    lfr_model = None

    # ── Step 2: Bias Mitigation (only S2 and S6) ───────────────────────────────
    if scenario in ['S2', 'S6']:
        if mitigator_name == 'Reweighing':
            X_rw = train.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
            y_rw = train[TARGET]
            rw = Reweighing(prot_attr=SENSITIVE)
            _, w = rw.fit_transform(X_rw, y_rw)
            weights = w.values if hasattr(w, 'values') else np.asarray(w)
            
        elif mitigator_name == 'DIRemover':
            bld_train = BinaryLabelDataset(df=train, label_names=[TARGET], protected_attribute_names=[SENSITIVE])
            dir_remover = DisparateImpactRemover(repair_level=1.0)
            train_repaired = dir_remover.fit_transform(bld_train)
            train_repaired_df, _ = train_repaired.convert_to_dataframe()
            for col in train.columns:
                if col in train_repaired_df.columns:
                    train[col] = train_repaired_df[col].values
                    
        elif mitigator_name == 'LFR':
            X_lfr_in = train.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
            y_lfr_in = train[TARGET]
            lfr_model = LearnedFairRepresentations(prot_attr=SENSITIVE)
            X_lfr_out = lfr_model.fit_transform(X_lfr_in, y_lfr_in)
            
            # Reconstruct train
            train_lfr = X_lfr_out.copy()
            train_lfr[TARGET] = y_lfr_in.values
            train_lfr = train_lfr.reset_index(drop=True)
            train = train_lfr

    # ── Step 3: Synthetic data (S3, S4, S5, S6) ───────────────────────────────
    if scenario.startswith('S3') or scenario in ['S4', 'S5', 'S6']:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train)

        gen = build_generator(generator_name, metadata)

        if scenario.startswith('S3'):  # Full Augmentation (alpha)
            alpha = float(scenario.split('_')[1])
            gen.fit(train)
            n_to_generate = int((alpha - 1.0) * len(train))
            if n_to_generate > 0:
                synth = gen.sample(n_to_generate)
                train_final = pd.concat([train, synth], ignore_index=True)
            else:
                synth = pd.DataFrame()
                train_final = train

        elif scenario == 'S4':  # Equalize all Sensitive x Target groups
            group_counts = train.groupby([SENSITIVE, TARGET]).size()
            max_count = group_counts.max()
            
            synths = []
            for (sens_val, targ_val), count in group_counts.items():
                if count < max_count:
                    n_to_generate = max_count - count
                    subset = train[(train[SENSITIVE] == sens_val) & (train[TARGET] == targ_val)]
                    gen.fit(subset)
                    synth_subset = gen.sample(n_to_generate)
                    synths.append(synth_subset)
            
            if synths:
                synth = pd.concat(synths, ignore_index=True)
                train_final = pd.concat([train, synth], ignore_index=True)
            else:
                synth = pd.DataFrame()
                train_final = train

        elif scenario == 'S5':  # Generate up to N_ideal = total / 4
            n_ideal = len(train) // 4
            group_counts = train.groupby([SENSITIVE, TARGET]).size()
            
            synths = []
            for (sens_val, targ_val), count in group_counts.items():
                if count < n_ideal:
                    n_to_generate = n_ideal - count
                    subset = train[(train[SENSITIVE] == sens_val) & (train[TARGET] == targ_val)]
                    gen.fit(subset)
                    synth_subset = gen.sample(n_to_generate)
                    synths.append(synth_subset)
                    
            if synths:
                synth = pd.concat(synths, ignore_index=True)
                train_final = pd.concat([train, synth], ignore_index=True)
            else:
                synth = pd.DataFrame()
                train_final = train

        elif scenario == 'S6':  # Hybrid — synth from already-mitigated train + S5 logic
            if weights is not None:
                train_for_gen = train.sample(n=len(train), replace=True, weights=weights, random_state=seed)
            else:
                train_for_gen = train
                
            n_ideal = len(train_for_gen) // 4
            group_counts = train_for_gen.groupby([SENSITIVE, TARGET]).size()
            
            synths = []
            for (sens_val, targ_val), count in group_counts.items():
                if count < n_ideal:
                    n_to_generate = n_ideal - count
                    subset = train_for_gen[(train_for_gen[SENSITIVE] == sens_val) & (train_for_gen[TARGET] == targ_val)]
                    gen.fit(subset)
                    synth_subset = gen.sample(n_to_generate)
                    synths.append(synth_subset)
                    
            if synths:
                synth = pd.concat(synths, ignore_index=True)
                train_final = pd.concat([train, synth], ignore_index=True)
            else:
                synth = pd.DataFrame()
                train_final = train
            
            # Recompute weights on expanded set if Reweighing was used
            if weights is not None:
                X_rw2 = train_final.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
                y_rw2 = train_final[TARGET]
                rw2 = Reweighing(prot_attr=SENSITIVE)
                _, w2 = rw2.fit_transform(X_rw2, y_rw2)
                weights = w2.values if hasattr(w2, 'values') else np.asarray(w2)

        # Save synthetic data only if something was actually generated
        if synth is not None and len(synth) > 0:
            save_synthetic(synth, scenario, generator_name, seed,
                           metadata_info={'generator': generator_name,
                                          'scenario': scenario, 'seed': seed,
                                          'n_samples': len(synth),
                                          'strategy': scenario})
    else:
        train_final = train

    # ── Step 4 & 5: Train and Evaluate ────────────────────────────────────
    # Apply mitigator on test set if necessary
    # Apply DIRemover/LFR on test only for S2 (S6 uses only Reweighing, no test transform needed)
    if scenario == 'S2':
        if mitigator_name == 'DIRemover':
            bld_test = BinaryLabelDataset(df=test, label_names=[TARGET], protected_attribute_names=[SENSITIVE])
            test_repaired = dir_remover.fit_transform(bld_test)
            test_repaired_df, _ = test_repaired.convert_to_dataframe()
            for col in test.columns:
                if col in test_repaired_df.columns:
                    test[col] = test_repaired_df[col].values
                    
        elif mitigator_name == 'LFR':
            X_test_lfr_in = test.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
            X_test_lfr_out = lfr_model.transform(X_test_lfr_in)
            test_lfr = X_test_lfr_out.copy()
            test_lfr[TARGET] = test[TARGET].values
            test_lfr = test_lfr.reset_index(drop=True)
            test = test_lfr

    X_train = train_final.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
    y_train = train_final[TARGET]
    X_test  = test.drop(columns=[TARGET]).set_index(SENSITIVE, drop=False)
    y_test  = test[TARGET]
    s_test  = test[SENSITIVE]
    
    # Standard scale features
    num_cols = X_train.select_dtypes(include=['int64', 'int32', 'float64', 'float32', 'uint8']).columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if len(num_cols) > 0:
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    results = []
    models  = get_models(seed)

    for model_name, model in models.items():
        if pbar:
            pbar.set_postfix_str(f'{model_name}', refresh=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if weights is not None:
                model.fit(X_train_scaled, y_train, sample_weight=weights)
            else:
                model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        save_path = os.path.join('outputs', 'predictions', f"{DATASET}_{scenario}_{mitigator_name}_{generator_name}_{model_name}_seed{seed}.csv")
        metrics = calculate_metrics(y_test, y_pred, y_prob, s_test, save_path=save_path)
        metrics.update({
            'dataset':   DATASET,
            'scenario':  scenario,
            'mitigator': mitigator_name,
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
            mit = row.get('mitigator', 'None')
            processed.add((row['scenario'], mit, row['generator'], int(row['seed'])))
        all_results = df_old.to_dict('records')

    # ── Count total tasks ──────────────────────────────────────────────────
    total_tasks = 0
    for sc in SCENARIOS:
        gens = len(GENERATORS_FOR[sc])
        # S2 iterates over all mitigators; S6 uses only Reweighing (1 mitigator)
        mits = len(MITIGATORS) if sc == 'S2' else 1
        total_tasks += gens * mits * len(seeds)
        
    done_tasks = len(processed)

    print(f'  → Resuming: {done_tasks}/{total_tasks} tasks already done.\n')

    with tqdm(total=total_tasks, initial=done_tasks, unit='run',
              bar_format='  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
              colour='green') as pbar:

        for scenario in SCENARIOS:
            generators = GENERATORS_FOR[scenario]
            mits = MITIGATORS if scenario == 'S2' else (['Reweighing'] if scenario == 'S6' else ['None'])
            pbar.set_description(f'{SCENARIO_LABELS[scenario][:42]}')

            for mitigator in mits:
                for gen in generators:
                    for seed in seeds:
                        if (scenario, mitigator, gen, seed) in processed:
                            pbar.update(0)  # Already counted in initial
                            continue

                        pbar.set_postfix_str(f'mit={mitigator[:3]} gen={gen} seed={seed}', refresh=True)

                        try:
                            res = run_single(scenario, gen, mitigator, seed, data, pbar=pbar)
                            all_results.extend(res)
                            processed.add((scenario, mitigator, gen, seed))

                            # Incremental save
                            pd.DataFrame(all_results).to_csv(csv_path, index=False)
                        except Exception as e:
                            tqdm.write(f'  ⚠ Error [{scenario}/{mitigator}/{gen}/seed={seed}]: {e}')

                        pbar.update(1)

    # ── Summary ────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    summary = df.groupby(['scenario', 'mitigator', 'generator', 'model']).agg({
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
