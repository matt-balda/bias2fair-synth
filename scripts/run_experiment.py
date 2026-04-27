import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import sys
import shutil
import datetime
import pandas as pd
import numpy as np

# Override print to also log to experiment.log
import builtins
_original_print = builtins.print

def _logged_print(*args, **kwargs):
    _original_print(*args, **kwargs)
    # Only log if writing to stdout (default)
    if kwargs.get('file') in (None, sys.stdout):
        sep = kwargs.get('sep', ' ')
        msg = sep.join(map(str, args))
        try:
            with open('experiment.log', 'a', encoding='utf-8') as f:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"[{ts}] {msg}\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

builtins.print = _logged_print

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
from sdv.sampling import Condition
from tqdm import tqdm
import logging

# Silence everything
logging.disable(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.data_loader import load_dataset, DATASET_CONFIGS
from utils.metrics import calculate_metrics
from scripts.seed_manager import get_fixed_seeds, set_seed
from utils.diffusion_wrapper import TabDDPMWrapper

# ─── SCENARIOS ────────────────────────────────────────────────────────────────
SCENARIOS = ['S1', 'S2', 'S3_1.5', 'S3_2.0', 'S3_3.0', 'S4', 'S5', 'S6']
MITIGATORS = ['Reweighing', 'DIRemover', 'LFR']

GENERATORS_FOR = {
    'S1':     ['Baseline'],
    'S2':     ['Baseline'],
    'S3_1.5': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S3_2.0': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S3_3.0': ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S4':     ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S5':     ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
    'S6':     ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM'],
}

SCENARIO_LABELS = {
    'S1':     'S1 — Baseline (No Intervention)',
    'S2':     'S2 — Bias Mitigation Only',
    'S3_1.5': 'S3_1.5 — Full Augmentation (alpha=1.5)',
    'S3_2.0': 'S3_2.0 — Full Augmentation (alpha=2.0)',
    'S3_3.0': 'S3_3.0 — Full Augmentation (alpha=3.0)',
    'S4':     'S4 — Minority Oversampling (Sensitive x Target)',
    'S5':     'S5 — Conditional Generation by Outcome',
    'S6':     'S6 — Hybrid (Mitigation + Conditional Gen)',
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


def save_synthetic(synth_df, scenario, generator, seed, dataset_name):
    path = os.path.join('synthetic_data', dataset_name, scenario, generator)
    os.makedirs(path, exist_ok=True)
    synth_df.to_csv(os.path.join(path, f'seed_{seed}.csv'), index=False)


def run_single(scenario, generator_name, mitigator_name, seed, data,
               target, sensitive, dataset_name, pbar=None):
    """
    Run one experiment cell.

    Args:
        scenario:       e.g. 'S1', 'S3_2.0', …
        generator_name: e.g. 'CTGAN', 'Baseline', …
        mitigator_name: e.g. 'Reweighing', 'None', …
        seed:           int random seed
        data:           full dataset DataFrame
        target:         column name of the binary target
        sensitive:      column name of the binary sensitive attribute
        dataset_name:   'compas' | 'adult' | 'diabetes'
        pbar:           optional tqdm progress bar
    """
    set_seed(seed)

    # ── Step 1: Stratified split ───────────────────────────────────────────
    train, test = train_test_split(
        data, test_size=0.2, random_state=seed,
        stratify=data[[target, sensitive]]
    )
    train = train.copy()
    test  = test.copy()

    weights     = None
    dir_remover = None
    lfr_model   = None

    # ── Step 2: Bias Mitigation (S2 and S6) ───────────────────────────────
    if scenario in ['S2', 'S6']:
        if mitigator_name == 'Reweighing':
            X_rw = train.drop(columns=[target]).set_index(sensitive, drop=False)
            y_rw = train[target]
            rw = Reweighing(prot_attr=sensitive)
            _, w = rw.fit_transform(X_rw, y_rw)
            weights = w.values if hasattr(w, 'values') else np.asarray(w)

        elif mitigator_name == 'DIRemover':
            bld_train = BinaryLabelDataset(df=train, label_names=[target],
                                           protected_attribute_names=[sensitive])
            dir_remover = DisparateImpactRemover(repair_level=1.0)
            train_repaired = dir_remover.fit_transform(bld_train)
            train_repaired_df, _ = train_repaired.convert_to_dataframe()
            for col in train.columns:
                if col in train_repaired_df.columns:
                    train[col] = train_repaired_df[col].values

        elif mitigator_name == 'LFR':
            X_lfr_in = train.drop(columns=[target]).set_index(sensitive, drop=False)
            y_lfr_in = train[target]
            lfr_model = LearnedFairRepresentations(prot_attr=sensitive)
            X_lfr_out = lfr_model.fit_transform(X_lfr_in, y_lfr_in)
            train_lfr = X_lfr_out.copy()
            train_lfr[target] = y_lfr_in.values
            train = train_lfr.reset_index(drop=True)

    # ── Step 3: Synthetic data (S3, S4, S5, S6) ───────────────────────────
    synth = None
    if scenario.startswith('S3') or scenario in ['S4', 'S5', 'S6']:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train)

        # Fix misidentified PIIs (e.g. SDV thinking 'workclass_State-gov' is a US State)
        for col in list(metadata.columns.keys()):
            meta = metadata.columns[col]
            if meta['sdtype'] not in ['numerical', 'categorical', 'boolean', 'datetime']:
                if pd.api.types.is_numeric_dtype(train[col]):
                    metadata.update_column(column_name=col, sdtype='numerical')
                else:
                    metadata.update_column(column_name=col, sdtype='categorical')

        gen = build_generator(generator_name, metadata)

        if scenario.startswith('S3'):  # Full Augmentation (alpha)
            alpha = float(scenario.split('_')[1])
            gen.fit(train)
            n_to_generate = int((alpha - 1.0) * len(train))
            if n_to_generate > 0:
                synth = gen.sample(n_to_generate)
                train_final = pd.concat([train, synth], ignore_index=True)
            else:
                train_final = train

        elif scenario == 'S4':  # Equalize all Sensitive × Target groups
            group_counts = train.groupby([sensitive, target]).size()
            max_count = group_counts.max()
            gen.fit(train)
            synths = []
            for (sens_val, targ_val), count in group_counts.items():
                if count < max_count:
                    n_to_generate = int(max_count - count)
                    if hasattr(gen, 'sample_from_conditions'):
                        cond = Condition(
                            num_rows=n_to_generate,
                            column_values={sensitive: sens_val, target: targ_val}
                        )
                        synths.append(gen.sample_from_conditions(conditions=[cond]))
                    else:
                        subset = train[(train[sensitive] == sens_val) &
                                       (train[target] == targ_val)]
                        tmp = build_generator(generator_name, metadata)
                        tmp.fit(subset)
                        synths.append(tmp.sample(n_to_generate))
            if synths:
                synth = pd.concat(synths, ignore_index=True)
                train_final = pd.concat([train, synth], ignore_index=True)
            else:
                train_final = train

        elif scenario == 'S5':  # Generate up to n_ideal per stratum
            n_ideal = len(train) // 4
            group_counts = train.groupby([sensitive, target]).size()
            gen.fit(train)
            synths = []
            for (sens_val, targ_val), count in group_counts.items():
                if count < n_ideal:
                    n_to_generate = int(n_ideal - count)
                    if hasattr(gen, 'sample_from_conditions'):
                        cond = Condition(
                            num_rows=n_to_generate,
                            column_values={sensitive: sens_val, target: targ_val}
                        )
                        synths.append(gen.sample_from_conditions(conditions=[cond]))
                    else:
                        subset = train[(train[sensitive] == sens_val) &
                                       (train[target] == targ_val)]
                        tmp = build_generator(generator_name, metadata)
                        tmp.fit(subset)
                        synths.append(tmp.sample(n_to_generate))
            if synths:
                synth = pd.concat(synths, ignore_index=True)
                train_final = pd.concat([train, synth], ignore_index=True)
            else:
                train_final = train

        elif scenario == 'S6':  # Hybrid: Reweighing + S5 logic
            train_for_gen = (
                train.sample(n=len(train), replace=True, weights=weights,
                             random_state=seed)
                if weights is not None else train
            )
            n_ideal = len(train_for_gen) // 4
            group_counts = train_for_gen.groupby([sensitive, target]).size()
            gen.fit(train_for_gen)
            synths = []
            for (sens_val, targ_val), count in group_counts.items():
                if count < n_ideal:
                    n_to_generate = int(n_ideal - count)
                    if hasattr(gen, 'sample_from_conditions'):
                        cond = Condition(
                            num_rows=n_to_generate,
                            column_values={sensitive: sens_val, target: targ_val}
                        )
                        synths.append(gen.sample_from_conditions(conditions=[cond]))
                    else:
                        subset = train_for_gen[
                            (train_for_gen[sensitive] == sens_val) &
                            (train_for_gen[target] == targ_val)
                        ]
                        tmp = build_generator(generator_name, metadata)
                        tmp.fit(subset)
                        synths.append(tmp.sample(n_to_generate))
            if synths:
                synth = pd.concat(synths, ignore_index=True)
                train_final = pd.concat([train, synth], ignore_index=True)
            else:
                train_final = train

            # Recompute weights on expanded set if Reweighing was used
            if weights is not None:
                X_rw2 = train_final.drop(columns=[target]).set_index(sensitive, drop=False)
                y_rw2 = train_final[target]
                rw2 = Reweighing(prot_attr=sensitive)
                _, w2 = rw2.fit_transform(X_rw2, y_rw2)
                weights = w2.values if hasattr(w2, 'values') else np.asarray(w2)

        # Save synthetic data
        if synth is not None and len(synth) > 0:
            save_synthetic(synth, scenario, generator_name, seed, dataset_name)
    else:
        train_final = train

    # Capture sensitive and target from ORIGINAL test (before any transformation)
    s_test       = test[sensitive].values.copy()
    y_test_orig  = test[target].values.copy()

    # Apply DIRemover / LFR transform on test (only S2)
    if scenario == 'S2':
        if mitigator_name == 'DIRemover':
            bld_test = BinaryLabelDataset(df=test, label_names=[target],
                                          protected_attribute_names=[sensitive])
            test_repaired = dir_remover.fit_transform(bld_test)
            test_repaired_df, _ = test_repaired.convert_to_dataframe()
            for col in test.columns:
                if col in test_repaired_df.columns:
                    test[col] = test_repaired_df[col].values

        elif mitigator_name == 'LFR':
            X_test_lfr_in = test.drop(columns=[target]).set_index(sensitive, drop=False)
            X_test_lfr_out = lfr_model.transform(X_test_lfr_in)
            test_lfr = X_test_lfr_out.copy()
            test_lfr[target] = test[target].values
            test = test_lfr.reset_index(drop=True)

    X_train = train_final.drop(columns=[target]).set_index(sensitive, drop=False)
    y_train = train_final[target]
    X_test  = test.drop(columns=[target]).set_index(sensitive, drop=False)
    y_test  = test[target]

    # Z-score normalisation — fit only on train
    num_cols = X_train.select_dtypes(
        include=['int64', 'int32', 'float64', 'float32', 'uint8']
    ).columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    if len(num_cols) > 0:
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

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

        save_path = os.path.join(
            'outputs', 'predictions',
            f"{dataset_name}_{scenario}_{mitigator_name}_{generator_name}"
            f"_{model_name}_seed{seed}.csv"
        )
        metrics = calculate_metrics(y_test, y_pred, y_prob, s_test,
                                    save_path=save_path)
        metrics.update({
            'dataset':   dataset_name,
            'scenario':  scenario,
            'mitigator': mitigator_name,
            'generator': generator_name,
            'model':     model_name,
            'seed':      seed,
        })
        results.append(metrics)

    return results


def run_for_dataset(dataset_name: str) -> None:
    """Run all 6 scenarios for a single dataset."""
    cfg       = DATASET_CONFIGS[dataset_name]
    TARGET    = cfg['target']
    SENSITIVE = cfg['sensitive']

    print('\n' + '═' * 60)
    print(f'  bias2fair-synth  ·  Fairness Experiment Pipeline v3')
    print(f'  Dataset : {dataset_name.upper()}')
    print('═' * 60)

    data = cfg['loader']()

    # Pre-experiment limit: 2000 rows
    if len(data) > 2000:
        data = data.sample(n=2000, random_state=42).reset_index(drop=True)

    print(f'\n  ✔ Dataset loaded : {data.shape[0]} records × {data.shape[1]} features')
    print(f'  ✔ Target         : {TARGET}  |  {dict(data[TARGET].value_counts().items())}')
    print(f'  ✔ Sensitive      : {SENSITIVE}  |  '
          f'0={(data[SENSITIVE]==0).sum()}, 1={(data[SENSITIVE]==1).sum()}')
    print()

    seeds = get_fixed_seeds()
    os.makedirs('outputs', exist_ok=True)
    os.makedirs(os.path.join('outputs', 'predictions'), exist_ok=True)

    # ── Resume logic ───────────────────────────────────────────────────────
    processed   = set()
    all_results = []
    csv_path    = f'outputs/{dataset_name}_results.csv'

    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        for _, row in df_old.iterrows():
            mit = row.get('mitigator', 'None')
            if pd.isna(mit):
                mit = 'None'
            processed.add((row['scenario'], mit, row['generator'], int(row['seed'])))
        all_results = df_old.to_dict('records')

    # ── Count total tasks ──────────────────────────────────────────────────
    total_tasks = 0
    for sc in SCENARIOS:
        gens = len(GENERATORS_FOR[sc])
        mits = len(MITIGATORS) if sc == 'S2' else 1
        total_tasks += gens * mits * len(seeds)

    done_tasks = len(processed)
    print(f'  → Resuming: {done_tasks}/{total_tasks} tasks already done.\n')

    with tqdm(total=total_tasks, initial=done_tasks, unit='run',
              bar_format='  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
              colour='green') as pbar:

        for scenario in SCENARIOS:
            generators = GENERATORS_FOR[scenario]
            mits = (MITIGATORS if scenario == 'S2'
                    else (['Reweighing'] if scenario == 'S6' else ['None']))
            pbar.set_description(f'{SCENARIO_LABELS[scenario][:42]}')

            for mitigator in mits:
                for gen in generators:
                    for seed in seeds:
                        if (scenario, mitigator, gen, seed) in processed:
                            continue

                        pbar.set_postfix_str(
                            f'mit={mitigator[:3]} gen={gen} seed={seed}',
                            refresh=True
                        )
                        try:
                            res = run_single(
                                scenario, gen, mitigator, seed, data,
                                target=TARGET, sensitive=SENSITIVE,
                                dataset_name=dataset_name, pbar=pbar
                            )
                            all_results.extend(res)
                            processed.add((scenario, mitigator, gen, seed))
                            pd.DataFrame(all_results).to_csv(csv_path, index=False)
                            try:
                                with open('experiment.log', 'a', encoding='utf-8') as f:
                                    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    f.write(f"[{ts}] ✔ Completed: {scenario} | {mitigator} | {gen} | seed={seed}\n")
                                    f.flush()
                                    os.fsync(f.fileno())
                            except Exception:
                                pass
                        except Exception as e:
                            err_msg = f'  ⚠ Error [{scenario}/{mitigator}/{gen}/seed={seed}]: {e}'
                            tqdm.write(err_msg)
                            try:
                                with open('experiment.log', 'a', encoding='utf-8') as f:
                                    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    f.write(f"[{ts}] {err_msg}\n")
                            except Exception:
                                pass
                        pbar.update(1)

    # ── Summary ────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    summary = df.groupby(['scenario', 'mitigator', 'generator', 'model']).agg({
        'f1':                            ['mean', 'std'],
        'auc_roc':                       ['mean', 'std'],
        'disparate_impact':              ['mean', 'std'],
        'statistical_parity_difference': ['mean', 'std'],
    }).round(4)
    summary.to_csv(f'outputs/summary_metrics_{dataset_name}.csv')

    print('\n' + '─' * 60)
    print(f'  ✔ Done!  Results in outputs/{dataset_name}_results.csv')
    print('─' * 60 + '\n')


def main():
    all_datasets = list(DATASET_CONFIGS.keys())
    parser = argparse.ArgumentParser(
        description='bias2fair-synth — Fairness Experiment Pipeline',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--dataset', nargs='+', default=['compas'],
        choices=all_datasets,
        metavar='DATASET',
        help=(
            'One or more datasets to run.\n'
            f'Choices: {all_datasets}\n'
            'Examples:\n'
            '  --dataset compas\n'
            '  --dataset compas adult diabetes'
        )
    )
    parser.add_argument(
        '--clear', action='store_true',
        help='Clear previous results in outputs/ and plots/ (starts experiment from zero)'
    )
    args = parser.parse_args()

    if args.clear:
        if os.path.exists('outputs'):
            shutil.rmtree('outputs', ignore_errors=True)
        if os.path.exists('plots'):
            shutil.rmtree('plots', ignore_errors=True)
        if os.path.exists('experiment.log'):
            try:
                os.remove('experiment.log')
            except OSError:
                pass
        print("- Cleared previous results and logs. Starting fresh...")

    datasets = [d.lower() for d in args.dataset]
    # deduplicate while preserving order
    seen = set()
    datasets = [d for d in datasets if not (d in seen or seen.add(d))]

    print(f'\n  Datasets queued: {datasets}')

    for i, dataset_name in enumerate(datasets, 1):
        print(f'\n  [{i}/{len(datasets)}] Starting: {dataset_name.upper()}')
        run_for_dataset(dataset_name)

    print('\n' + '═' * 60)
    print(f'  ✔ All datasets complete: {", ".join(d.upper() for d in datasets)}')
    print('═' * 60 + '\n')


if __name__ == '__main__':
    main()
