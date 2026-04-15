import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from aif360.sklearn.preprocessing import Reweighing
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata

from utils.data_loader import load_compas
from utils.metrics import calculate_metrics, aggregate_results
from scripts.seed_manager import get_fixed_seeds, set_seed
from utils.diffusion_wrapper import TabDDPMWrapper

def get_models(seed):
    return {
        'LogisticRegression': LogisticRegression(random_state=seed, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=seed, max_depth=7),
        'CatBoost': CatBoostClassifier(random_state=seed, verbose=0, iterations=500, task_type='GPU'),
        'SVM': SVC(random_state=seed, probability=True)
    }

def run_scenario(scenario_id, data, seed, generator_name='GaussianCopula'):
    """
    Executes a specific scenario (S1-S6).
    """
    set_seed(seed)
    
    # Target and Sensitive Attribute
    target = 'two_year_recid'
    sensitive = 'race'
    
    # 1. Split
    train, test = train_test_split(
        data, test_size=0.2, random_state=seed, stratify=data[[target, sensitive]]
    )
    
    X_train = train.drop(columns=[target]).set_index(sensitive, drop=False)
    y_train = train[target]
    X_test = test.drop(columns=[target]).set_index(sensitive, drop=False)
    y_test = test[target]
    s_train = train[sensitive]
    s_test = test[sensitive]
    
    weights = None
    
    # 2. Preprocessing (Mitigation)
    if scenario_id in ['S2', 'S6']:
        # Reweighing from aif360.sklearn expects a DataFrame
        rw = Reweighing(prot_attr=sensitive)
        weights = rw.fit_transform(X_train, y_train)[1] # Get result weight vector
    
    # 3. Synthetic Data (Placeholders for S3-S6)
    if scenario_id in ['S3', 'S4', 'S5', 'S6']:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train)
        
        if generator_name == 'GaussianCopula':
            generator = GaussianCopulaSynthesizer(metadata)
        elif generator_name == 'CTGAN':
            generator = CTGANSynthesizer(metadata, enforce_min_max_values=True, enable_gpu=True)
        elif generator_name == 'TVAE':
            generator = TVAESynthesizer(metadata, enforce_min_max_values=True, enable_gpu=True)
        elif generator_name == 'TabDDPM':
            generator = TabDDPMWrapper(metadata, device='cuda')
        else:
            raise ValueError(f"Unknown generator: {generator_name}")
            
        generator.fit(train)
        
        if scenario_id == 'S3': # Augmentation
            synthetic_data = generator.sample(len(train))
            train = pd.concat([train, synthetic_data])
        elif scenario_id == 'S4': # Full Replacement
            train = generator.sample(len(train))
        elif scenario_id == 'S5': # Minority Group Only
            minority_data = train[train[sensitive] == 0]
            synthetic_minority = generator.sample(len(minority_data))
            train = pd.concat([train, synthetic_minority])
        elif scenario_id == 'S6': # Combined
            # Note: For combined, we should use the weighted training data or adjust generator
            # Protocol: "Apply bias mitigation ... then generate synthetic from adjusted"
            # Since Reweighing doesn't change features, we'll just augment.
            synthetic_data = generator.sample(len(train))
            train = pd.concat([train, synthetic_data])
            # We recalculate weights for the augmented dataset if needed, 
            # or keep previous ones (Protocol says "adjusted real + synthetic")
            X_tmp = train.drop(columns=[target]).set_index(sensitive, drop=False)
            y_tmp = train[target]
            rw = Reweighing(prot_attr=sensitive)
            weights = rw.fit_transform(X_tmp, y_tmp)[1]

        # Update X, y after augmentation/replacement
        X_train = train.drop(columns=[target]).set_index(sensitive, drop=False)
        y_train = train[target]
        s_train = train[sensitive]

    # 4. Model Training and Evaluation
    models = get_models(seed)
    scenario_results = []
    
    for model_name, model in models.items():
        if weights is not None:
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model.fit(X_train, y_train)
            
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_prob, s_test)
        metrics['model'] = model_name
        metrics['scenario'] = scenario_id
        metrics['generator'] = generator_name
        metrics['seed'] = seed
        scenario_results.append(metrics)
        
    return scenario_results

if __name__ == "__main__":
    data = load_compas()
    seeds = get_fixed_seeds()
    
    scenarios = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    generators = ['GaussianCopula', 'CTGAN', 'TVAE', 'TabDDPM']
    
    # Resume logic
    processed_configs = set()
    all_results = []
    if os.path.exists('outputs/compas_results.csv'):
        df_old = pd.read_csv('outputs/compas_results.csv')
        for _, row in df_old.iterrows():
            processed_configs.add((row['scenario'], row['generator'], int(row['seed'])))
        all_results = df_old.to_dict('records')
        print(f"Resuming from {len(processed_configs)//4} already processed experiments.")

    for scenario in scenarios:
        current_generators = generators if scenario in ['S3', 'S4', 'S5', 'S6'] else ['GaussianCopula']
        for gen in current_generators:
            print(f"Running Scenario {scenario} with Generator {gen}...")
            for seed in seeds:
                if (scenario, gen, seed) in processed_configs:
                    print(f"  Skipping Seed {seed} (already processed)")
                    continue
                print(f"  Seed {seed}...")
                res = run_scenario(scenario, data, seed, generator_name=gen)
                all_results.extend(res)
                
                # Incremental Save
                os.makedirs('outputs', exist_ok=True)
                pd.DataFrame(all_results).to_csv('outputs/compas_results.csv', index=False)
            
    # Save results
    os.makedirs('outputs', exist_ok=True)
    df_results = pd.DataFrame(all_results)
    # Final Save handled by incremental save in loop
    print("All results saved to outputs/compas_results.csv")
    
    # Summary
    summary = df_results.groupby(['scenario', 'generator', 'model']).mean()
    print("\nSummary (Mean Metrics):")
    print(summary[['f1', 'auc_roc', 'statistical_parity_difference', 'disparate_impact']])
