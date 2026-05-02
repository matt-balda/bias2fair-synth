import pandas as pd
import numpy as np

def generate_table(dataset_name):
    # Load dataset
    file_path = f'outputs/{dataset_name}_results.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return

    df['mitigator'] = df['mitigator'].fillna('None')
    df['generator'] = df['generator'].fillna('None')

    # Identify S1 mean DI
    s1_df = df[(df['scenario'] == 'S1') & (df['mitigator'] == 'None') & (df['generator'] == 'Baseline')]
    if s1_df.empty:
        print("S1 DF empty")
        s1_di_mean = np.nan
    else:
        s1_di_mean = s1_df['disparate_impact'].mean()
        print(f"S1 Mean DI: {s1_di_mean}")

    # We want to group by specific scenarios:
    # S1 (baseline)
    # S2 (DIRemover, LFR, Reweighing)
    # S3 (S3_1.5, S3_2.0, S3_3.0)

    # Let's create a custom "Scenario_Group" column
    def get_group(row):
        scen = row['scenario']
        mit = row['mitigator']
        if scen == 'S1':
            return 'S1 (Baseline)'
        elif scen == 'S2':
            return f'S2 ({mit})'
        elif scen in ['S3_1.5', 'S3_2.0', 'S3_3.0']:
            alpha = scen.split('_')[1]
            return f'S3 (alpha={alpha})'
        elif scen == 'S4':
            return 'S4 (Augmentation)'
        elif scen == 'S5':
            return 'S5 (Fair Synthetic)'
        elif scen == 'S6':
            return 'S6 (Mitigation + Fair Synthetic)'
        else:
            return None

    df['Scenario_Group'] = df.apply(get_group, axis=1)
    df_filtered = df[df['Scenario_Group'].notnull()]

    # Group by Scenario_Group and calculate mean and std
    metrics = ['f1', 'statistical_parity_difference', 'disparate_impact']
    
    grouped = df_filtered.groupby('Scenario_Group')[metrics].agg(['mean', 'std'])
    
    # Calculate Delta DI
    # For delta DI mean, it's (mean DI) - (S1 mean DI)
    # For std, it's just the std of DI (since shifting by a constant doesn't change std)
    grouped[('delta_DI', 'mean')] = grouped[('disparate_impact', 'mean')] - s1_di_mean
    grouped[('delta_DI', 'std')] = grouped[('disparate_impact', 'std')]

    # Sort the index based on the desired order
    order = [
        'S1 (Baseline)',
        'S2 (DIRemover)',
        'S2 (LFR)',
        'S2 (Reweighing)',
        'S3 (alpha=1.5)',
        'S3 (alpha=2.0)',
        'S3 (alpha=3.0)',
        'S4 (Augmentation)',
        'S5 (Fair Synthetic)',
        'S6 (Mitigation + Fair Synthetic)'
    ]
    # Only keep groups that are present in the data
    order = [o for o in order if o in grouped.index]
    grouped = grouped.loc[order]

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*50}")

    # Format the table nicely
    print(f"{'Scenario':<20} | {'F1-Score':<18} | {'SPD':<18} | {'DI':<18} | {'Delta DI':<18}")
    print("-" * 100)
    
    for idx, row in grouped.iterrows():
        f1_str = f"{row[('f1', 'mean')]:.4f} ± {row[('f1', 'std')]:.4f}"
        spd_str = f"{row[('statistical_parity_difference', 'mean')]:.4f} ± {row[('statistical_parity_difference', 'std')]:.4f}"
        di_str = f"{row[('disparate_impact', 'mean')]:.4f} ± {row[('disparate_impact', 'std')]:.4f}"
        delta_di_str = f"{row[('delta_DI', 'mean')]:.4f} ± {row[('delta_DI', 'std')]:.4f}"
        
        print(f"{idx:<20} | {f1_str:<18} | {spd_str:<18} | {di_str:<18} | {delta_di_str:<18}")

if __name__ == '__main__':
    for ds in ['compas', 'adult', 'diabetes']:
        generate_table(ds)
