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
    metrics = ['f1', 'disparate_impact', 'average_absolute_odds_difference', 'equal_opportunity_difference']
    
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
    print(f"{'Scenario':<20} | {'F1-Score':<18} | {'DI':<18} | {'Delta DI':<18} | {'AOD':<18} | {'EOD':<18}")
    print("-" * 110)
    
    for idx, row in grouped.iterrows():
        f1_str = f"{row[('f1', 'mean')]:.4f} ± {row[('f1', 'std')]:.4f}"
        di_str = f"{row[('disparate_impact', 'mean')]:.4f} ± {row[('disparate_impact', 'std')]:.4f}"
        delta_di_str = f"{row[('delta_DI', 'mean')]:.4f} ± {row[('delta_DI', 'std')]:.4f}"
        aod_str = f"{row[('average_absolute_odds_difference', 'mean')]:.4f} ± {row[('average_absolute_odds_difference', 'std')]:.4f}"
        eod_str = f"{row[('equal_opportunity_difference', 'mean')]:.4f} ± {row[('equal_opportunity_difference', 'std')]:.4f}"
        
        print(f"{idx:<20} | {f1_str:<18} | {di_str:<18} | {delta_di_str:<18} | {aod_str:<18} | {eod_str:<18}")

    return grouped


def fmt(mean, std):
    """Format a mean ± std cell for LaTeX."""
    return f"${mean:.4f} \\pm {std:.4f}$"


def export_latex(dataset_name, grouped):
    """Export the summary table as a LaTeX booktabs table."""
    import os
    os.makedirs('outputs/tables', exist_ok=True)

    label = dataset_name.upper()
    out_path = f'outputs/tables/{dataset_name}_summary.tex'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Results for ' + label + r' dataset (mean $\pm$ std across seeds).}')
    lines.append(r'  \label{tab:results_' + dataset_name + r'}')
    lines.append(r'  \resizebox{\textwidth}{!}{')
    lines.append(r'  \begin{tabular}{lcccccc}')
    lines.append(r'    \toprule')
    lines.append(r'    \textbf{Scenario} & \textbf{F1-Score} & \textbf{DI} & \textbf{$\Delta$DI} & \textbf{AOD} & \textbf{EOD} \\')
    lines.append(r'    \midrule')

    # Group separators
    group_prefixes = {
        'S1': None,
        'S2': 'S1',
        'S3': 'S2',
        'S4': 'S3',
        'S5': 'S4',
        'S6': 'S5',
    }
    last_prefix = None

    for idx, row in grouped.iterrows():
        prefix = idx.split(' ')[0]  # e.g., 'S1', 'S2', ...
        if last_prefix is not None and prefix != last_prefix:
            lines.append(r'    \midrule')
        last_prefix = prefix

        # Escape special LaTeX chars in scenario name
        scenario_escaped = idx.replace('&', r'\&').replace('_', r'\_').replace('#', r'\#')

        f1   = fmt(row[('f1', 'mean')],                                     row[('f1', 'std')])
        di   = fmt(row[('disparate_impact', 'mean')],                        row[('disparate_impact', 'std')])
        ddi  = fmt(row[('delta_DI', 'mean')],                                row[('delta_DI', 'std')])
        aod  = fmt(row[('average_absolute_odds_difference', 'mean')],        row[('average_absolute_odds_difference', 'std')])
        eod  = fmt(row[('equal_opportunity_difference', 'mean')],            row[('equal_opportunity_difference', 'std')])

        lines.append(f'    {scenario_escaped} & {f1} & {di} & {ddi} & {aod} & {eod} \\\\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'  }')
    lines.append(r'\end{table}')

    tex = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f"LaTeX table saved to {out_path}")


if __name__ == '__main__':
    for ds in ['compas', 'adult', 'diabetes']:
        grouped = generate_table(ds)
        if grouped is not None:
            export_latex(ds, grouped)
