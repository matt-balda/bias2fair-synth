import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
import os

def check_stat_tests(df, scenarios_to_compare, models):
    test_results = []
    for s_a, s_b in scenarios_to_compare:
        for model in models:
            # For Baseline (S1), we only have GaussianCopula as placeholder
            gen_a = 'GaussianCopula'
            
            # For target scenario, find unique generators
            generators_b = df[df['scenario'] == s_b]['generator'].unique()
            
            for gen_b in generators_b:
                df_a = df[(df['scenario'] == s_a) & (df['model'] == model) & (df['generator'] == gen_a)].sort_values('seed')
                df_b = df[(df['scenario'] == s_b) & (df['model'] == model) & (df['generator'] == gen_b)].sort_values('seed')
                
                # Intersect seeds to ensure pairing
                common_seeds = set(df_a['seed']).intersection(set(df_b['seed']))
                df_a_p = df_a[df_a['seed'].isin(common_seeds)].sort_values('seed')
                df_b_p = df_b[df_b['seed'].isin(common_seeds)].sort_values('seed')
                
                if len(df_a_p) > 2:
                    try:
                        # Test if there's any difference (two-sided)
                        # We use F1 and Disparate Impact
                        _, p_f1 = wilcoxon(df_a_p['f1'], df_b_p['f1'])
                        _, p_di = wilcoxon(df_a_p['disparate_impact'], df_b_p['disparate_impact'])
                        test_results.append({
                            'comparison': f'{s_a} vs {s_b}',
                            'generator': gen_b,
                            'model': model,
                            'p_value_f1': p_f1,
                            'p_value_di': p_di,
                            'n': len(df_a_p)
                        })
                    except:
                        pass
    return pd.DataFrame(test_results)

def plot_radar(df, output_dir):
    # Radar chart for the best model (e.g., CatBoost)
    model_name = 'CatBoost'
    subset = df[df['model'] == model_name].groupby(['scenario', 'generator']).mean(numeric_only=True).reset_index()
    
    # We'll pick S3 (Augmentation) to compare generators
    radar_data = subset[subset['scenario'] == 'S3']
    
    categories = ['f1', 'auc_roc', 'disparate_impact', 'statistical_parity_difference']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for _, row in radar_data.iterrows():
        values = row[categories].values.flatten().tolist()
        # Scale DI to 0-1 (already is) and SP (abs and inverted)
        # SP: lower is better, so we plot 1 - abs(SP)
        values[3] = 1 - abs(values[3])
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['generator'])
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], ['F1', 'AUC', 'Fairness (DI)', 'Equity (1-SP)'])
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(f'Generator Comparison - {model_name} (Scenario S3)')
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'))
    plt.close()

def generate_robust_visualizations(csv_path='outputs/compas_results.csv', output_dir='reports'):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Generating Robust Visualizations...")
    
    # 1. Violin Plots (Stability)
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=df, x='scenario', y='f1', hue='generator', split=False, inner="quart")
    plt.title('Performance Stability (F1-Score) across Scenarios and Generators')
    plt.savefig(os.path.join(plot_dir, 'violin_f1_stability.png'))
    plt.close()
    
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=df, x='scenario', y='disparate_impact', hue='generator', split=False, inner="quart")
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Rule')
    plt.title('Fairness Stability (Disparate Impact) across Scenarios and Generators')
    plt.savefig(os.path.join(plot_dir, 'violin_di_stability.png'))
    plt.close()

    # 2. Radar Comparison
    plot_radar(df, plot_dir)

    # 3. Pareto Frontier (Publication Style)
    means = df.groupby(['scenario', 'generator', 'model']).mean(numeric_only=True).reset_index()
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=means, x='disparate_impact', y='f1', 
        hue='scenario', style='generator', s=100, alpha=0.9, palette='viridis'
    )
    plt.axvline(x=0.8, color='red', linestyle='--', alpha=0.5)
    plt.title('Fairness-Utility Pareto Frontier (Means)')
    plt.xlabel('Fairness (Disparate Impact)')
    plt.ylabel('Utility (F1-Score)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'pareto_publication.png'))
    plt.close()

    # 4. Statistical Summary
    models = df['model'].unique()
    scenarios_to_compare = [('S1', 'S2'), ('S1', 'S3'), ('S1', 'S4'), ('S1', 'S6'), ('S2', 'S6')]
    stats_df = check_stat_tests(df, scenarios_to_compare, models)
    stats_df.to_csv(os.path.join(output_dir, 'statistical_significance.csv'), index=False)
    
    # 5. Summary Table
    summary = df.groupby(['scenario', 'generator', 'model']).agg({
        'f1': ['mean', 'std'],
        'disparate_impact': ['mean', 'std']
    }).round(4)
    summary.to_csv(os.path.join(output_dir, 'summary_metrics_robust.csv'))
    
    print(f"All reports saved to {output_dir}/")

if __name__ == "__main__":
    generate_robust_visualizations()
