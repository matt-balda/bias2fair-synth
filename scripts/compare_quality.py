import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

from utils.data_loader import load_compas
from utils.diffusion_wrapper import TabDDPMWrapper

def generate_quality_comparison(output_dir='reports'):
    print("Starting Quality Comparison (Real vs Synthetic)...")
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Load Real Data
    real_data = load_compas()
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_data)
    
    seed = 42
    
    # 2. Train Generators
    print("  Training TVAE...")
    tvae = TVAESynthesizer(metadata, enforce_min_max_values=True, enable_gpu=True)
    tvae.fit(real_data)
    synth_tvae = tvae.sample(len(real_data))
    
    print("  Training TabDDPM...")
    tabddpm = TabDDPMWrapper(metadata, device='cuda', steps=500) # Slightly faster for comparison
    tabddpm.fit(real_data)
    # Ensure column order
    synth_tabddpm = tabddpm.sample(len(real_data))
    
    # 3. SDV Evaluation
    print("  Evaluating Fidelity (SDV Metrics)...")
    metrics_summary = []
    
    for name, synth in [('TVAE', synth_tvae), ('TabDDPM', synth_tabddpm)]:
        quality_report = evaluate_quality(real_data, synth, metadata)
        diag_report = run_diagnostic(real_data, synth, metadata)
        
        quality_score = quality_report.get_score()
        diag_score = diag_report.get_score()
        
        metrics_summary.append({
            'Generator': name,
            'Quality Score': quality_score,
            'Diagnostic Score': diag_score
        })
        
    pd.DataFrame(metrics_summary).to_csv(os.path.join(output_dir, 'fidelity_metrics.csv'), index=False)
    
    # 4. Distribution Comparison (Plots)
    print("  Generating Comparison Plots...")
    cols_to_plot = ['age', 'priors_count', 'race']
    
    plt.figure(figsize=(18, 5))
    for i, col in enumerate(cols_to_plot):
        plt.subplot(1, 3, i+1)
        sns.kdeplot(real_data[col], label='Real', fill=True, alpha=0.3)
        sns.kdeplot(synth_tvae[col], label='TVAE', linestyle='--')
        sns.kdeplot(synth_tabddpm[col], label='TabDDPM', linestyle=':')
        plt.title(f'Distribution of {col}')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'distribution_comparison.png'))
    plt.close()
    
    # 5. Correlation Comparison (Heatmaps)
    print("  Generating Correlation Heatmaps...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    for ax, (name, df) in zip(axes, [('Real', real_data), ('TVAE', synth_tvae), ('TabDDPM', synth_tabddpm)]):
        corr = df.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax, cbar=False)
        ax.set_title(f'Correlation Matrix: {name}')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'correlation_comparison.png'))
    plt.close()
    
    print(f"Quality comparison finished. Reports saved to {output_dir}/")

if __name__ == "__main__":
    generate_quality_comparison()
