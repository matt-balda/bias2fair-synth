import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

def plot_traditional_roc():
    seed = 42
    model = "CatBoost"
    
    # Define the files to load to represent each scenario.
    # Using CatBoost, Seed 42, and CTGAN (as representative for synthetic scenarios)
    scenarios = {
        "S1 (Baseline)": f"outputs/predictions/compas_S1_None_Baseline_{model}_seed{seed}.csv",
        "S2 (Reweighing)": f"outputs/predictions/compas_S2_Reweighing_Baseline_{model}_seed{seed}.csv",
        "S3 (Data Augmentation - CTGAN)": f"outputs/predictions/compas_S3_1.5_None_CTGAN_{model}_seed{seed}.csv",
        "S4 (Full Synthetic - CTGAN)": f"outputs/predictions/compas_S4_None_CTGAN_{model}_seed{seed}.csv",
        "S5 (Conditional - CTGAN)": f"outputs/predictions/compas_S5_None_CTGAN_{model}_seed{seed}.csv",
        "S6 (Hybrid - CTGAN)": f"outputs/predictions/compas_S6_Reweighing_CTGAN_{model}_seed{seed}.csv"
    }
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette("tab10", len(scenarios))
    
    for (label, filepath), color in zip(scenarios.items(), colors):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            fpr, tpr, _ = roc_curve(df['y_true'], df['y_prob'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2, 
                     label=f'{label} (AUC = {roc_auc:.3f})')
        else:
            print(f"File not found: {filepath}")

    # Plot random guessing line
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random Guessing (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(f'ROC Curve Comparison Across Scenarios\n(Model: {model} | Seed: {seed})', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    
    out_dir = 'plots/compas'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'roc_curve_traditional.png')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"ROC Curve saved to {out_path}")

if __name__ == "__main__":
    plot_traditional_roc()
