"""
plot_utility_metrics.py

Plots saved to plots/compas/:
  - accuracy_lines.png         : Accuracy by scenario × model
  - auc_pr_lines.png           : AUC-PR by scenario × model
  - precision_recall_lines.png : Precision + Recall (2-panel)
  - roc_curve_traditional.png  : ROC curve per scenario
  - pr_curve_traditional.png   : PR  curve per scenario

Usage:
    python scripts/plots/plot_utility_metrics.py
    python scripts/plots/plot_utility_metrics.py --model SVM --seed 0
"""
import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# ── Constants ─────────────────────────────────────────────────────────────────
SCENARIO_ORDER = ['S1', 'S2', 'S3_1.5', 'S3_2.0', 'S3_3.0', 'S4', 'S5', 'S6']
SCENARIO_LABELS = {
    'S1':     'S1\nBaseline',
    'S2':     'S2\nMitigator',
    'S3_1.5': 'S3\n1.5x',
    'S3_2.0': 'S3\n2.0x',
    'S3_3.0': 'S3\n3.0x',
    'S4':     'S4\nMinority',
    'S5':     'S5\nCombined',
    'S6':     'S6\nRew+Synth',
}

# OUT_DIR is set after argparse in main()
OUT_DIR = 'plots/compas'  # default; overridden at runtime

# ── Theme & Helpers ───────────────────────────────────────────────────────────
def apply_theme():
    sns.set_context('paper', font_scale=1.4)
    sns.set_style('whitegrid', {'grid.linestyle': '--'})


def save_fig(fig, filename, dpi=300):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {path}')


def load_results(dataset_name: str):
    """Load {dataset}_results.csv with ordered scenario_label column."""
    df = pd.read_csv(f'outputs/{dataset_name}_results.csv')
    df['scenario_label'] = df['scenario'].map(SCENARIO_LABELS).fillna(df['scenario'])
    label_seq = [SCENARIO_LABELS.get(s, s) for s in SCENARIO_ORDER]
    present = [l for l in label_seq if l in df['scenario_label'].unique()]
    df['scenario_label'] = pd.Categorical(df['scenario_label'], categories=present, ordered=True)
    return df, present


# ── Line plot factory ─────────────────────────────────────────────────────────
def _line_plot(df, y_col, title, ylabel, filename, ylim_pad=0.01):
    agg = df.groupby('scenario_label', observed=True)[y_col].mean()
    y_min = np.floor(agg.min() * 100) / 100
    y_max = np.ceil(agg.max() * 100) / 100
    pad = max(ylim_pad, 0.005)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(
        data=df, ax=ax,
        x='scenario_label', y=y_col,
        hue='model', style='model',
        markers=['o', 's', 'D'],
        dashes=False,
        linewidth=2.5, markersize=9,
        errorbar=None,
        palette='Dark2',
    )
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Cenários Experimentais', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.legend(title='Modelo', bbox_to_anchor=(1.02, 1), loc='upper left',
              frameon=True, shadow=True)
    sns.despine()
    fig.tight_layout()
    save_fig(fig, filename)


def plot_accuracy_lines(df):
    print('Plotting accuracy lines...')
    _line_plot(df, 'accuracy',
               'Impacto das Estratégias de Mitigação na Acurácia',
               'Acurácia Média', 'accuracy_lines.png')


def plot_auc_pr_lines(df):
    print('Plotting AUC-PR lines...')
    _line_plot(df, 'auc_pr',
               'Impacto das Estratégias de Mitigação na AUC-PR',
               'AUC-PR Média', 'auc_pr_lines.png')


def plot_precision_recall_lines(df):
    print('Plotting precision/recall lines...')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, y_col, title, ylabel in [
        (axes[0], 'recall',    'Evolução do Recall',    'Recall Médio'),
        (axes[1], 'precision', 'Evolução da Precision', 'Precisão Média'),
    ]:
        agg = df.groupby('scenario_label', observed=True)[y_col].mean()
        y_min = np.floor(agg.min() * 100) / 100
        y_max = np.ceil(agg.max() * 100) / 100
        show_legend = (ax is axes[1])

        sns.lineplot(
            data=df, ax=ax,
            x='scenario_label', y=y_col,
            hue='model', style='model',
            markers=['o', 's', 'D'],
            dashes=False,
            linewidth=2.5, markersize=9,
            errorbar=None,
            palette='Dark2',
            legend=show_legend,
        )
        ax.set_ylim(y_min - 0.02, y_max + 0.02)
        ax.set_title(title, fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel('Cenários Experimentais', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')

    axes[1].legend(title='Modelo', bbox_to_anchor=(1.02, 1), loc='upper left',
                   frameon=True, shadow=True)
    fig.suptitle('Análise Detalhada: Recall e Precision por Cenário',
                 fontsize=17, fontweight='bold', y=1.02)
    sns.despine()
    fig.tight_layout()
    save_fig(fig, 'precision_recall_lines.png')


# ── Traditional curves ────────────────────────────────────────────────────────
def _load_predictions(seed, model, dataset_name='compas'):
    """Return dict {label: DataFrame} for each scenario prediction file."""
    d = dataset_name
    scenarios = {
        'S1 (Baseline)':          f'outputs/predictions/{d}_S1_None_Baseline_{model}_seed{seed}.csv',
        'S2 (Reweighing)':        f'outputs/predictions/{d}_S2_Reweighing_Baseline_{model}_seed{seed}.csv',
        'S3 (Aug-CTGAN 1.5x)':    f'outputs/predictions/{d}_S3_1.5_None_CTGAN_{model}_seed{seed}.csv',
        'S4 (Full Synth-CTGAN)':  f'outputs/predictions/{d}_S4_None_CTGAN_{model}_seed{seed}.csv',
        'S5 (Conditional-CTGAN)': f'outputs/predictions/{d}_S5_None_CTGAN_{model}_seed{seed}.csv',
        'S6 (Hybrid-CTGAN)':      f'outputs/predictions/{d}_S6_Reweighing_CTGAN_{model}_seed{seed}.csv',
    }
    loaded = {}
    for label, path in scenarios.items():
        if os.path.exists(path):
            loaded[label] = pd.read_csv(path)
        else:
            print(f'  [WARN] file not found: {path}')
    return loaded


def plot_roc_curve(seed=42, model='CatBoost', dataset_name='compas'):
    print(f'Plotting ROC curve (model={model}, seed={seed})...')
    preds = _load_predictions(seed, model, dataset_name)
    if not preds:
        print('  [SKIP] No prediction files found.')
        return

    colors = sns.color_palette('tab10', len(preds))
    fig, ax = plt.subplots(figsize=(9, 8))

    for (label, df_pred), color in zip(preds.items(), colors):
        fpr, tpr, _ = roc_curve(df_pred['y_true'], df_pred['y_prob'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--',
            label='Random (AUC=0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'ROC Curve — Cenários\n(Model: {model} | Seed: {seed})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'roc_curve_traditional.png')


def plot_pr_curve(seed=42, model='CatBoost', dataset_name='compas'):
    print(f'Plotting PR curve (model={model}, seed={seed})...')
    preds = _load_predictions(seed, model, dataset_name)
    if not preds:
        print('  [SKIP] No prediction files found.')
        return

    colors = sns.color_palette('tab10', len(preds))
    base_rate = 0.5
    fig, ax = plt.subplots(figsize=(9, 8))

    for (label, df_pred), color in zip(preds.items(), colors):
        precision, recall, _ = precision_recall_curve(df_pred['y_true'], df_pred['y_prob'])
        pr_auc = average_precision_score(df_pred['y_true'], df_pred['y_prob'])
        ax.plot(recall, precision, color=color, lw=2,
                label=f'{label} (AUC-PR={pr_auc:.3f})')
        if label.startswith('S1'):
            base_rate = df_pred['y_true'].mean()

    ax.plot([0, 1], [base_rate, base_rate], color='black', lw=1.5, linestyle='--',
            label=f'Random (AUC-PR={base_rate:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve — Cenários\n(Model: {model} | Seed: {seed})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'pr_curve_traditional.png')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description='Plot utility metrics for bias2fair-synth.')
    parser.add_argument('--dataset', type=str, default='compas',
                        choices=['compas', 'adult', 'diabetes'],
                        help='Dataset to plot (default: compas)')
    parser.add_argument('--model', default='CatBoost',
                        choices=['CatBoost', 'LogisticRegression', 'SVM'],
                        help='Model for ROC/PR traditional curves (default: CatBoost)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for ROC/PR traditional curves (default: 42)')
    args = parser.parse_args()

    OUT_DIR = os.path.join('plots', args.dataset)
    apply_theme()
    df, _ = load_results(args.dataset)

    print('\n── Line plots ───────────────────────────────')
    plot_accuracy_lines(df)
    plot_auc_pr_lines(df)
    plot_precision_recall_lines(df)

    print('\n── Traditional curves ───────────────────────')
    plot_roc_curve(seed=args.seed, model=args.model, dataset_name=args.dataset)
    plot_pr_curve(seed=args.seed, model=args.model, dataset_name=args.dataset)

    print(f'\nDone! All plots saved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
