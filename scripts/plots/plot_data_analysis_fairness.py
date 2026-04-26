"""
1. Bhattacharyya Overlap Coefficient (T5a)
2. PCA 2D (T5b)
3. Bootstrap Disparate Impact (DI)
4. EDA (Target, Sensitive, Crosstab)
"""

import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from utils.data_loader import load_compas

# ── Paleta ───────────────────────────────────────────────────────────────────
BG     = '#ffffff'
PANEL  = '#ffffff'
PANEL2 = '#f8f9fa'
C0     = '#1f77b4'   # Azul padrão
C1     = '#ff7f0e'   # Laranja padrão
GOLD   = '#ffc107'
GREEN  = '#2ca02c'
RED    = '#d62728'
TEAL   = '#17becf'
TEXT   = '#111111'
SUBTEXT = '#555555'
GRID   = '#e0e0e0'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': PANEL,
    'axes.edgecolor':  GRID, 'axes.labelcolor': TEXT,
    'xtick.color':  SUBTEXT, 'ytick.color':    SUBTEXT,
    'text.color':      TEXT, 'grid.color':      GRID,
    'grid.linewidth':  0.6,  'font.family': 'DejaVu Sans',
    'legend.facecolor': PANEL2, 'legend.edgecolor': GRID,
})

def bhattacharyya_overlap(a, b):
    try:
        k0 = gaussian_kde(a, bw_method='silverman')
        k1 = gaussian_kde(b, bw_method='silverman')
        xs = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 300)
        return float(np.trapz(np.sqrt(k0(xs) * k1(xs)), xs))
    except Exception:
        return np.nan

def di_from_pred(y_pred, sensitive):
    s, p = np.asarray(sensitive), np.asarray(y_pred)
    pr1 = p[s == 1].mean()
    return p[s == 0].mean() / pr1 if pr1 > 0 else np.nan

def main():
    
    data = load_compas()
    TARGET    = 'two_year_recid'
    SENSITIVE = 'race'

    X_raw = data.drop(columns=[TARGET])
    y     = data[TARGET].values
    s     = data[SENSITIVE].values
    N     = len(data)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    out_dir = 'plots/compas/individual'
    os.makedirs(out_dir, exist_ok=True)

    # =========================================================================
    # 1. Bhattacharyya Overlap
    # =========================================================================
    print("Gerando: Bhattacharyya Overlap...")
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    bhat = {}
    for col in numeric_cols:
        a = X_raw[col][s == 0].dropna().values.astype(float)
        b = X_raw[col][s == 1].dropna().values.astype(float)
        if len(a) > 5 and len(b) > 5 and a.std() > 0 and b.std() > 0:
            bhat[col] = bhattacharyya_overlap(a, b)

    bhat_series  = pd.Series(bhat).sort_values()
    mean_overlap = bhat_series.mean()

    fig1 = plt.figure(figsize=(10, 8), facecolor=BG)
    ax1 = fig1.add_subplot(111)
    cols_s = bhat_series.index.tolist()
    vals_s = bhat_series.values
    colors1 = [RED if v < 0.85 else GREEN for v in vals_s]
    ax1.barh(range(len(cols_s)), vals_s, color=colors1, edgecolor=GRID, linewidth=0.8, height=0.65)
    ax1.set_yticks(range(len(cols_s)))
    ax1.set_yticklabels(cols_s, fontsize=10)
    ax1.axvline(0.85, color=GOLD, ls='--', lw=2, label='Limiar 0.85')
    ax1.axvline(mean_overlap, color=TEAL, ls=':', lw=2, label=f'Média = {mean_overlap:.3f}')
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel('Bhattacharyya Overlap Coefficient', color=SUBTEXT, fontsize=11)
    ax1.set_title('Sobreposição de Distribuição por Feature\n(African-American vs Caucasian)', fontsize=14, color=TEXT, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='lower right')
    ax1.grid(axis='x', alpha=0.3)
    for i, (v, col) in enumerate(zip(vals_s, colors1)):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9, color=col, fontweight='bold')
    out1 = os.path.join(out_dir, 'bhattacharyya_overlap.png')
    fig1.savefig(out1, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig1)

    # =========================================================================
    # 2. PCA 2D
    # =========================================================================
    print("Gerando: PCA 2D...")
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    c0 = X_pca[s == 0].mean(axis=0)
    c1 = X_pca[s == 1].mean(axis=0)
    centroid_dist = np.linalg.norm(c0 - c1)

    fig2 = plt.figure(figsize=(10, 8), facecolor=BG)
    ax2 = fig2.add_subplot(111)
    for grp, color, lbl in [(0, C0, 'African-American'), (1, C1, 'Caucasian')]:
        mask = s == grp
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.3, s=15, edgecolors='none', label=lbl)
        try:
            xi, yi = np.mgrid[X_pca[:,0].min():X_pca[:,0].max():80j, X_pca[:,1].min():X_pca[:,1].max():80j]
            zi = gaussian_kde(X_pca[mask].T)(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
            ax2.contour(xi, yi, zi, levels=5, colors=color, alpha=0.6, linewidths=1.2)
        except Exception:
            pass
    ax2.scatter(*c0, s=200, marker='X', c=C0, edgecolors='white', lw=1.5, zorder=5)
    ax2.scatter(*c1, s=200, marker='X', c=C1, edgecolors='white', lw=1.5, zorder=5)
    ax2.annotate('', xy=c1, xytext=c0, arrowprops=dict(arrowstyle='<->', color=TEXT, lw=2))
    mid = (c0 + c1) / 2
    ax2.text(mid[0], mid[1] + 0.4, f'd={centroid_dist:.3f}', ha='center', color=TEXT, fontsize=11, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', color=SUBTEXT, fontsize=11)
    ax2.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', color=SUBTEXT, fontsize=11)
    ax2.set_title('PCA Biplot: Separação dos Grupos no Espaço de Features', fontsize=14, color=TEXT, fontweight='bold', pad=15)
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.2)
    out2 = os.path.join(out_dir, 'pca_2d.png')
    fig2.savefig(out2, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig2)

    # =========================================================================
    # 3. Bootstrap DI
    # =========================================================================
    print("Gerando: Bootstrap DI...")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_cv  = LogisticRegression(random_state=42, max_iter=1000)
    y_prob_cv = cross_val_predict(model_cv, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred_cv = (y_prob_cv >= 0.5).astype(int)

    obs_di = di_from_pred(y_pred_cv, s)
    B = 2000
    rng = np.random.default_rng(42)
    n = len(y_pred_cv)
    boot_di = np.array([di_from_pred(y_pred_cv[idx := rng.integers(0, n, n)], s[idx]) for _ in range(B)])
    ci_di = np.percentile(boot_di, [2.5, 97.5])

    fig3 = plt.figure(figsize=(10, 7), facecolor=BG)
    ax3 = fig3.add_subplot(111)
    ax3.hist(boot_di, bins=60, color=C0, alpha=0.65, edgecolor='none', density=True, label=f'Bootstrap (B={B})')
    try:
        kde = gaussian_kde(boot_di)
        xs = np.linspace(boot_di.min(), boot_di.max(), 400)
        ax3.plot(xs, kde(xs), color=C0, lw=2.5, alpha=0.9)
    except Exception:
        pass
    ax3.axvspan(ci_di[0], ci_di[1], alpha=0.18, color=GOLD, zorder=2, label=f'CI 95%: [{ci_di[0]:.3f}, {ci_di[1]:.3f}]')
    ax3.axvline(ci_di[0], color=GOLD, lw=1.5, ls=':', zorder=3)
    ax3.axvline(ci_di[1], color=GOLD, lw=1.5, ls=':', zorder=3)
    ax3.axvline(1.0, color=GREEN, lw=2.0, ls='--', zorder=4, label='DI = 1 (equidade)')
    ax3.axvline(obs_di, color=RED, lw=2.8, zorder=5, label=f'Observado = {obs_di:.4f}')

    ax3.set_xlabel('Disparate Impact (DI)', color=SUBTEXT, fontsize=11)
    ax3.set_ylabel('Densidade', color=SUBTEXT, fontsize=11)
    ax3.set_title('Bootstrap CI — Disparate Impact', fontsize=14, color=TEXT, fontweight='bold', pad=15)
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(alpha=0.3)
    out3 = os.path.join(out_dir, 'bootstrap_di.png')
    fig3.savefig(out3, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig3)

    # =========================================================================
    # 4. EDA (1 figura, 3 painéis)
    # =========================================================================
    print("Gerando: EDA...")
    fig4 = plt.figure(figsize=(18, 6), facecolor=BG)
    gs = gridspec.GridSpec(1, 3, figure=fig4, wspace=0.3)

    # A - Target
    ax4a = fig4.add_subplot(gs[0])
    class_counts = pd.Series(y).value_counts().sort_index()
    bars_t = ax4a.bar(['Não reincidiu (0)', 'Reincidiu (1)'], [class_counts[0], class_counts[1]], color=[TEAL, C1], width=0.5, edgecolor=GRID, lw=1.2)
    for bar, v in zip(bars_t, [class_counts[0], class_counts[1]]):
        ax4a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{v}\n({v/N*100:.1f}%)', ha='center', va='bottom', fontsize=11, color=TEXT, fontweight='bold')
    ax4a.set_title('A: Distribuição do Target\n(Reincidência)', fontsize=13, color=TEXT, fontweight='bold', pad=12)
    ax4a.set_ylabel('Amostras', color=SUBTEXT)
    ax4a.set_ylim(0, max(class_counts)*1.25)
    ax4a.axhline(N/2, color=GOLD, ls='--', lw=1.5, alpha=0.7, label='50%')
    ax4a.legend(loc='upper right', fontsize=10)
    ax4a.grid(axis='y', alpha=0.3)

    # B - Sensitive
    ax4b = fig4.add_subplot(gs[1])
    group_counts = pd.Series(s).value_counts().sort_index()
    bars_s = ax4b.bar(['African-Am (0)', 'Caucasian (1)'], [group_counts[0], group_counts[1]], color=[C0, TEAL], width=0.5, edgecolor=GRID, lw=1.2)
    for bar, v in zip(bars_s, [group_counts[0], group_counts[1]]):
        ax4b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{v}\n({v/N*100:.1f}%)', ha='center', va='bottom', fontsize=11, color=TEXT, fontweight='bold')
    ax4b.set_title('B: Distribuição do Atributo Sensível\n(Raça)', fontsize=13, color=TEXT, fontweight='bold', pad=12)
    ax4b.set_ylabel('Amostras', color=SUBTEXT)
    ax4b.set_ylim(0, max(group_counts)*1.25)
    ax4b.grid(axis='y', alpha=0.3)

    # C - Crosstab
    ax4c = fig4.add_subplot(gs[2])
    cross = pd.crosstab(s, y)
    matrix = cross.values
    im = ax4c.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax4c.set_xticks([0, 1]); ax4c.set_xticklabels(['Não reincidiu', 'Reincidiu'], color=TEXT)
    ax4c.set_yticks([0, 1]); ax4c.set_yticklabels(['African-Am', 'Caucasian'], color=TEXT)
    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            col_txt = 'black' if val > matrix.max() * 0.55 else TEXT
            ax4c.text(j, i, f'{val}\n({val/N*100:.1f}%)', ha='center', va='center', fontsize=13, color=col_txt, fontweight='bold')
    ax4c.set_title('C: Target vs Sensível\n(Crosstab)', fontsize=13, color=TEXT, fontweight='bold', pad=12)
    plt.colorbar(im, ax=ax4c, pad=0.03)

    plt.tight_layout()
    out4 = os.path.join(out_dir, 'eda_groups.png')
    fig4.savefig(out4, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig4)

    print(f"Salvos em {out_dir}/")

if __name__ == "__main__":
    main()
