"""
Extended Sufficiency Analysis — COMPAS Dataset
===============================================
T5 — Representação no Espaço de Features
      Bhattacharyya overlap, PCA biplot, densidade por grupo

T6 — Estabilidade da Métrica de Fairness
      Forest plot CI bootstrap para SPD e DI
      Bootstrap histogramas (SPD e DI)
      Permutation test com distribuição nula anotada

T7 — Calibração por Grupo
      Calibration curves por grupo sensível
      ECE (Expected Calibration Error) por grupo e modelo
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
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from utils.data_loader import load_compas

# ── Paleta ───────────────────────────────────────────────────────────────────
BG     = '#0f1117'
PANEL  = '#1a1d27'
PANEL2 = '#20243a'
C0     = '#f97b4f'   # African-American (unprivileged)
C1     = '#7c6ff7'   # Caucasian        (privileged)
GOLD   = '#ffd54f'
GREEN  = '#4caf7d'
RED    = '#ef5350'
TEAL   = '#4fc3f7'
TEXT   = '#e8eaf0'
SUBTEXT = '#8b8fa8'
GRID   = '#2e3250'

plt.rcParams.update({
    'figure.facecolor': BG,  'axes.facecolor': PANEL,
    'axes.edgecolor':  GRID, 'axes.labelcolor': TEXT,
    'xtick.color':  SUBTEXT, 'ytick.color':    SUBTEXT,
    'text.color':      TEXT, 'grid.color':      GRID,
    'grid.linewidth':  0.6,  'font.family': 'DejaVu Sans',
    'legend.facecolor': PANEL2, 'legend.edgecolor': GRID,
})


def section(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print('─'*62)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers métricas de fairness
# ─────────────────────────────────────────────────────────────────────────────
def spd_from_pred(y_pred, sensitive):
    s, p = np.asarray(sensitive), np.asarray(y_pred)
    return p[s == 0].mean() - p[s == 1].mean()

def di_from_pred(y_pred, sensitive):
    s, p = np.asarray(sensitive), np.asarray(y_pred)
    pr1 = p[s == 1].mean()
    return p[s == 0].mean() / pr1 if pr1 > 0 else np.nan

def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece_val += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece_val / len(y_true)

def bhattacharyya_overlap(a, b):
    try:
        k0 = gaussian_kde(a, bw_method='silverman')
        k1 = gaussian_kde(b, bw_method='silverman')
        xs = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 300)
        return float(np.trapz(np.sqrt(k0(xs) * k1(xs)), xs))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Funções de plot T6 — reutilizadas na figura composta e na dedicada
# ─────────────────────────────────────────────────────────────────────────────
def plot_t6_forest(ax, obs_spd, ci_spd, obs_di, ci_di, detailed=False):
    """
    Forest plot horizontal mostrando CI 95% bootstrap para SPD e DI.
    A linha verde indica o valor 'justo' de referência.
    CI que não toca a linha verde = viés robusto.
    """
    metrics  = ['SPD\n(Stat. Parity Diff.)', 'DI\n(Disparate Impact)']
    obs_vals = [obs_spd, obs_di]
    cis      = [ci_spd,  ci_di]
    refs     = [0.0, 1.0]
    colors_m = [TEAL, C0]

    for i, (obs, ci, ref, col) in enumerate(zip(obs_vals, cis, refs, colors_m)):
        y = i * 3.0
        ax.barh(y, ci[1] - ci[0], left=ci[0], height=0.7,
                color=col, alpha=0.22, edgecolor='none', zorder=2)
        ax.plot(ci, [y, y], color=col, lw=3.0, solid_capstyle='round', zorder=3)
        ax.scatter(obs, y, s=180, color=col, zorder=5, edgecolors='white', lw=1.8)
        # Rótulo da métrica
        ax.text(-0.55 + ref * 0.6, y,
                f'Observado: {obs:+.4f}\nCI 95%: [{ci[0]:+.3f}, {ci[1]:+.3f}]',
                va='center', ha='right' if obs < ref else 'left',
                fontsize=8.5, color=col, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc=PANEL2, ec=col, lw=0.8))
        ax.axvline(ref, color=GREEN, ls='--', lw=1.6, alpha=0.8,
                   label=f'{"SPD" if i==0 else "DI"} = {ref} (justo)' if i <= 1 else None)

    ax.set_yticks([0, 3.0])
    ax.set_yticklabels(metrics, fontsize=9.5, color=TEXT)
    ax.set_xlabel('Valor da Métrica de Fairness', color=SUBTEXT, fontsize=9)
    ax.set_title('Forest Plot — CI Bootstrap 95%\nCI fora da linha verde = viés estatisticamente robusto',
                 fontsize=10, color=TEXT, fontweight='bold', pad=8)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    spd_ok = ci_spd[0] > 0
    di_ok  = ci_di[0] > 1
    verdict_txt = "✔ Ambos CIs confirmam viés\n→ Intervenção justificada" \
                  if (spd_ok and di_ok) else "⚠ Verificar métricas individualmente"
    verdict_c = GREEN if (spd_ok and di_ok) else GOLD
    ax.text(0.98, 0.05, verdict_txt, transform=ax.transAxes,
            fontsize=9, color=verdict_c, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc=PANEL2, ec=verdict_c, lw=1.3))

    if detailed:
        ax.text(0.02, 0.98,
                "Como ler:\n"
                "• Ponto colorido = valor observado\n"
                "• Barra larga = intervalo 95% das\n"
                "  2000 replicações bootstrap\n"
                "• Linha verde tracejada = equidade\n"
                "• CI sem tocar a linha verde\n"
                "  → viés é robusto e real",
                transform=ax.transAxes, fontsize=8.5,
                va='top', ha='left', color=SUBTEXT,
                bbox=dict(boxstyle='round,pad=0.5', fc=PANEL2,
                          ec=GRID, lw=1.0, alpha=0.95))


def plot_t6_boot_hist(ax, boot_vals, obs_val, ci, metric, fair_val,
                      fair_label, color=TEAL):
    """
    Histograma + KDE da distribuição bootstrap de uma métrica.
    Shading CI, linha de referência justa, anotação do gap.
    """
    # Histograma
    ax.hist(boot_vals, bins=55, color=color, alpha=0.55, edgecolor='none',
            density=True, label=f'Bootstrap ({len(boot_vals):,} replicas)')

    # KDE suave
    try:
        kde = gaussian_kde(boot_vals)
        xs  = np.linspace(boot_vals.min(), boot_vals.max(), 400)
        ax.plot(xs, kde(xs), color=color, lw=2.5, alpha=0.9)
    except Exception:
        pass

    # CI shading
    ax.axvspan(ci[0], ci[1], alpha=0.18, color=GOLD, zorder=2,
               label=f'CI 95%: [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    ax.axvline(ci[0], color=GOLD, lw=1.5, ls=':', zorder=3)
    ax.axvline(ci[1], color=GOLD, lw=1.5, ls=':', zorder=3)

    # Linha de equidade
    ax.axvline(fair_val, color=GREEN, lw=2.0, ls='--', zorder=4, label=fair_label)

    # Valor observado
    ax.axvline(obs_val, color=RED, lw=2.8, zorder=5,
               label=f'Valor observado = {obs_val:+.4f}')

    # Seta de gap
    ylim_top = ax.get_ylim()[1] or 1.0
    mid_y = ylim_top * 0.6
    offset = boot_vals.std() * 1.8
    ax.annotate(
        f'Gap até equidade:\n{obs_val - fair_val:+.4f}',
        xy=(obs_val, mid_y),
        xytext=(obs_val + offset, mid_y + ylim_top * 0.12),
        color=RED, fontsize=9, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.6),
        bbox=dict(boxstyle='round,pad=0.35', fc=PANEL2, ec=RED, lw=1.0)
    )

    # Status
    ci_excl_fair = (ci[0] > fair_val) or (ci[1] < fair_val)
    status = "✔ CI não contém equidade\n→ Viés estatisticamente robusto" \
             if ci_excl_fair else \
             "⚠ CI inclui equidade\n→ Viés pode ser instável"
    status_c = GREEN if ci_excl_fair else GOLD

    ax.set_xlabel(metric, color=SUBTEXT, fontsize=9)
    ax.set_ylabel('Densidade', color=SUBTEXT, fontsize=9)
    ax.set_title(f'Bootstrap — {metric}\nUncerteza em torno da estimativa (B=2000)',
                 fontsize=10, color=TEXT, fontweight='bold', pad=8)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.05, status, transform=ax.transAxes, fontsize=9,
            color=status_c, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', fc=PANEL2, ec=status_c, lw=1.3))


def plot_t6_permutation(ax, perm_spd, obs_spd, p_perm, B_perm, detailed=False):
    """
    Permutation test: histograma da distribuição nula vs SPD observado.
    Região de rejeição pintada em vermelho. p-valor anotado.
    """
    # Histograma nulo
    ax.hist(perm_spd, bins=60, color=C1, alpha=0.55, edgecolor='none',
            density=True, label=f'Distribuição nula H₀\n({B_perm:,} permutações)')

    # KDE da nula
    try:
        kde_null = gaussian_kde(perm_spd)
        xs_all   = np.linspace(perm_spd.min(), perm_spd.max(), 400)
        ax.plot(xs_all, kde_null(xs_all), color=C1, lw=2.5, alpha=0.9)

        # Região de rejeição bilateral
        thresh = abs(obs_spd)
        xs_lo  = np.linspace(perm_spd.min(), -thresh, 200)
        xs_hi  = np.linspace(thresh, perm_spd.max(), 200)
        ax.fill_between(xs_lo, kde_null(xs_lo), alpha=0.35, color=RED,
                        label=f'Região de rejeição\n(p = {p_perm:.4f})')
        ax.fill_between(xs_hi, kde_null(xs_hi), alpha=0.35, color=RED)
    except Exception:
        pass

    # Linha H0 = 0
    ax.axvline(0, color=GREEN, lw=1.8, ls='--', alpha=0.7,
               label='SPD = 0 (H₀ esperado)')

    # Linhas do valor observado (bilateral)
    ax.axvline(obs_spd, color=RED, lw=2.8, zorder=5,
               label=f'SPD observado = {obs_spd:+.4f}')
    ax.axvline(-obs_spd, color=RED, lw=1.5, ls=':', alpha=0.55, zorder=4,
               label='Espelho bilateral')

    # Estatística: p-valor
    sig = p_perm < 0.05
    sig_txt = (f"✔ p = {p_perm:.4f} < 0.05\nViés não é acidental\n→ Justifica intervenção"
               if sig else
               f"✗ p = {p_perm:.4f} ≥ 0.05\nViés pode ser acidental")
    sig_c = GREEN if sig else RED

    ax.set_xlabel('Statistical Parity Difference', color=SUBTEXT, fontsize=9)
    ax.set_ylabel('Densidade', color=SUBTEXT, fontsize=9)
    ax.set_title('Permutation Test\nDistribuição nula (H₀: sem viés) vs. SPD real',
                 fontsize=10, color=TEXT, fontweight='bold', pad=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)
    ax.text(0.98, 0.97, sig_txt, transform=ax.transAxes,
            fontsize=9.5, color=sig_c, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.45', fc=PANEL2, ec=sig_c, lw=1.5))

    if detailed:
        ax.text(0.98, 0.05,
                "Como funciona:\n"
                "• Embaralha race aleatoriamente\n"
                "  (2000×) → distribuição nula\n"
                "• Se |SPD real| está na cauda\n"
                "  vermelha → p pequeno\n"
                "  → viés não ocorre por acaso",
                transform=ax.transAxes, fontsize=8.5,
                va='bottom', ha='right', color=SUBTEXT,
                bbox=dict(boxstyle='round,pad=0.5', fc=PANEL2,
                          ec=GRID, lw=1.0, alpha=0.95))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print('\n' + '═'*62)
    print('  Extended Analysis — COMPAS  (T5 · T6 · T7)')
    print('═'*62)

    data = load_compas()
    TARGET    = 'two_year_recid'
    SENSITIVE = 'race'

    X_raw = data.drop(columns=[TARGET])
    y     = data[TARGET].values
    s     = data[SENSITIVE].values   # 0=African-Am, 1=Caucasian

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    os.makedirs('plots/compas', exist_ok=True)
    results = {}

    # ═══════════════════════════════════════════════════════════
    # T5 — REPRESENTAÇÃO NO ESPAÇO DE FEATURES
    # ═══════════════════════════════════════════════════════════
    section("T5 — Representação no Espaço de Features")

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    bhat = {}
    for col in numeric_cols:
        a = X_raw[col][s == 0].dropna().values.astype(float)
        b = X_raw[col][s == 1].dropna().values.astype(float)
        if len(a) > 5 and len(b) > 5 and a.std() > 0 and b.std() > 0:
            bhat[col] = bhattacharyya_overlap(a, b)

    bhat_series  = pd.Series(bhat).sort_values()
    low_overlap  = (bhat_series < 0.85).sum()
    mean_overlap = bhat_series.mean()

    print(f"\n  Bhattacharyya overlap (médio): {mean_overlap:.4f}")
    print(f"  Features com overlap < 0.85: {low_overlap}/{len(bhat_series)}")
    for col, val in bhat_series.items():
        flag = " ⚠" if val < 0.85 else ""
        print(f"    {col:30s}: {val:.4f}{flag}")

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_
    c0 = X_pca[s == 0].mean(axis=0)
    c1 = X_pca[s == 1].mean(axis=0)
    centroid_dist = np.linalg.norm(c0 - c1)
    print(f"\n  PCA — PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%")
    print(f"  Distância entre centroides: {centroid_dist:.4f}")

    results['T5_mean_overlap']       = mean_overlap
    results['T5_low_overlap']        = low_overlap
    results['T5_pca_centroid_dist']  = float(centroid_dist)

    # ═══════════════════════════════════════════════════════════
    # T6 — ESTABILIDADE DA MÉTRICA DE FAIRNESS
    # ═══════════════════════════════════════════════════════════
    section("T6 — Estabilidade da Métrica de Fairness")

    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_cv  = LogisticRegression(random_state=42, max_iter=1000)
    y_prob_cv = cross_val_predict(model_cv, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred_cv = (y_prob_cv >= 0.5).astype(int)

    obs_spd = spd_from_pred(y_pred_cv, s)
    obs_di  = di_from_pred(y_pred_cv, s)
    print(f"\n  Métricas observadas (LR, CV 5-fold):")
    print(f"    SPD = {obs_spd:+.4f}  (0 = justo)")
    print(f"    DI  = {obs_di:.4f}   (1 = justo)")

    # Bootstrap
    B   = 2000
    rng = np.random.default_rng(42)
    n   = len(y_pred_cv)
    boot_spd = np.array([spd_from_pred(y_pred_cv[idx := rng.integers(0, n, n)], s[idx]) for _ in range(B)])
    boot_di  = np.array([di_from_pred(y_pred_cv[idx := rng.integers(0, n, n)], s[idx]) for _ in range(B)])

    ci_spd = np.percentile(boot_spd, [2.5, 97.5])
    ci_di  = np.percentile(boot_di,  [2.5, 97.5])
    ci_crosses_zero  = ci_spd[0] <= 0 <= ci_spd[1]
    ci_di_includes_1 = ci_di[0] <= 1.0 <= ci_di[1]

    print(f"\n  Bootstrap 95% CI (B={B}):")
    print(f"    SPD: [{ci_spd[0]:+.4f}, {ci_spd[1]:+.4f}]  {'✔ Robusto' if not ci_crosses_zero else '⚠ Instável'}")
    print(f"    DI : [{ci_di[0]:.4f},  {ci_di[1]:.4f}]   {'✔ Significativo' if not ci_di_includes_1 else '⚠ Inclui 1'}")

    # Permutation test
    B_perm   = 2000
    perm_spd = np.array([spd_from_pred(y_pred_cv, rng.permutation(s)) for _ in range(B_perm)])
    p_perm   = (np.abs(perm_spd) >= np.abs(obs_spd)).mean()

    print(f"\n  Permutation test (B={B_perm}):")
    print(f"    p-valor = {p_perm:.4f}  {'✔ Viés real (p<0.05)' if p_perm < 0.05 else '✗ Pode ser aleatório'}")

    results.update({
        'T6_obs_spd': obs_spd, 'T6_obs_di': obs_di,
        'T6_ci_spd': ci_spd, 'T6_ci_di': ci_di,
        'T6_robust': not ci_crosses_zero, 'T6_p_perm': p_perm,
        'T6_boot_spd': boot_spd, 'T6_boot_di': boot_di,
        'T6_perm_spd': perm_spd,
    })

    # ═══════════════════════════════════════════════════════════
    # T7 — CALIBRAÇÃO POR GRUPO
    # ═══════════════════════════════════════════════════════════
    section("T7 — Calibração por Grupo")

    models_t7 = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest':       RandomForestClassifier(random_state=42, max_depth=7, n_estimators=100),
    }
    cal_results = {}
    print()
    for mname, clf in models_t7.items():
        probs = cross_val_predict(clf, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        ece_g0 = ece(y[s == 0], probs[s == 0])
        ece_g1 = ece(y[s == 1], probs[s == 1])
        frac0, mean0 = calibration_curve(y[s == 0], probs[s == 0], n_bins=8, strategy='quantile')
        frac1, mean1 = calibration_curve(y[s == 1], probs[s == 1], n_bins=8, strategy='quantile')
        gap = abs(ece_g0 - ece_g1)
        cal_results[mname] = dict(ece_g0=ece_g0, ece_g1=ece_g1, ece_gap=gap,
                                  frac0=frac0, mean0=mean0, frac1=frac1, mean1=mean1)
        print(f"  {mname:22s}  ECE Afr-Am={ece_g0:.4f} | ECE Caucas={ece_g1:.4f} | GAP={gap:.4f}  {'⚠' if gap>0.02 else '✔'}")

    results['T7_cal'] = cal_results

    # ═══════════════════════════════════════════════════════════
    # VEREDICTO
    # ═══════════════════════════════════════════════════════════
    section("VEREDICTO ESTENDIDO (T5 + T6 + T7)")

    criteria = {
        "T5 Overlap médio < 0.90 (grupos distintos no feature space)": mean_overlap < 0.90,
        "T5 Features com overlap < 0.85 existem":                       low_overlap > 0,
        "T6 SPD robusto ao bootstrap (CI não cruza 0)":                 results['T6_robust'],
        "T6 Viés significativo no permutation test (p < 0.05)":         p_perm < 0.05,
        "T7 ECE gap > 0.02 (descalibração diferencial por grupo)":
            any(v['ece_gap'] > 0.02 for v in cal_results.values()),
    }

    n_ok = sum(criteria.values())
    print()
    for c, ok in criteria.items():
        print(f"  {'✔' if ok else '✗'} {c}")

    verdict = "JUSTIFICADO" if n_ok >= 3 else ("PARCIALMENTE" if n_ok >= 2 else "NÃO JUSTIFICADO")
    print(f"\n  {n_ok}/5 critérios atendidos → {verdict}")
    if n_ok >= 3:
        print("  → O viés é real, robusto e estrutural.")
        print("    Dados sintéticos de fairness (S3-S5) são metodologicamente justificados.")

    # ═══════════════════════════════════════════════════════════
    # FIGURA 1 — COMPOSTA (T5 + T6 resumido + T7)
    # ═══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    fig.suptitle('Extended Sufficiency Analysis — COMPAS\nT5: Feature Space · T6: Fairness Stability · T7: Group Calibration',
                 fontsize=16, fontweight='bold', color=TEXT, y=0.99)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38,
                           left=0.06, right=0.97, top=0.93, bottom=0.05)

    # T5a barplot
    ax = fig.add_subplot(gs[0, :2])
    cols_s = bhat_series.index.tolist()
    vals_s = bhat_series.values
    ax.barh(range(len(cols_s)), vals_s,
            color=[RED if v < 0.85 else GREEN for v in vals_s],
            edgecolor=GRID, linewidth=0.8, height=0.65)
    ax.set_yticks(range(len(cols_s)))
    ax.set_yticklabels(cols_s, fontsize=8.5)
    ax.axvline(0.85, color=GOLD, ls='--', lw=1.5, label='Limiar 0.85')
    ax.axvline(mean_overlap, color=TEAL, ls=':', lw=1.8, label=f'Média = {mean_overlap:.3f}')
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Bhattacharyya Overlap', color=SUBTEXT, fontsize=9)
    ax.set_title('T5a — Sobreposição de Distribuição por Feature\n(African-Am vs Caucasian)',
                 fontsize=10, color=TEXT, fontweight='bold', pad=8)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    for i, (v, col) in enumerate(zip(vals_s, [RED if v < 0.85 else GREEN for v in vals_s])):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=7.5, color=col, fontweight='bold')

    # T5b PCA
    ax2 = fig.add_subplot(gs[0, 2:])
    for grp, color, lbl in [(0, C0, 'African-Am'), (1, C1, 'Caucasian')]:
        mask = s == grp
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, alpha=0.22, s=12,
                    edgecolors='none', label=lbl)
        try:
            xi, yi = np.mgrid[X_pca[:,0].min():X_pca[:,0].max():60j,
                               X_pca[:,1].min():X_pca[:,1].max():60j]
            zi = gaussian_kde(X_pca[mask].T)(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
            ax2.contour(xi, yi, zi, levels=4, colors=color, alpha=0.55, linewidths=0.9)
        except Exception:
            pass
    ax2.scatter(*c0, s=180, marker='X', c=C0, edgecolors='white', lw=1.5, zorder=5)
    ax2.scatter(*c1, s=180, marker='X', c=C1, edgecolors='white', lw=1.5, zorder=5)
    ax2.annotate('', xy=c1, xytext=c0,
                 arrowprops=dict(arrowstyle='<->', color=GOLD, lw=1.8))
    mid = (c0 + c1) / 2
    ax2.text(mid[0], mid[1] + 0.3, f'd={centroid_dist:.3f}',
             ha='center', color=GOLD, fontsize=9, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}%)', color=SUBTEXT, fontsize=9)
    ax2.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}%)', color=SUBTEXT, fontsize=9)
    ax2.set_title('T5b — PCA Biplot: Separação dos Grupos', fontsize=10,
                  color=TEXT, fontweight='bold', pad=8)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # T6 — Forest plot (composto)
    plot_t6_forest(fig.add_subplot(gs[1, :2]), obs_spd, ci_spd, obs_di, ci_di)

    # T6 — Permutation test (composto)
    plot_t6_permutation(fig.add_subplot(gs[1, 2:]), perm_spd, obs_spd, p_perm, B_perm)

    # T7 — Calibração por grupo
    model_names = list(cal_results.keys())
    for i, mname in enumerate(model_names):
        cr  = cal_results[mname]
        ax5 = fig.add_subplot(gs[2, i*2:(i*2)+2])
        ax5.plot([0, 1], [0, 1], 'w--', lw=1.2, alpha=0.4, label='Perfeito')
        ax5.plot(cr['mean0'], cr['frac0'], 'o-', color=C0, lw=2, ms=7,
                 label=f"African-Am  ECE={cr['ece_g0']:.4f}")
        ax5.plot(cr['mean1'], cr['frac1'], 's-', color=C1, lw=2, ms=7,
                 label=f"Caucasian   ECE={cr['ece_g1']:.4f}")
        try:
            xs_c = np.linspace(0.1, 0.9, 200)
            f0 = interp1d(cr['mean0'], cr['frac0'], bounds_error=False, fill_value='extrapolate')
            f1 = interp1d(cr['mean1'], cr['frac1'], bounds_error=False, fill_value='extrapolate')
            ax5.fill_between(xs_c, f0(xs_c), f1(xs_c), alpha=0.12, color=GOLD,
                             label=f"Gap ECE = {cr['ece_gap']:.4f}")
        except Exception:
            pass
        ax5.set_xlim(-0.02, 1.02); ax5.set_ylim(-0.02, 1.02)
        ax5.set_xlabel('Probabilidade Predita', color=SUBTEXT, fontsize=9)
        ax5.set_ylabel('Fração Positivos Reais', color=SUBTEXT, fontsize=9)
        flag = "⚠ Descalibração diferencial" if cr['ece_gap'] > 0.02 else "✔ Calibração equivalente"
        ax5.set_title(f"T7 — Calibração: {mname}\n{flag}", fontsize=10,
                      color=TEXT, fontweight='bold', pad=8)
        ax5.legend(fontsize=8)
        ax5.grid(alpha=0.25)

    verdict_color = GREEN if n_ok >= 3 else (GOLD if n_ok >= 2 else RED)
    fig.text(0.5, 0.002, f"VEREDICTO: {verdict}  ·  {n_ok}/5 critérios T5-T7 atendidos",
             ha='center', fontsize=12, fontweight='bold', color=verdict_color,
             bbox=dict(boxstyle='round,pad=0.4', fc=BG, ec=verdict_color, lw=1.5))

    p1 = 'plots/compas/extended_sufficiency_analysis.png'
    fig.savefig(p1, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"\n  → Figura composta salva em: {p1}")

    # ═══════════════════════════════════════════════════════════
    # FIGURA 2 — T6 DEDICADA (4 painéis didáticos detalhados)
    # ═══════════════════════════════════════════════════════════
    fig6 = plt.figure(figsize=(18, 12), facecolor=BG)
    fig6.suptitle(
        'T6 — Fairness Stability Analysis · COMPAS\n'
        'Logistic Regression · CV 5-fold · Bootstrap B=2000 · Permutation B=2000',
        fontsize=15, fontweight='bold', color=TEXT, y=0.99)
    gs6 = gridspec.GridSpec(2, 2, figure=fig6, hspace=0.50, wspace=0.40,
                            left=0.07, right=0.97, top=0.92, bottom=0.07)

    plot_t6_forest(fig6.add_subplot(gs6[0, 0]),
                   obs_spd, ci_spd, obs_di, ci_di, detailed=True)

    plot_t6_boot_hist(fig6.add_subplot(gs6[0, 1]),
                      boot_spd, obs_spd, ci_spd,
                      metric='Statistical Parity Difference (SPD)',
                      fair_val=0.0, fair_label='SPD = 0  (equidade perfeita)',
                      color=TEAL)

    plot_t6_boot_hist(fig6.add_subplot(gs6[1, 0]),
                      boot_di, obs_di, ci_di,
                      metric='Disparate Impact (DI)',
                      fair_val=1.0, fair_label='DI = 1  (equidade perfeita)',
                      color=C0)

    plot_t6_permutation(fig6.add_subplot(gs6[1, 1]),
                        perm_spd, obs_spd, p_perm, B_perm, detailed=True)

    p2 = 'plots/compas/t6_fairness_stability.png'
    fig6.savefig(p2, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig6)
    print(f"  → Figura T6 dedicada salva em:  {p2}\n")
    print('═'*62 + '\n')


if __name__ == '__main__':
    main()
