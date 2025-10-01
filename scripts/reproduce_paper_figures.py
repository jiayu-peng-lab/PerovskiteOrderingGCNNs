#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.decomposition import PCA

# Ensure figures directory exists
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

BEST_BASE = "./best_models"
MODELS = ["CGCNN", "e3nn", "ALIGNN"]
STRUCTS = ["unrelaxed", "relaxed"]
TARGET_PROP = "dft_e_hull"
PRED_COL = f"predicted_{TARGET_PROP}"

# Optional imports for embeddings
try:
    from inference.plot_utils import plot_pca_embedding
    PLOT_UTILS_AVAILABLE = True
except Exception:
    PLOT_UTILS_AVAILABLE = False

# Optional sinaplot import
try:
    from inference.sina_plot import sinaplot
    SINAPLOT_AVAILABLE = True
except Exception:
    SINAPLOT_AVAILABLE = False


def coerce_pred_array(arr_like):
    arr = np.asarray(arr_like, dtype=object)
    if len(arr) > 0 and isinstance(arr[0], (list, tuple, np.ndarray)):
        arr = np.array([x[0] if isinstance(x, (list, tuple, np.ndarray)) else x for x in arr], dtype=object)
    ser = pd.to_numeric(pd.Series(arr, dtype=object), errors="coerce")
    return ser.values


def _scan_best_dirs(model_root: str):
    best_dirs = {}
    if not os.path.exists(model_root):
        return best_dirs
    for root, dirs, files in os.walk(model_root):
        for k in [0, 1, 2]:
            dname = f"best_{k}"
            if dname in dirs:
                full = os.path.join(root, dname)
                best_dirs[k] = full
    return best_dirs  # map k->path


def find_model_path(model: str, struct: str):
    # Return model root (not best_*) and let scanners find best_* anywhere below
    base_path = os.path.join(BEST_BASE, model, f"dft_e_hull_htvs_data_{struct}_{model}")
    return base_path if os.path.exists(base_path) else None


def load_predictions(model: str, struct: str, split: str):
    model_root = find_model_path(model, struct)
    if model_root is None:
        raise FileNotFoundError(f"No directory found for {model}-{struct}")
    best = _scan_best_dirs(model_root)
    if not best:
        raise FileNotFoundError(f"No best_* dirs found under {model_root}")
    # Prefer best_0 if available
    folder = best.get(0) or list(best.values())[0]
    path = os.path.join(folder, f"{split}_predictions.json")
    if not os.path.exists(path):
        # try any best_k that has the file
        found = None
        for k, d in best.items():
            p = os.path.join(d, f"{split}_predictions.json")
            if os.path.exists(p):
                found = p
                break
        if not found:
            raise FileNotFoundError(f"Missing predictions file for {model}-{struct}-{split}")
        path = found
    df = pd.read_json(path)
    if PRED_COL in df.columns:
        df[PRED_COL] = coerce_pred_array(df[PRED_COL].values)
    else:
        raise KeyError(f"Column {PRED_COL} not found in {path}")
    if TARGET_PROP in df.columns:
        df[TARGET_PROP] = pd.to_numeric(df[TARGET_PROP], errors="coerce")
    else:
        raise KeyError(f"Column {TARGET_PROP} not found in {path}")
    df = df.dropna(subset=[TARGET_PROP, PRED_COL])
    return df


def load_predictions_ensemble(model: str, struct: str, split: str):
    model_root = find_model_path(model, struct)
    if model_root is None:
        raise FileNotFoundError(f"No directory found for {model}-{struct}")
    best = _scan_best_dirs(model_root)
    if not best:
        raise FileNotFoundError(f"No best_* dirs for {model}-{struct}")
    dfs = []
    for k in [0, 1, 2]:
        if k not in best:
            continue
        path = os.path.join(best[k], f"{split}_predictions.json")
        if not os.path.exists(path):
            continue
        dfk = pd.read_json(path)
        if PRED_COL not in dfk.columns or TARGET_PROP not in dfk.columns:
            continue
        dfk[PRED_COL] = coerce_pred_array(dfk[PRED_COL].values)
        dfk[TARGET_PROP] = pd.to_numeric(dfk[TARGET_PROP], errors="coerce")
        dfs.append(dfk[["formula", TARGET_PROP, PRED_COL]].copy())
    if not dfs:
        raise FileNotFoundError(f"No ensemble files for {model}-{struct}-{split}")
    ref = dfs[0].copy()
    preds = [d[PRED_COL].values.astype(float) for d in dfs]
    pred_mean = np.mean(np.vstack(preds), axis=0)
    out = ref.copy()
    out[PRED_COL] = pred_mean
    out = out.dropna(subset=[TARGET_PROP, PRED_COL])
    return out


def find_model_path_with_suffix(model: str, struct: str, suffix: str | None):
    base_name = f"dft_e_hull_htvs_data_{struct}_{model}"
    if suffix:
        base_name = base_name + suffix
    base_path = os.path.join(BEST_BASE, model, base_name)
    return base_path if os.path.exists(base_path) else None


def load_predictions_ensemble_with_suffix(model: str, struct: str, split: str, suffix: str | None):
    model_root = find_model_path_with_suffix(model, struct, suffix)
    if model_root is None:
        raise FileNotFoundError(f"No directory found for {model}-{struct}{suffix or ''}")
    best = _scan_best_dirs(model_root)
    if not best:
        raise FileNotFoundError(f"No best_* dirs for {model}-{struct}{suffix or ''}")
    preds, trues = [], []
    for k in [0, 1, 2]:
        if k not in best:
            continue
        path = os.path.join(best[k], f"{split}_predictions.json")
        if not os.path.exists(path):
            continue
        dfk = pd.read_json(path)
        if PRED_COL not in dfk.columns or TARGET_PROP not in dfk.columns:
            continue
        dfk[PRED_COL] = coerce_pred_array(dfk[PRED_COL].values)
        dfk[TARGET_PROP] = pd.to_numeric(dfk[TARGET_PROP], errors="coerce")
        preds.append(dfk[PRED_COL].values.astype(float))
        trues.append(dfk[TARGET_PROP].values.astype(float))
    if not preds:
        raise FileNotFoundError(f"No ensemble files for {model}-{struct}{suffix or ''}-{split}")
    y_pred = np.mean(np.vstack(preds), axis=0)
    y_true = trues[0]
    return y_true, y_pred


def _load_series_ensemble(model: str, struct: str):
    model_root = find_model_path(model, struct)
    if model_root is None:
        raise FileNotFoundError(f"No path for {model}-{struct}")
    best = _scan_best_dirs(model_root)
    dfs = []
    for k, folder in best.items():
        f = os.path.join(folder, "holdout_set_series_predictions.json")
        if os.path.exists(f):
            dfk = pd.read_json(f)
            if PRED_COL in dfk.columns and TARGET_PROP in dfk.columns:
                dfk[PRED_COL] = coerce_pred_array(dfk[PRED_COL].values)
                dfk[TARGET_PROP] = pd.to_numeric(dfk[TARGET_PROP], errors='coerce')
                dfs.append(dfk)
    if not dfs:
        raise FileNotFoundError("No series predictions found")
    df = dfs[0][['framework', 'formula', TARGET_PROP, PRED_COL]].copy()
    if len(dfs) > 1:
        preds = [d.set_index('framework')[PRED_COL] for d in dfs]
        merged = pd.concat(preds, axis=1).mean(axis=1)
        df = df.set_index('framework')
        df[PRED_COL] = merged
        df = df.reset_index()
    return df


def parity_plot(df: pd.DataFrame, model: str, struct: str, split: str):
    y_true = df[TARGET_PROP].values.astype(float)
    y_pred = df[PRED_COL].values.astype(float)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    denom = np.var(y_true)
    r2 = float(1.0 - np.mean((y_pred - y_true) ** 2) / denom) if denom > 0 else np.nan
    lim_min = float(min(y_true.min(), y_pred.min()))
    lim_max = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=y_true, y=y_pred, s=12, alpha=0.5, edgecolor=None)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
    plt.xlabel("DFT E_hull (eV/atom)")
    plt.ylabel("Predicted E_hull (eV/atom)")
    plt.title(f"{model} ({struct}, {split})")
    plt.text(0.02, 0.98, f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR2={r2:.3f}",
             transform=plt.gca().transAxes, va='top', ha='left', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    out = os.path.join(FIG_DIR, f"parity_{model}_{struct}_{split}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return {"mae": mae, "rmse": rmse, "r2": r2, "file": out}


def compute_relative_per_formula(df: pd.DataFrame, values: np.ndarray):
    if 'formula' not in df.columns:
        raise KeyError("Expected 'formula' column for relative computation")
    s = pd.Series(values, index=df.index)
    tmp = pd.DataFrame({"formula": df['formula'], "val": s})
    val_min = tmp.groupby('formula')['val'].transform('min')
    rel = s.values - val_min.values
    return rel


def relative_holdout_plot(df: pd.DataFrame, model: str, struct: str):
    if 'formula' not in df.columns:
        raise KeyError("Expected 'formula' column in holdout predictions")
    y_pred = df[PRED_COL].values.astype(float)
    y_true = df[TARGET_PROP].values.astype(float)
    pred_rel = compute_relative_per_formula(df, y_pred)
    true_rel = compute_relative_per_formula(df, y_true)
    lim_min = float(min(true_rel.min(), pred_rel.min()))
    lim_max = float(max(true_rel.max(), pred_rel.max()))
    plt.figure(figsize=(5, 5))
    sns.scatterplot(x=true_rel, y=pred_rel, s=12, alpha=0.5, edgecolor=None)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
    plt.xlabel("DFT ΔE_hull rel. GS (eV/atom)")
    plt.ylabel("Pred ΔE_hull rel. GS (eV/atom)")
    plt.title(f"{model} ({struct}, holdout)")
    out = os.path.join(FIG_DIR, f"relative_holdout_{model}_{struct}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return {"file": out}


def figure3_relative_orderings():
    # Include all three models: CGCNN, e3nn, and ALIGNN
    cases = [
        ("CGCNN", "unrelaxed"), ("e3nn", "unrelaxed"), ("ALIGNN", "unrelaxed"),
        ("CGCNN", "relaxed"), ("e3nn", "relaxed"), ("ALIGNN", "relaxed")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics_rows = []
    
    for i, (model, struct) in enumerate(cases):
        row = i // 3  # 0 for unrelaxed, 1 for relaxed
        col = i % 3   # 0 for CGCNN, 1 for e3nn, 2 for ALIGNN
        ax = axes[row, col]
        
        try:
            df = load_predictions_ensemble(model, struct, split="holdout_set_B_sites")
            y_pred = df[PRED_COL].values.astype(float)
            y_true = df[TARGET_PROP].values.astype(float)
            pred_rel = compute_relative_per_formula(df, y_pred)
            true_rel = compute_relative_per_formula(df, y_true)
            mae = float(np.mean(np.abs(pred_rel - true_rel)))
            rmse = float(np.sqrt(np.mean((pred_rel - true_rel) ** 2)))
            denom = np.var(true_rel)
            r2 = float(1.0 - np.mean((pred_rel - true_rel) ** 2) / denom) if denom > 0 else np.nan
            lim_min = float(min(true_rel.min(), pred_rel.min()))
            lim_max = float(max(true_rel.max(), pred_rel.max()))
            sns.scatterplot(ax=ax, x=true_rel, y=pred_rel, s=12, alpha=0.6, edgecolor=None)
            ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
            ax.set_xlabel("DFT ΔE_hull rel. GS (eV/atom)")
            ax.set_ylabel("Pred ΔE_hull rel. GS (eV/atom)")
            ax.set_title(f"{model} ({struct})")
            ax.text(0.02, 0.98, f"MAE={mae:.3f}\nR2={r2:.3f}", transform=ax.transAxes,
                    va='top', ha='left', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))
            metrics_rows.append({"model": model, "struct": struct, "MAE": mae, "R2": r2, "RMSE": rmse})
        except Exception as e:
            ax.set_title(f"{model} ({struct}) - missing")
            print(f"Figure 3 panel skip {model}-{struct}: {e}")
    
    # Add row labels
    fig.text(0.02, 0.75, 'Unrelaxed', rotation=90, fontsize=14, fontweight='bold', ha='center', va='center')
    fig.text(0.02, 0.25, 'Relaxed', rotation=90, fontsize=14, fontweight='bold', ha='center', va='center')
    
    plt.tight_layout()
    out_panels = os.path.join(FIG_DIR, "figure3_relative_orderings_panels.png")
    plt.savefig(out_panels, dpi=300)
    plt.close(fig)

    # Panel f: metrics summary
    if metrics_rows:
        dfm = pd.DataFrame(metrics_rows)
        plt.figure(figsize=(8, 4))
        dfm['label'] = dfm['model'] + " (" + dfm['struct'] + ")"
        # Two subplots side by side: MAE and R2
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        sns.barplot(ax=ax1, data=dfm, x='label', y='MAE', color='#4C78A8')
        ax1.set_ylabel('MAE (eV/atom)')
        ax1.set_xlabel('')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right')
        sns.barplot(ax=ax2, data=dfm, x='label', y='R2', color='#F58518')
        ax2.set_ylabel('R2 score')
        ax2.set_xlabel('')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')
        plt.tight_layout()
        out_metrics = os.path.join(FIG_DIR, "figure3_relative_orderings_metrics.png")
        fig2.savefig(out_metrics, dpi=300)
        plt.close(fig2)
        print(f"Saved: {out_panels}, {out_metrics}")


def embedding_panels_and_spread():
    if not PLOT_UTILS_AVAILABLE:
        print("plot_utils.plot_pca_embedding not available; skipping embedding panels.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    spreads = []
    cfg = [("CGCNN", "unrelaxed", axes[0, 0]), ("e3nn", "unrelaxed", axes[1, 0]), ("CGCNN", "relaxed", axes[0, 1]), ("e3nn", "relaxed", axes[1, 1])]
    for model, struct, ax in cfg:
        try:
            spread = plot_pca_embedding(ax, model, struct, highlight=False)
            spreads.append({"model": model, "struct": struct, "spread": spread})
            ax.set_title(f"{model} ({struct})", fontsize=11)
        except Exception as e:
            ax.set_title(f"{model} ({struct}) - failed")
            print(f"Skip PCA panel {model}-{struct}: {e}")
    plt.tight_layout()
    out_panels = os.path.join(FIG_DIR, "embeddings_panels.png")
    plt.savefig(out_panels, dpi=300)
    plt.close(fig)
    rows = []
    for item in spreads:
        for val in item["spread"]:
            rows.append({"model": item["model"], "struct": item["struct"], "spread": val})
    if rows:
        df_spread = pd.DataFrame(rows)
        plt.figure(figsize=(6, 5))
        sns.violinplot(data=df_spread, x="spread", y="model", hue="struct", split=True, inner="point")
        plt.xlabel("Embedding spread across orderings in PCA")
        plt.ylabel("")
        out_violin = os.path.join(FIG_DIR, "embedding_spread_violin.png")
        plt.tight_layout()
        plt.savefig(out_violin, dpi=300)
        plt.close()
        print(f"Saved: {out_panels}, {out_violin}")


def _get_holdout_relatives_for(model: str, struct: str):
    df = load_predictions_ensemble(model, struct, split="holdout_set_B_sites")
    y_true = df[TARGET_PROP].values.astype(float)
    y_pred = df[PRED_COL].values.astype(float)
    dft_rel = compute_relative_per_formula(df, y_true)
    pred_rel = compute_relative_per_formula(df, y_pred)
    return dft_rel, pred_rel


def figure3_hexbin_composite():
    try:
        # Load data (eV)
        dft_cg_u, mdl_cg_u = _get_holdout_relatives_for("CGCNN", "unrelaxed")
        dft_e3_u, mdl_e3_u = _get_holdout_relatives_for("e3nn", "unrelaxed")
        dft_al_u, mdl_al_u = _get_holdout_relatives_for("ALIGNN", "unrelaxed")
        dft_cg_r, mdl_cg_r = _get_holdout_relatives_for("CGCNN", "relaxed")
        dft_e3_r, mdl_e3_r = _get_holdout_relatives_for("e3nn", "relaxed")
        dft_al_r, mdl_al_r = _get_holdout_relatives_for("ALIGNN", "relaxed")
    except Exception as e:
        print(f"Skipping hexbin composite due to data error: {e}")
        return

    # Convert to meV/atom
    to_mev = 1000.0
    diffs = dict(
        dft_u=dft_cg_u*to_mev,
        cg_u=mdl_cg_u*to_mev,
        e3_u=mdl_e3_u*to_mev,
        al_u=mdl_al_u*to_mev,
        dft_r=dft_cg_r*to_mev,
        cg_r=mdl_cg_r*to_mev,
        e3_r=mdl_e3_r*to_mev,
        al_r=mdl_al_r*to_mev,
    )

    fig = plt.figure(figsize=(19, 7), constrained_layout=True)
    subfig_l, subfig_r = fig.subfigures(nrows=1, ncols=2, width_ratios=[2.6, 1])

    axes_l = subfig_l.subplots(nrows=2, ncols=3, sharex=True, sharey=True, gridspec_kw={'left': 0.3})
    hex_cmap = 'inferno_r'
    hex_gridsize = 30
    hex_mincnt = 1
    hex_edgecolors = 'black'
    hex_linewidths = 0.5
    hex_xylim = [-40, 190]  # meV
    cbar_vmax = 22

    for ax in axes_l.flatten():
        ax.axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), color='black', linestyle='--', linewidth=2)

    # Unrelaxed row
    axes_l[0][0].hexbin(diffs['dft_u'], diffs['cg_u'], cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt,
                        edgecolors=hex_edgecolors, linewidths=hex_linewidths, extent=hex_xylim + hex_xylim,
                        vmin=0, vmax=cbar_vmax)
    axes_l[0][1].hexbin(diffs['dft_u'], diffs['e3_u'], cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt,
                        edgecolors=hex_edgecolors, linewidths=hex_linewidths, extent=hex_xylim + hex_xylim,
                        vmin=0, vmax=cbar_vmax)
    axes_l[0][2].hexbin(diffs['dft_u'], diffs['al_u'], cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt,
                        edgecolors=hex_edgecolors, linewidths=hex_linewidths, extent=hex_xylim + hex_xylim,
                        vmin=0, vmax=cbar_vmax)

    # Relaxed row
    axes_l[1][0].hexbin(diffs['dft_r'], diffs['cg_r'], cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt,
                        edgecolors=hex_edgecolors, linewidths=hex_linewidths, extent=hex_xylim + hex_xylim,
                        vmin=0, vmax=cbar_vmax)
    axes_l[1][1].hexbin(diffs['dft_r'], diffs['e3_r'], cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt,
                        edgecolors=hex_edgecolors, linewidths=hex_linewidths, extent=hex_xylim + hex_xylim,
                        vmin=0, vmax=cbar_vmax)
    hex_example = axes_l[1][2].hexbin(diffs['dft_r'], diffs['al_r'], cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt,
                                      edgecolors=hex_edgecolors, linewidths=hex_linewidths, extent=hex_xylim + hex_xylim,
                                      vmin=0, vmax=cbar_vmax)

    # Labels/colors
    axes_l[0][0].text(0.96, 0.05, 'CGCNN\n(unrelaxed)', ha='right', fontsize=16, transform=axes_l[0][0].transAxes, color='cornflowerblue')
    axes_l[0][1].text(0.96, 0.05, 'e3nn\n(unrelaxed)', ha='right', fontsize=16, transform=axes_l[0][1].transAxes, color='orchid')
    axes_l[0][2].text(0.96, 0.05, 'ALIGNN\n(unrelaxed)', ha='right', fontsize=16, transform=axes_l[0][2].transAxes, color='teal')
    axes_l[1][0].text(0.96, 0.05, 'CGCNN\n(relaxed)', ha='right', fontsize=16, transform=axes_l[1][0].transAxes, color='darkblue')
    axes_l[1][1].text(0.96, 0.05, 'e3nn\n(relaxed)', ha='right', fontsize=16, transform=axes_l[1][1].transAxes, color='darkmagenta')
    axes_l[1][2].text(0.96, 0.05, 'ALIGNN\n(relaxed)', ha='right', fontsize=16, transform=axes_l[1][2].transAxes, color='darkcyan')

    axes_l[0][0].set_yticks(np.arange(0, hex_xylim[1], 50))
    axes_l[0][0].set_xticks(np.arange(0, hex_xylim[1], 50))
    subfig_l.colorbar(hex_example, ax=axes_l, label='Count', ticks=np.arange(0, cbar_vmax+1, 5), aspect=40)
    subfig_l.supxlabel('DFT $\\mathit{\\Delta E}_{\\mathrm{hull}}$ (meV/atom vs. ground-state ordering)', x=0.49, fontsize=16)
    subfig_l.supylabel('ML $\\mathit{\\Delta E}_{\\mathrm{hull}}$ (meV/atom vs. ground-state ordering)', y=0.55, fontsize=16)

    # Right metrics panel (unchanged)
    ax_r = subfig_r.subplots(nrows=1, ncols=1)
    ax_r.get_yaxis().set_visible(False)
    ax_r.set_xlabel('Holdout set MAE (meV/atom)', labelpad=9)
    ax_r.set_xlim(7.5, 24)
    ax_r.set_ylim(-7, 1)
    ax_r.set_xticks(np.arange(10, 21, 5))

    from sklearn.metrics import mean_absolute_error, r2_score
    mae_vals = [
        mean_absolute_error(diffs['dft_u'], diffs['cg_u']),
        mean_absolute_error(diffs['dft_r'], diffs['cg_r']),
        mean_absolute_error(diffs['dft_u'], diffs['e3_u']),
        mean_absolute_error(diffs['dft_r'], diffs['e3_r']),
        mean_absolute_error(diffs['dft_u'], diffs['al_u']),
        mean_absolute_error(diffs['dft_r'], diffs['al_r']),
    ]
    colors = ['cornflowerblue', 'darkblue', 'orchid', 'darkmagenta', 'teal', 'darkcyan']
    ys = [-0.5, -1.5, -2.5, -3.5, -4.5, -5.5]
    labels_left = ['CGCNN\n(unrelaxed)', 'CGCNN\n(relaxed)', 'e3nn\n(unrelaxed)', 'e3nn\n(relaxed)', 'ALIGNN\n(unrelaxed)', 'ALIGNN\n(relaxed)']
    for y, w, c in zip(ys, mae_vals, colors):
        ax_r.barh(y=y, width=w, height=0.4, color=c)

    ax_r_twiny = ax_r.twiny()
    ax_r_twiny.set_xlabel('Holdout set $\\mathit{R}^2$ score', labelpad=9)
    ax_r_twiny.set_xlim(-0.3, 0.9)
    ax_r_twiny.set_xticks(np.arange(-0.2, 0.9, 0.2))

    r2_vals = [
        r2_score(diffs['dft_u'], diffs['cg_u']),
        r2_score(diffs['dft_r'], diffs['cg_r']),
        r2_score(diffs['dft_u'], diffs['e3_u']),
        r2_score(diffs['dft_r'], diffs['e3_r']),
        r2_score(diffs['dft_u'], diffs['al_u']),
        r2_score(diffs['dft_r'], diffs['al_r']),
    ]
    r2_y_positions = [5.5, 4.5, 3.5, 2.5, 1.5, 0.5]
    for x, y, c in zip(r2_vals, r2_y_positions, colors):
        ax_r_twiny.plot(x, y, 'o', markersize=20, markeredgecolor='black', markeredgewidth=0, color=c)

    ax_r.hlines(0, 0, 100, color='black', linestyle='-', linewidth=2)

    # Text labels
    xpos_top = [0.165, 0.390, 0.605, 0.755, 0.165, 0.390]
    ypos_top = [0.905, 0.780, 0.655, 0.530, 0.405, 0.280]
    for (xt, yt, lab, col) in zip(xpos_top, ypos_top, labels_left, colors):
        ax_r.text(xt, yt, lab, color=col, fontsize=16, ha='left', transform=ax_r.transAxes)

    xpos_bot = [0.700, 0.620, 0.415, 0.290, 0.700, 0.620]
    ypos_bot = [0.410, 0.285, 0.160, 0.035, -0.090, -0.215]
    for (xt, yt, lab, col) in zip(xpos_bot, ypos_bot, labels_left, colors):
        ax_r.text(xt, yt, lab, color=col, fontsize=16, ha='left', transform=ax_r.transAxes)

    out_png = os.path.join(FIG_DIR, 'Main_ordering_dependence.png')
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"Saved: {out_png}")


def _hexbin(ax, x, y, hex_xylim, cbar_axis=None, vmin=0, vmax=22):
    hb = ax.hexbin(x, y,
                   cmap='inferno_r', gridsize=30, mincnt=1,
                   edgecolors='black', linewidths=0.5,
                   extent=hex_xylim + hex_xylim,
                   vmin=vmin, vmax=vmax)
    ax.axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), color='black', linestyle='--', linewidth=2)
    return hb


def parity_hexbin(df: pd.DataFrame, model: str, struct: str, split: str):
    # Convert to meV
    y_true = (df[TARGET_PROP].values.astype(float) * 1000.0)
    y_pred = (df[PRED_COL].values.astype(float) * 1000.0)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    denom = np.var(y_true)
    r2 = float(1.0 - np.mean((y_pred - y_true) ** 2) / denom) if denom > 0 else np.nan

    hex_xylim = [-40, 190]
    fig, ax = plt.subplots(figsize=(5, 5))
    hb = _hexbin(ax, y_true, y_pred, hex_xylim)
    fig.colorbar(hb, ax=ax, label='Count', ticks=np.arange(0, 22+1, 5))
    ax.set_xlabel('DFT $\\mathit{\\Delta E}_{\\mathrm{hull}}$ (meV/atom)')
    ax.set_ylabel('ML $\\mathit{\\Delta E}_{\\mathrm{hull}}$ (meV/atom)')
    ax.set_title(f"{model} ({struct}, {split})")
    ax.text(0.02, 0.98, f"MAE={mae:.1f} meV\nRMSE={rmse:.1f} meV\nR2={r2:.3f}",
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))
    ax.set_xticks(np.arange(0, hex_xylim[1], 50))
    ax.set_yticks(np.arange(0, hex_xylim[1], 50))
    out_png = os.path.join(FIG_DIR, f"hex_parity_{model}_{struct}_{split}.png")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {"mae": mae, "rmse": rmse, "r2": r2, "file_png": out_png}


def relative_hexbin(df: pd.DataFrame, model: str, struct: str):
    if 'formula' not in df.columns:
        raise KeyError("Expected 'formula' column in holdout predictions")
    y_pred = df[PRED_COL].values.astype(float)
    y_true = df[TARGET_PROP].values.astype(float)
    pred_rel = compute_relative_per_formula(df, y_pred) * 1000.0
    true_rel = compute_relative_per_formula(df, y_true) * 1000.0

    hex_xylim = [-40, 190]
    fig, ax = plt.subplots(figsize=(5, 5))
    hb = _hexbin(ax, true_rel, pred_rel, hex_xylim)
    fig.colorbar(hb, ax=ax, label='Count', ticks=np.arange(0, 22+1, 5))
    ax.set_xlabel('DFT $\\mathit{\\Delta E}_{\\mathrm{hull}}$ rel. GS (meV/atom)')
    ax.set_ylabel('ML $\\mathit{\\Delta E}_{\\mathrm{hull}}$ rel. GS (meV/atom)')
    ax.set_title(f"{model} ({struct}, holdout)")
    ax.set_xticks(np.arange(0, hex_xylim[1], 50))
    ax.set_yticks(np.arange(0, hex_xylim[1], 50))
    out_png = os.path.join(FIG_DIR, f"hex_relative_{model}_{struct}.png")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return {"file_png": out_png}


def figure2_composition_hex_and_learning_curve():
    # Panels b–e plus ALIGNN (2x3): test-set absolute Ehull parity
    combos = [
        ("CGCNN", "unrelaxed", 'cornflowerblue'),
        ("e3nn", "unrelaxed", 'orchid'),
        ("ALIGNN", "unrelaxed", 'teal'),
        ("CGCNN", "relaxed", 'darkblue'),
        ("e3nn", "relaxed", 'darkmagenta'),
        ("ALIGNN", "relaxed", 'darkcyan'),
    ]
    fig = plt.figure(figsize=(19, 7), constrained_layout=True)
    subfig_l, subfig_r = fig.subfigures(nrows=1, ncols=2, width_ratios=[2.6, 1])
    axes = subfig_l.subplots(nrows=2, ncols=3, sharex=True, sharey=True, gridspec_kw={'left': 0.3})
    hex_xylim = [-40, 500]  # meV as shown in example
    cbar_vmax = 22
    hb_last = None
    for idx, (model, struct, color) in enumerate(combos):
        ax = axes[idx//3][idx%3]
        try:
            y_true, y_pred = load_predictions_ensemble(model, struct, split="test_set")[[TARGET_PROP, PRED_COL]]  # placeholder to trigger except
        except Exception:
            try:
                df = load_predictions(model, struct, split="test_set")
                y_true = df[TARGET_PROP].values.astype(float)
                y_pred = df[PRED_COL].values.astype(float)
            except Exception as e:
                ax.set_title(f"{model} ({struct}) - missing")
                continue
        if not isinstance(y_true, np.ndarray):
            df_tmp = load_predictions(model, struct, split="test_set")
            y_true = df_tmp[TARGET_PROP].values.astype(float)
            y_pred = df_tmp[PRED_COL].values.astype(float)
        x = y_true * 1000.0
        y = y_pred * 1000.0
        hb_last = _hexbin(ax, x, y, hex_xylim, vmin=0, vmax=cbar_vmax)
        ax.text(0.96, 0.05, f"{model}\n({struct})", ha='right', fontsize=14, transform=ax.transAxes, color=color)
    subfig_l.colorbar(hb_last, ax=axes, label='Count', ticks=np.arange(0, cbar_vmax+1, 5), aspect=40)
    subfig_l.supxlabel('DFT $\\mathit{\\Delta E}_{\\mathrm{hull}}$ (meV/atom)', x=0.49, fontsize=16)
    subfig_l.supylabel('ML $\\mathit{\\Delta E}_{\\mathrm{hull}}$ (meV/atom)', y=0.55, fontsize=16)

    # Learning curve on right: MAE vs training fraction
    axr = subfig_r.subplots(nrows=1, ncols=1)
    axr.set_xlabel('Training fraction')
    axr.set_ylabel('Test MAE (meV/atom)')
    fractions = [1.0, 0.5, 0.25, 0.125]
    frac_suffix = {1.0: None, 0.5: '_TrainingFraction0.5', 0.25: '_TrainingFraction0.25', 0.125: '_TrainingFraction0.125'}
    curve_specs = [
        ("CGCNN", "unrelaxed", 'cornflowerblue'),
        ("CGCNN", "relaxed", 'darkblue'),
        ("e3nn", "unrelaxed", 'orchid'),
        ("e3nn", "relaxed", 'darkmagenta'),
        ("ALIGNN", "unrelaxed", 'teal'),
        ("ALIGNN", "relaxed", 'darkcyan'),
    ]
    for model, struct, color in curve_specs:
        xs, ys, yerr = [], [], []
        for f in fractions:
            suff = frac_suffix[f]
            try:
                y_true, y_pred = load_predictions_ensemble_with_suffix(model, struct, split="test_set", suffix=suff)
            except Exception:
                try:
                    base_path = find_model_path_with_suffix(model, struct, suff)
                    if base_path is None:
                        continue
                    dfk = pd.read_json(os.path.join(base_path, 'best_0', 'test_set_predictions.json'))
                    dfk[PRED_COL] = coerce_pred_array(dfk[PRED_COL].values)
                    dfk[TARGET_PROP] = pd.to_numeric(dfk[TARGET_PROP], errors='coerce')
                    y_true = dfk[TARGET_PROP].values.astype(float)
                    y_pred = dfk[PRED_COL].values.astype(float)
                except Exception:
                    continue
            mae = np.mean(np.abs((y_pred - y_true) * 1000.0))
            xs.append(f)
            ys.append(mae)
            yerr.append(0.0)
        if xs:
            axr.errorbar(xs, ys, yerr=yerr, marker='o', color=color, label=f"{model} ({struct})")
    axr.set_xscale('log')
    axr.set_xticks([0.125, 0.25, 0.5, 1.0])
    axr.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axr.legend(frameon=False, fontsize=9)

    out_png = os.path.join(FIG_DIR, 'figure2_composition.png')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"Saved: {out_png}")


def _load_embeddings_df(model: str, struct: str):
    # Hard-code the specific directory for CGCNN relaxed
    if model == "CGCNN" and struct == "relaxed":
        model_root = "./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612"
        if not os.path.exists(model_root):
            raise FileNotFoundError(f"Hard-coded path not found: {model_root}")
    else:
        # First try the exact path
        model_root = find_model_path(model, struct)
        
        # If not found, search for directories with suffixes (like 837612)
        if model_root is None:
            base_pattern = f"dft_e_hull_htvs_data_{struct}_{model}"
            model_base = os.path.join(BEST_BASE, model)
            if os.path.exists(model_base):
                for item in os.listdir(model_base):
                    if item.startswith(base_pattern):
                        model_root = os.path.join(model_base, item)
                        break
        
        if model_root is None:
            raise FileNotFoundError(f"No directory found for {model}-{struct}")
    
    # Look for best_* directories, including nested ones (like 837612/best_*)
    best_dirs = {}
    for root, dirs, files in os.walk(model_root):
        for d in dirs:
            if d.startswith('best_'):
                try:
                    k = int(d.split('_')[1])
                    best_dirs[k] = os.path.join(root, d)
                except (IndexError, ValueError):
                    continue
    
    if not best_dirs:
        raise FileNotFoundError(f"No best_* dirs found under {model_root}")
    
    # Prefer best_0
    search_dirs = [best_dirs.get(0)] + [d for k, d in best_dirs.items() if k != 0]
    for d in search_dirs:
        if not d:
            continue
        # Prefer holdout_set_B_sites embeddings, then series, then any *embeddings*.json
        candidates = []
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith('.json') and 'embeddings' in f:
                    fp = os.path.join(root, f)
                    # Prioritize by keyword
                    prio = 2
                    if 'holdout_set_B_sites_embeddings' in f:
                        prio = 0
                    elif 'holdout_set_series_embeddings' in f:
                        prio = 1
                    candidates.append((prio, fp))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            path = candidates[0][1]
            df = pd.read_json(path)
            return df
    raise FileNotFoundError(f"No embeddings json under {model_root}")


def _np_embedding_from_column(col):
    arrs = []
    for emb in col:
        e = emb
        # unwrap until numeric array
        while isinstance(e, (list, tuple)) and len(e) > 0 and isinstance(e[0], (list, tuple, np.ndarray, float, int)):
            if isinstance(e[0], (float, int)):
                break
            e = e[0]
        arrs.append(np.asarray(e, dtype=float))
    return np.stack(arrs, axis=0)


def figure4_embeddings_alignn_only():
    # Generate embeddings figure for CGCNN and ALIGNN only (both unrelaxed and relaxed)
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    subfig_l, subfig_r = fig.subfigures(nrows=1, ncols=2, width_ratios=[1.5, 1])

    model_types = ["CGCNN", "ALIGNN"]
    struct_types = ["unrelaxed", "relaxed"]

    pca_spread = {m: {s: [] for s in struct_types} for m in model_types}

    axes_l = subfig_l.subplots(nrows=2, ncols=2, gridspec_kw={'left': 0.3, 'bottom': 0.6, 'hspace': 0.04})

    for i, m in enumerate(model_types):
        for j, s in enumerate(struct_types):
            ax = axes_l[i][j]
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            try:
                df = _load_embeddings_df(m, s)
                if 'embedding_0' not in df.columns:
                    ax.set_title(f"{m} ({s}) - no embeddings", fontsize=10)
                    continue
                np_emb = _np_embedding_from_column(df['embedding_0'])
                pca = PCA(n_components=2)
                pca.fit(np_emb)
                proj = pca.transform(np_emb)
                # Background scatter
                ax.scatter(-proj[:,0], -proj[:,1], s=30, color='black', alpha=0.1, linewidth=0)
                # Compute per-formula spread normalized by mean norm
                mean_pos_all = proj.mean(axis=0)
                norm_dist = np.mean(np.linalg.norm(proj - mean_pos_all, axis=1))
                spreads = []
                if 'formula' in df.columns:
                    for form, grp in df.groupby('formula'):
                        pj = pca.transform(_np_embedding_from_column(grp['embedding_0']))
                        mean_pos = pj.mean(axis=0)
                        dists = np.linalg.norm(pj - mean_pos, axis=1)
                        spreads.append(np.mean(dists) / (norm_dist if norm_dist > 0 else 1.0))
                pca_spread[m][s] = spreads
            except Exception as e:
                ax.set_title(f"{m} ({s}) - missing", fontsize=10)
                pca_spread[m][s] = []

    # Labels/colors
    axes_l[0][0].text(0.96, 0.05, 'CGCNN\n(unrelaxed)', ha='right', fontsize=16, transform=axes_l[0][0].transAxes, color='cornflowerblue')
    axes_l[0][1].text(0.96, 0.05, 'CGCNN\n(relaxed)', ha='right', fontsize=16, transform=axes_l[0][1].transAxes, color='darkblue')
    axes_l[1][0].text(0.96, 0.05, 'ALIGNN\n(unrelaxed)', ha='right', fontsize=16, transform=axes_l[1][0].transAxes, color='teal')
    axes_l[1][1].text(0.96, 0.05, 'ALIGNN\n(relaxed)', ha='right', fontsize=16, transform=axes_l[1][1].transAxes, color='darkcyan')

    subfig_l.supxlabel('PCA 1 of GCNN embedding after graph convolution', x=0.515, fontsize=16)
    subfig_l.supylabel('PCA 2 of GCNN embedding after graph convolution', y=0.515, fontsize=16)

    # Right violin (use sinaplot if available else seaborn violinplot as fallback)
    ax_r = subfig_r.subplots(nrows=1, ncols=1)
    ax_r.get_yaxis().set_visible(False)
    ax_r.set_xlim(-0.05, 1.25)
    ax_r.set_xticks(np.arange(0, 1.25, 0.5))

    spread_rows = []
    for i, m in enumerate(model_types):
        for j, s in enumerate(struct_types):
            for v in pca_spread[m][s]:
                spread_rows.append({"category": 2*i + j, "spread": v})
    df_sp = pd.DataFrame(spread_rows)
    if not df_sp.empty:
        if SINAPLOT_AVAILABLE:
            pal = sns.color_palette(['cornflowerblue', 'darkblue', 'teal', 'darkcyan'])
            sns.set_palette(pal)
            sinaplot(y='category', x='spread', orient='h', data=df_sp, width=0.5, cut=0, linewidth=2,
                     violin_facealpha=0.1, saturation=0.6, point_size=10, point_facealpha=0.6, random_state=10)
        else:
            sns.violinplot(data=df_sp, x='spread', y='category', orient='h', inner=None, linewidth=1, cut=0)

    # Label rows - categories are: 0=CGCNN(unrelaxed), 1=CGCNN(relaxed), 2=ALIGNN(unrelaxed), 3=ALIGNN(relaxed)
    ax_r.text(0.12, 0.845, 'CGCNN\n(unrelaxed)', color='cornflowerblue', fontsize=14, ha='left', transform=ax_r.transAxes)
    ax_r.text(0.30, 0.695, 'CGCNN\n(relaxed)', color='darkblue', fontsize=14, ha='left', transform=ax_r.transAxes)
    ax_r.text(0.70, 0.545, 'ALIGNN\n(unrelaxed)', color='teal', fontsize=14, ha='left', transform=ax_r.transAxes)
    ax_r.text(0.78, 0.395, 'ALIGNN\n(relaxed)', color='darkcyan', fontsize=14, ha='left', transform=ax_r.transAxes)

    # Set y-axis limits to match the 4 categories (0, 1, 2, 3)
    ax_r.set_ylim(3.5, -0.5)
    ax_r.set_xlabel('Embedding spread across various orderings in PCA')

    out_png = os.path.join(FIG_DIR, 'fig4_embeddings_alignn.png')
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    print(f"Saved: {out_png}")


def main():
    print("Generating hexbin parity and relative plots for all models...")
    summary = []
    for model in MODELS:
        for struct in STRUCTS:
            # Hex parity (test)
            try:
                test_df = load_predictions(model, struct, split="test_set")
                stats = parity_hexbin(test_df, model, struct, split="test")
                print(f"Saved hex parity: {stats['file_png']}")
                summary.append({"model": model, "struct": struct, "split": "test", **stats})
            except Exception as e:
                print(f"Skip hex parity for {model}-{struct}: {e}")
            # Hex parity (holdout absolute)
            try:
                hold_df = load_predictions(model, struct, split="holdout_set_B_sites")
                stats_h = parity_hexbin(hold_df, model, struct, split="holdout")
                print(f"Saved hex parity: {stats_h['file_png']}")
                summary.append({"model": model, "struct": struct, "split": "holdout", **stats_h})
                # Hex relative per formula
                rel = relative_hexbin(hold_df, model, struct)
                print(f"Saved hex relative: {rel['file_png']}")
            except Exception as e:
                print(f"Skip holdout hex for {model}-{struct}: {e}")

    # Composite panels
    figure3_relative_orderings()
    figure3_hexbin_composite()
    figure2_composition_hex_and_learning_curve()
    if 'figure4_embeddings_alignn_only' in globals():
        try:
            figure4_embeddings_alignn_only()
        except Exception as e:
            print(f"Skip Fig4 embeddings: {e}")
    if 'figure5_generalizability_alignn' in globals():
        try:
            figure5_generalizability_alignn()
        except Exception as e:
            print(f"Skip Fig5 generalizability: {e}")
    if 'figure6_interatomic_potentials_scaffold' in globals():
        try:
            figure6_interatomic_potentials_scaffold()
        except Exception as e:
            print(f"Skip Fig6 IP scaffold: {e}")
    embedding_panels_and_spread()

    with open(os.path.join(FIG_DIR, "parity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Figure 4: Embeddings with ALIGNN using sinaplot
    # figure4_embeddings_with_alignn() # This line is now handled by the main function call


if __name__ == "__main__":
    main()
