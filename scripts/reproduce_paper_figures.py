#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


def coerce_pred_array(arr_like):
    arr = np.asarray(arr_like, dtype=object)
    if len(arr) > 0 and isinstance(arr[0], (list, tuple, np.ndarray)):
        arr = np.array([x[0] if isinstance(x, (list, tuple, np.ndarray)) else x for x in arr], dtype=object)
    ser = pd.to_numeric(pd.Series(arr, dtype=object), errors="coerce")
    return ser.values


def find_model_path(model: str, struct: str):
    """Find the correct path for a model, handling both directory structures"""
    base_path = os.path.join(BEST_BASE, model, f"dft_e_hull_htvs_data_{struct}_{model}")
    
    if not os.path.exists(base_path):
        return None
    
    # Check if it has experiment IDs (like e3nn)
    if os.path.isdir(base_path):
        exp_dirs = [d for d in os.listdir(base_path) if d.isdigit() and not d.startswith('.')]
        if exp_dirs:
            # Use the first experiment ID
            exp_id = exp_dirs[0]
            return os.path.join(base_path, exp_id)
        else:
            # No experiment ID (like CGCNN, ALIGNN)
            return base_path
    
    return base_path


def load_predictions(model: str, struct: str, split: str):
    """Load predictions from the correct path structure"""
    model_path = find_model_path(model, struct)
    if model_path is None:
        raise FileNotFoundError(f"No directory found for {model}-{struct}")
    
    folder = os.path.join(model_path, "best_0")
    path = os.path.join(folder, f"{split}_predictions.json")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing predictions file: {path}")
    
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
    """Load ensemble predictions from best_0, best_1, best_2"""
    model_path = find_model_path(model, struct)
    if model_path is None:
        raise FileNotFoundError(f"No directory found for {model}-{struct}")
    
    dfs = []
    for k in [0, 1, 2]:
        folder = os.path.join(model_path, f"best_{k}")
        path = os.path.join(folder, f"{split}_predictions.json")
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
    
    # Ensure same ordering; use index of first
    ref = dfs[0].copy()
    preds = [d[PRED_COL].values.astype(float) for d in dfs]
    pred_mean = np.mean(np.vstack(preds), axis=0)
    out = ref.copy()
    out[PRED_COL] = pred_mean
    out = out.dropna(subset=[TARGET_PROP, PRED_COL])
    return out


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


def main():
    print("Generating parity and relative holdout plots...")
    summary = []
    for model in MODELS:
        for struct in STRUCTS:
            try:
                test_df = load_predictions(model, struct, split="test_set")
                stats = parity_plot(test_df, model, struct, split="test")
                print(f"Saved parity plot: {stats['file']}")
                summary.append({"model": model, "struct": struct, "split": "test", **stats})
            except Exception as e:
                print(f"Skip parity for {model}-{struct}: {e}")
            try:
                hold_df = load_predictions(model, struct, split="holdout_set_B_sites")
                stats = parity_plot(hold_df, model, struct, split="holdout")
                print(f"Saved parity plot: {stats['file']}")
                summary.append({"model": model, "struct": struct, "split": "holdout", **stats})
                rel = relative_holdout_plot(hold_df, model, struct)
                print(f"Saved relative plot: {rel['file']}")
            except Exception as e:
                print(f"Skip holdout for {model}-{struct}: {e}")

    # Figure 3 replication (ensemble mean of best_0..2)
    figure3_relative_orderings()

    # Embedding panels + spread
    embedding_panels_and_spread()

    with open(os.path.join(FIG_DIR, "parity_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
