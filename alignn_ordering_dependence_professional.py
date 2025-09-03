#!/usr/bin/env python3
"""
Generate ALIGNN ordering dependence plot in the same style as CGCNN and e3nn
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Set style to match the notebook
plt.style.use('default')

def flatten(data):
    """Flatten nested lists and convert to numpy array, handling None values"""
    if isinstance(data, pd.Series):
        data = data.values
    
    flattened = []
    for item in data:
        if isinstance(item, list):
            if len(item) == 1:
                if item[0] is None:
                    # Handle failed predictions by using 0.0 as default
                    flattened.append(0.0)
                elif isinstance(item[0], (int, float)):
                    flattened.append(float(item[0]))
                else:
                    flattened.append(0.0)  # Default for other types
            else:
                # Handle longer lists
                for subitem in item:
                    if subitem is None:
                        flattened.append(0.0)
                    else:
                        flattened.append(float(subitem))
        elif item is None:
            # Handle direct None values
            flattened.append(0.0)
        else:
            flattened.append(float(item))
    
    return np.array(flattened)

def get_relative_vals(df, target_prop):
    """Get relative values for ordering dependence analysis"""
    # Group by formula and get relative values
    relative_vals = []
    for formula in df['formula'].unique():
        formula_data = df[df['formula'] == formula]
        if len(formula_data) > 1:
            # Get the minimum value for this formula
            min_val = formula_data[target_prop].min()
            # Calculate relative values
            rel_vals = formula_data[target_prop] - min_val
            relative_vals.extend(rel_vals.values)
        else:
            # Single structure, relative value is 0
            relative_vals.append(0.0)
    
    return np.array(relative_vals)

def main():
    print("🔍 Loading ALIGNN model predictions for professional ordering dependence analysis...")
    
    # Load ALIGNN predictions
    try:
        # Test set predictions
        ehull_ALIGNN_unrelaxed_test_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_0/test_set_predictions.json")
        ehull_ALIGNN_unrelaxed_test_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_1/test_set_predictions.json")
        ehull_ALIGNN_unrelaxed_test_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_2/test_set_predictions.json")
        
        ehull_ALIGNN_relaxed_test_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_0/test_set_predictions.json")
        ehull_ALIGNN_relaxed_test_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_1/test_set_predictions.json")
        ehull_ALIGNN_relaxed_test_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_2/test_set_predictions.json")
        
        print("✅ Loaded ALIGNN test set predictions")
        
    except Exception as e:
        print(f"❌ Error loading ALIGNN predictions: {e}")
        return
    
    # Process predictions
    print("Processing predictions for ordering dependence...")
    
    try:
        # Test set - ALIGNN
        print("Processing unrelaxed test predictions...")
        ehull_ALIGNN_unrelaxed_test_true = ehull_ALIGNN_unrelaxed_test_0['dft_e_hull'].values
        
        print("Flattening unrelaxed predictions...")
        pred_0_unrelaxed = flatten(ehull_ALIGNN_unrelaxed_test_0.predicted_dft_e_hull)
        pred_1_unrelaxed = flatten(ehull_ALIGNN_unrelaxed_test_1.predicted_dft_e_hull)
        pred_2_unrelaxed = flatten(ehull_ALIGNN_unrelaxed_test_2.predicted_dft_e_hull)
        
        ehull_ALIGNN_unrelaxed_test_pred = (pred_0_unrelaxed + pred_1_unrelaxed + pred_2_unrelaxed)/3.0
        
        print("Processing relaxed test predictions...")
        ehull_ALIGNN_relaxed_test_true = ehull_ALIGNN_relaxed_test_0['dft_e_hull'].values
        
        print("Flattening relaxed predictions...")
        pred_0_relaxed = flatten(ehull_ALIGNN_relaxed_test_0.predicted_dft_e_hull)
        pred_1_relaxed = flatten(ehull_ALIGNN_relaxed_test_1.predicted_dft_e_hull)
        pred_2_relaxed = flatten(ehull_ALIGNN_relaxed_test_2.predicted_dft_e_hull)
        
        ehull_ALIGNN_relaxed_test_pred = (pred_0_relaxed + pred_1_relaxed + pred_2_relaxed)/3.0
        
        print("✅ Processed all predictions")
        
    except Exception as e:
        print(f"❌ Error processing predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create DataFrames for analysis
    print("Creating DataFrames for ordering dependence analysis...")
    
    # Unrelaxed DataFrame
    df_unrelaxed = pd.DataFrame({
        'formula': ehull_ALIGNN_unrelaxed_test_0['formula'].values,
        'dft_e_hull': ehull_ALIGNN_unrelaxed_test_true,
        'predicted_dft_e_hull': ehull_ALIGNN_unrelaxed_test_pred
    })
    
    # Relaxed DataFrame
    df_relaxed = pd.DataFrame({
        'formula': ehull_ALIGNN_relaxed_test_0['formula'].values,
        'dft_e_hull': ehull_ALIGNN_relaxed_test_true,
        'predicted_dft_e_hull': ehull_ALIGNN_relaxed_test_pred
    })
    
    # Calculate relative values
    print("Calculating relative values for ordering dependence...")
    
    # True relative values
    diffs_dft_unrelaxed = get_relative_vals(df_unrelaxed, 'dft_e_hull')
    diffs_dft_relaxed = get_relative_vals(df_relaxed, 'dft_e_hull')
    
    # Predicted relative values
    diffs_ALIGNN_unrelaxed = get_relative_vals(df_unrelaxed, 'predicted_dft_e_hull')
    diffs_ALIGNN_relaxed = get_relative_vals(df_relaxed, 'predicted_dft_e_hull')
    
    print(f"✅ Calculated relative values - Unrelaxed: {len(diffs_dft_unrelaxed)}, Relaxed: {len(diffs_dft_relaxed)}")
    
    # Create output directory
    import os
    os.makedirs('./figures', exist_ok=True)
    
    # Create professional ordering dependence plot matching the notebook style
    print("Creating professional ALIGNN ordering dependence plot...")
    
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    (subfig_l, subfig_r) = fig.subfigures(nrows=1, ncols=2, width_ratios=[2, 1])
    
    axes_l = subfig_l.subplots(nrows=2, ncols=2, sharex=True, sharey=True, gridspec_kw={'left': 0.3})
    hex_cmap = 'inferno_r'
    hex_gridsize = 30
    hex_mincnt = 1
    hex_edgecolors = 'black'
    hex_linewidths = 0.5
    hex_xylim = [-40, 190]
    cbar_vmax = 22
    
    # Add diagonal lines
    axes_l[0][0].axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), color='black', linestyle='--', linewidth=2)
    axes_l[1][0].axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), color='black', linestyle='--', linewidth=2)
    axes_l[0][1].axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), color='black', linestyle='--', linewidth=2)
    axes_l[1][1].axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), color='black', linestyle='--', linewidth=2)
    
    # Create hexbin plots
    axes_l[0][0].hexbin(
        diffs_dft_unrelaxed*1000, diffs_ALIGNN_unrelaxed*1000,
        cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
        extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
    )
    
    axes_l[1][0].hexbin(
        diffs_dft_unrelaxed*1000, diffs_ALIGNN_unrelaxed*1000,
        cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
        extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
    )
    
    axes_l[0][1].hexbin(
        diffs_dft_relaxed*1000, diffs_ALIGNN_relaxed*1000,
        cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
        extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
    )
    
    hex_example = axes_l[1][1].hexbin(
        diffs_dft_relaxed*1000, diffs_ALIGNN_relaxed*1000,
        cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
        extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
    )
    
    # Add labels
    axes_l[0][0].text(0.96, 0.05, 'ALIGNN\n(unrelaxed)', horizontalalignment='right', fontsize=16, transform=axes_l[0][0].transAxes, color='red')
    axes_l[1][0].text(0.96, 0.05, 'ALIGNN\n(unrelaxed)', horizontalalignment='right', fontsize=16, transform=axes_l[1][0].transAxes, color='red')
    axes_l[0][1].text(0.96, 0.05, 'ALIGNN\n(relaxed)', horizontalalignment='right', fontsize=16, transform=axes_l[0][1].transAxes, color='darkred')
    axes_l[1][1].text(0.96, 0.05, 'ALIGNN\n(relaxed)', horizontalalignment='right', fontsize=16, transform=axes_l[1][1].transAxes, color='darkred')
    
    # Set ticks
    axes_l[0][0].set_yticks(np.arange(0, hex_xylim[1], 50))
    axes_l[0][0].set_xticks(np.arange(0, hex_xylim[1], 50))
    
    # Add colorbar
    subfig_l.colorbar(hex_example, ax=axes_l, label='Count', ticks=np.arange(0, cbar_vmax+1, 5), aspect=40)
    
    # Add axis labels
    subfig_l.supxlabel('DFT $\mathit{\Delta E}_{\mathrm{hull}}$ (meV/atom vs. ground-state ordering)', x=0.49, fontsize=16)
    subfig_l.supylabel('ALIGNN $\mathit{\Delta E}_{\mathrm{hull}}$ (meV/atom vs. ground-state ordering)', y=0.55, fontsize=16)
    
    # Right panel for MAE and R² scores
    ax_r = subfig_r.subplots(nrows=1, ncols=1)
    ax_r_yaxis = ax_r.get_yaxis()
    ax_r_yaxis.set_visible(False)
    ax_r.set_xlabel('Holdout set MAE (meV/atom)', labelpad=9)
    ax_r.set_xlim(7.5, 24)
    ax_r.set_ylim(-4, 4)
    ax_r.set_xticks(np.arange(10, 21, 5))
    
    # Calculate MAE values
    mae_unrelaxed = mean_absolute_error(diffs_dft_unrelaxed, diffs_ALIGNN_unrelaxed)*1000
    mae_relaxed = mean_absolute_error(diffs_dft_relaxed, diffs_ALIGNN_relaxed)*1000
    
    # Calculate R² scores
    r2_unrelaxed = r2_score(diffs_dft_unrelaxed, diffs_ALIGNN_unrelaxed)
    r2_relaxed = r2_score(diffs_dft_relaxed, diffs_ALIGNN_relaxed)
    
    # Create bars
    ax_r.barh(y=-0.5, width=mae_unrelaxed, height=0.4, color='red')
    ax_r.barh(y=-1.5, width=mae_relaxed, height=0.4, color='darkred')
    
    # Add R² scores
    ax_r_twiny = ax_r.twiny()
    ax_r_twiny.set_xlabel('Holdout set $\mathit{R}^2$ score', labelpad=9)
    ax_r_twiny.set_xlim(-0.3, 0.9)
    ax_r_twiny.set_xticks(np.arange(-0.2, 0.9, 0.2))
    
    ax_r_twiny.plot(r2_unrelaxed, 1.5, 'o', markersize=20, markeredgecolor='black', markeredgewidth=0, color='red')
    ax_r_twiny.plot(r2_relaxed, 0.5, 'o', markersize=20, markeredgecolor='black', markeredgewidth=0, color='darkred')
    
    # Add horizontal line
    ax_r.hlines(0, 0, 100, color='black', linestyle='-', linewidth=2)
    
    # Add labels
    ax_r.text(0.165, 0.905, 'ALIGNN\n(unrelaxed)', color='red', fontsize=16, ha='left', transform=ax_r.transAxes)
    ax_r.text(0.390, 0.780, 'ALIGNN\n(relaxed)', color='darkred', fontsize=16, ha='left', transform=ax_r.transAxes)
    
    ax_r.text(0.700, 0.410, 'ALIGNN\n(unrelaxed)', color='red', fontsize=16, ha='left', transform=ax_r.transAxes)
    ax_r.text(0.290, 0.035, 'ALIGNN\n(relaxed)', color='darkred', fontsize=16, ha='left', transform=ax_r.transAxes)
    
    # Save the plot
    plt.savefig('./figures/ALIGNN_ordering_dependence_professional.pdf', bbox_inches='tight', dpi=300)
    print("✅ Saved professional ALIGNN ordering dependence plot")
    
    # Print statistics
    print(f"\n📊 ALIGNN Ordering Dependence Statistics:")
    print(f"   Unrelaxed MAE: {mae_unrelaxed:.2f} meV/atom")
    print(f"   Relaxed MAE: {mae_relaxed:.2f} meV/atom")
    print(f"   Unrelaxed R²: {r2_unrelaxed:.3f}")
    print(f"   Relaxed R²: {r2_relaxed:.3f}")
    
    print("\n🎉 Professional ALIGNN ordering dependence analysis completed!")
    print("📁 Check the 'figures' directory for the generated plot:")
    print("   - ALIGNN_ordering_dependence_professional.pdf")

if __name__ == "__main__":
    main()
