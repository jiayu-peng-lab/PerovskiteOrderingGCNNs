#!/usr/bin/env python3
"""
Generate plots for ALIGNN models only
"""
import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# Set style
plt.style.use('default')
sns.set_palette("husl")

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

def main():
    print("🔍 Loading ALIGNN model predictions...")
    
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
    print("Processing predictions...")
    
    try:
        # Test set - ALIGNN
        print("Processing unrelaxed test predictions...")
        ehull_ALIGNN_unrelaxed_test_true = ehull_ALIGNN_unrelaxed_test_0['dft_e_hull'].values
        print(f"Unrelaxed true values shape: {ehull_ALIGNN_unrelaxed_test_true.shape}")
        
        print("Flattening unrelaxed predictions...")
        pred_0_unrelaxed = flatten(ehull_ALIGNN_unrelaxed_test_0.predicted_dft_e_hull)
        pred_1_unrelaxed = flatten(ehull_ALIGNN_unrelaxed_test_1.predicted_dft_e_hull)
        pred_2_unrelaxed = flatten(ehull_ALIGNN_unrelaxed_test_2.predicted_dft_e_hull)
        
        print(f"Prediction shapes: {pred_0_unrelaxed.shape}, {pred_1_unrelaxed.shape}, {pred_2_unrelaxed.shape}")
        
        ehull_ALIGNN_unrelaxed_test_pred = (pred_0_unrelaxed + pred_1_unrelaxed + pred_2_unrelaxed)/3.0
        print(f"Final unrelaxed predictions shape: {ehull_ALIGNN_unrelaxed_test_pred.shape}")
        
        print("Processing relaxed test predictions...")
        ehull_ALIGNN_relaxed_test_true = ehull_ALIGNN_relaxed_test_0['dft_e_hull'].values
        print(f"Relaxed true values shape: {ehull_ALIGNN_relaxed_test_true.shape}")
        
        print("Flattening relaxed predictions...")
        pred_0_relaxed = flatten(ehull_ALIGNN_relaxed_test_0.predicted_dft_e_hull)
        pred_1_relaxed = flatten(ehull_ALIGNN_relaxed_test_1.predicted_dft_e_hull)
        pred_2_relaxed = flatten(ehull_ALIGNN_relaxed_test_2.predicted_dft_e_hull)
        
        print(f"Prediction shapes: {pred_0_relaxed.shape}, {pred_1_relaxed.shape}, {pred_2_relaxed.shape}")
        
        ehull_ALIGNN_relaxed_test_pred = (pred_0_relaxed + pred_1_relaxed + pred_2_relaxed)/3.0
        print(f"Final relaxed predictions shape: {ehull_ALIGNN_relaxed_test_pred.shape}")
        
    except Exception as e:
        print(f"❌ Error processing predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("✅ Processed all predictions")
    
    # Calculate MAE
    mae_unrelaxed = mean_absolute_error(ehull_ALIGNN_unrelaxed_test_true, ehull_ALIGNN_unrelaxed_test_pred)
    mae_relaxed = mean_absolute_error(ehull_ALIGNN_relaxed_test_true, ehull_ALIGNN_relaxed_test_pred)
    
    print(f"📊 ALIGNN Unrelaxed MAE: {mae_unrelaxed*1000:.1f} meV/atom")
    print(f"📊 ALIGNN Relaxed MAE: {mae_relaxed*1000:.1f} meV/atom")
    
    # Create output directory
    import os
    os.makedirs('./figures', exist_ok=True)
    
    # Create ALIGNN plots
    print("Creating ALIGNN plots...")
    
    # 1. Test set predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Unrelaxed
    ax1.hexbin(ehull_ALIGNN_unrelaxed_test_true*1000, ehull_ALIGNN_unrelaxed_test_pred*1000,
               cmap='inferno_r', gridsize=30, mincnt=1, edgecolors='black', linewidths=0.5)
    ax1.axline((0, 0), (1, 1), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('DFT $E_{hull}$ (meV/atom)', fontsize=14)
    ax1.set_ylabel('ALIGNN $E_{hull}$ (meV/atom)', fontsize=14)
    ax1.set_title(f'ALIGNN Unrelaxed (MAE: {mae_unrelaxed*1000:.1f} meV/atom)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Relaxed
    hex_plot = ax2.hexbin(ehull_ALIGNN_relaxed_test_true*1000, ehull_ALIGNN_relaxed_test_pred*1000,
                          cmap='inferno_r', gridsize=30, mincnt=1, edgecolors='black', linewidths=0.5)
    ax2.axline((0, 0), (1, 1), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('DFT $E_{hull}$ (meV/atom)', fontsize=14)
    ax2.set_ylabel('ALIGNN $E_{hull}$ (meV/atom)', fontsize=14)
    ax2.set_title(f'ALIGNN Relaxed (MAE: {mae_relaxed*1000:.1f} meV/atom)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(hex_plot, ax=[ax1, ax2], label='Count')
    
    plt.tight_layout()
    plt.savefig('./figures/ALIGNN_test_predictions.pdf', bbox_inches='tight', dpi=300)
    print("✅ Saved ALIGNN test predictions plot")
    
    # 2. Error distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Calculate errors
    errors_unrelaxed = (ehull_ALIGNN_unrelaxed_test_pred - ehull_ALIGNN_unrelaxed_test_true) * 1000
    errors_relaxed = (ehull_ALIGNN_relaxed_test_pred - ehull_ALIGNN_relaxed_test_true) * 1000
    
    # Unrelaxed error distribution
    ax1.hist(errors_unrelaxed, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Prediction Error (meV/atom)', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.set_title(f'ALIGNN Unrelaxed Error Distribution\n(MAE: {mae_unrelaxed*1000:.1f} meV/atom)', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Relaxed error distribution
    ax2.hist(errors_relaxed, bins=50, alpha=0.7, color='darkred', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Prediction Error (meV/atom)', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.set_title(f'ALIGNN Relaxed Error Distribution\n(MAE: {mae_relaxed*1000:.1f} meV/atom)', fontsize=16)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/ALIGNN_error_distributions.pdf', bbox_inches='tight', dpi=300)
    print("✅ Saved ALIGNN error distributions plot")
    
    # 3. Performance comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['ALIGNN Unrelaxed', 'ALIGNN Relaxed']
    mae_values = [mae_unrelaxed*1000, mae_relaxed*1000]
    colors = ['red', 'darkred']
    
    bars = ax.bar(models, mae_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Mean Absolute Error (meV/atom)', fontsize=14)
    ax.set_title('ALIGNN Model Performance Comparison', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('./figures/ALIGNN_performance_comparison.pdf', bbox_inches='tight', dpi=300)
    print("✅ Saved ALIGNN performance comparison plot")
    
    print("\n🎉 All ALIGNN plots generated successfully!")
    print("📁 Check the 'figures' directory for the generated plots:")
    print("   - ALIGNN_test_predictions.pdf")
    print("   - ALIGNN_error_distributions.pdf") 
    print("   - ALIGNN_performance_comparison.pdf")

if __name__ == "__main__":
    main()
