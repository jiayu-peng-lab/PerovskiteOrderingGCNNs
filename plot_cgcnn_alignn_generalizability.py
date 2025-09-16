#!/usr/bin/env python3
"""
Plot generalizability comparison between CGCNN and ALIGNN models
for perovskite oxide compositions with violin plots showing energy distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Ensure figures directory exists
os.makedirs('./figures', exist_ok=True)

def flatten(lst):
    """Flatten a nested list."""
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list



# Load data for CGCNN, ALIGNN, and e3nn models
print("Loading CGCNN data...")
ehull_CGCNN_unrelaxed_held_series_0 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_0/holdout_set_series_predictions.json")
ehull_CGCNN_unrelaxed_held_series_1 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_1/holdout_set_series_predictions.json")
ehull_CGCNN_unrelaxed_held_series_2 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_2/holdout_set_series_predictions.json")

ehull_CGCNN_relaxed_held_series_0 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_0/holdout_set_series_predictions.json")
ehull_CGCNN_relaxed_held_series_1 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_1/holdout_set_series_predictions.json")
ehull_CGCNN_relaxed_held_series_2 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_2/holdout_set_series_predictions.json")

print("Loading ALIGNN data...")
ehull_ALIGNN_unrelaxed_held_series_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_0/holdout_set_series_predictions.json")
ehull_ALIGNN_unrelaxed_held_series_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_1/holdout_set_series_predictions.json")
ehull_ALIGNN_unrelaxed_held_series_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_2/holdout_set_series_predictions.json")

ehull_ALIGNN_relaxed_held_series_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_0/holdout_set_series_predictions.json")
ehull_ALIGNN_relaxed_held_series_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_1/holdout_set_series_predictions.json")
ehull_ALIGNN_relaxed_held_series_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_2/holdout_set_series_predictions.json")

print("Loading e3nn data...")
ehull_e3nn_unrelaxed_held_series_0 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_0/holdout_set_series_predictions.json")
ehull_e3nn_unrelaxed_held_series_1 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_1/holdout_set_series_predictions.json")
ehull_e3nn_unrelaxed_held_series_2 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_1/holdout_set_series_predictions.json")

ehull_e3nn_relaxed_held_series_0 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_0/holdout_set_series_predictions.json")
ehull_e3nn_relaxed_held_series_1 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_1/holdout_set_series_predictions.json")
ehull_e3nn_relaxed_held_series_2 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_2/holdout_set_series_predictions.json")

# Define perovskite series names and symbols
series_names = [
    'La$_\mathit{x}$Bi$_{1-\mathit{x}}$Cu$_{0.5}$Mo$_{0.5}$O$_3$',
    'K$_\mathit{x}$Ba$_{1-\mathit{x}}$Al$_{0.5}$Ti$_{0.5}$O$_3$',
    'Sr$_\mathit{x}$La$_{1-\mathit{x}}$Fe$_{0.5}$Co$_{0.5}$O$_3$',
    'Mg$_\mathit{x}$Pr$_{1-\mathit{x}}$V$_{0.5}$Ni$_{0.5}$O$_3$',
    'Y$_\mathit{x}$La$_{1-\mathit{x}}$Mg$_{0.5}$In$_{0.5}$O$_3$',
    'La$_\mathit{x}$Pr$_{1-\mathit{x}}$Ni$_{0.5}$Y$_{0.5}$O$_3$',
]

series_symbols = [
    [["La", "Bi"], ["Cu", "Mo"]],
    [["K",  "Ba"], ["Al", "Ti"]],
    [["Sr", "La"], ["Fe", "Co"]],
    [["Mg", "Pr"], ["V",  "Ni"]],
    [["Y",  "La"], ["Mg", "In"]],
    [["La", "Pr"], ["Ni",  "Y"]],
]

# Create the main figure
print("Creating plot...")
fig = plt.figure(figsize=(13, 10), constrained_layout=True)
subfig_d = fig.subfigures(nrows=1, ncols=1)

# Create the main violin plot section
subfig_dd = subfig_d.subfigures(nrows=2, ncols=1, height_ratios=[0.08, 1])[1]
subfig_ddd = subfig_dd.subfigures(nrows=1, ncols=2, width_ratios=[1, 0.19])[0]
subfig_ddd.supxlabel('$\mathit{x}$ in perovskite oxide composition', x=0.535, fontsize=16)
subfig_ddd.supylabel('$\mathit{E}_{\mathrm{hull}}$ (meV/atom)', y=0.560, fontsize=16)
subfigs = subfig_ddd.subfigures(nrows=6, ncols=1, height_ratios=[1, 1, 1, 1, 1, 1.28])
axes = []

# Create violin plots for each perovskite series
for i in range(len(subfigs)):
    axes.append(subfigs[i].subplots(nrows=1, ncols=7, sharex=True, sharey=True, gridspec_kw={'wspace': -0.1}))
    
    # Add column labels for the first row only
    if i == 0:
        axes[i][0].set_title('CGCNN\n(unrelaxed)', fontsize=12, pad=10)
        axes[i][1].set_title('e3nn\n(unrelaxed)', fontsize=12, pad=10)
        axes[i][2].set_title('ALIGNN\n(unrelaxed)', fontsize=12, pad=10)
        axes[i][3].set_title('CGCNN\n(relaxed)', fontsize=12, pad=10)
        axes[i][4].set_title('e3nn\n(relaxed)', fontsize=12, pad=10)
        axes[i][5].set_title('ALIGNN\n(relaxed)', fontsize=12, pad=10)
        axes[i][6].set_title('DFT', fontsize=12, pad=10)

    for j in range(len(axes[i])):
        if j == 1:
            # e3nn unrelaxed
            temp_dfs = [ehull_e3nn_unrelaxed_held_series_0, ehull_e3nn_unrelaxed_held_series_1, ehull_e3nn_unrelaxed_held_series_2]            
        elif j == 2:
            # ALIGNN unrelaxed
            temp_dfs = [ehull_ALIGNN_unrelaxed_held_series_0, ehull_ALIGNN_unrelaxed_held_series_1, ehull_ALIGNN_unrelaxed_held_series_2]
        elif j == 3:
            # CGCNN relaxed
            temp_dfs = [ehull_CGCNN_relaxed_held_series_0, ehull_CGCNN_relaxed_held_series_1, ehull_CGCNN_relaxed_held_series_2]
        elif j == 4:
            # e3nn relaxed
            temp_dfs = [ehull_e3nn_relaxed_held_series_0, ehull_e3nn_relaxed_held_series_1, ehull_e3nn_relaxed_held_series_2]
        elif j == 5:
            # ALIGNN relaxed
            temp_dfs = [ehull_ALIGNN_relaxed_held_series_0, ehull_ALIGNN_relaxed_held_series_1, ehull_ALIGNN_relaxed_held_series_2]
        else:
            # CGCNN unrelaxed (first column)
            temp_dfs = [ehull_CGCNN_unrelaxed_held_series_0, ehull_CGCNN_unrelaxed_held_series_1, ehull_CGCNN_unrelaxed_held_series_2]

        # Filter data for specific perovskite composition
        temp_dfs_cuts = []
        for k in range(len(temp_dfs)):
            temp_dfs_cuts.append(temp_dfs[k][
                temp_dfs[k].formula.str.contains(series_symbols[i][0][0]) &
                temp_dfs[k].formula.str.contains(series_symbols[i][0][1]) &
                temp_dfs[k].formula.str.contains(series_symbols[i][1][0]) &
                temp_dfs[k].formula.str.contains(series_symbols[i][1][1])
            ])

        # Determine target property (DFT or predicted)
        if j == 6:
            target_prop = 'dft_e_hull'
        else:
            target_prop = 'predicted_dft_e_hull'

        # Prepare data for violin plot
        to_plot = pd.DataFrame(columns=['conc', 'entry'])
        counter = 0
        
        for (framework, subdf_0), (_, subdf_1), (_, subdf_2) in zip(temp_dfs_cuts[0].groupby('framework'), temp_dfs_cuts[1].groupby('framework'), temp_dfs_cuts[2].groupby('framework')):
            conc = float(re.findall(r'%s(0\.\d+)' % series_symbols[i][0][0], framework)[0])
            # Convert flattened lists to numpy arrays for arithmetic operations
            subdf_0_flat = np.array(flatten(subdf_0[target_prop]))
            subdf_1_flat = np.array(flatten(subdf_1[target_prop]))
            subdf_2_flat = np.array(flatten(subdf_2[target_prop]))
            subdf = (subdf_0_flat + subdf_1_flat + subdf_2_flat) / 3.0
            for entry in subdf:
                to_plot.loc[counter] = [conc, entry*1000]
                counter += 1
        
        # Create violin plot
        sns.violinplot(x='conc', y='entry', data=to_plot, ax=axes[i][j], inner=None, scale="width", linewidth=0.5, palette='crest', width=0.6)
        
        # Set axis properties
        axes[i][j].set(xlabel=None, ylabel=None)
        axes[i][j].set_xlim([-0.5, 6.5])
        axes[i][j].set_xticks(np.arange(1, 7, 2))

# Format axes
for i in range(len(subfigs)):
    if i != 5:
        for j in range(len(axes[i])):
            axes[i][j].set_xticklabels([])

    for j in range(len(axes[i])):
        ylim = axes[i][j].get_ylim()
        axes[i][j].set_ylim([ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0])])

# Save the plot as PNG
png_path = './figures/CGCNN_ALIGNN_generalizability.png'
plt.savefig(png_path, bbox_inches='tight', dpi=300)
print(f"Plot saved as PNG: {png_path}")

plt.show()
print("Plotting complete!")
