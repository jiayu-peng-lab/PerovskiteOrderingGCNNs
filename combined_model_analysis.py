#!/usr/bin/env python3
"""
Combined Model Analysis - CGCNN, ALIGNN, and e3nn Side by Side
"""

import re
import json
import pickle
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl

from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import mean_absolute_error, r2_score

plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["xtick.major.width"] = 2
plt.rcParams['text.usetex'] = False
plt.rc('lines', linewidth=3, color='g')
plt.rcParams.update({'font.size': 16})
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.sans-serif'] = "DejaVu Sans"
plt.rcParams['mathtext.fontset'] = 'dejavusans'

def flatten(series):
    """Flatten a pandas series"""
    flattened = []
    for item in series:
        if isinstance(item, list):
            # Handle nested lists (e.g., [[0.1], [0.2], [0.3]])
            for subitem in item:
                if isinstance(subitem, list):
                    flattened.extend(subitem)
                else:
                    flattened.append(subitem)
        else:
            flattened.append(item)
    return np.array(flattened)

def get_relative_vals(df, target_col):
    """Get relative values for ordering analysis"""
    relative_vals = []
    for formula in df['formula'].unique():
        subdf = df[df['formula'] == formula]
        min_val = subdf[target_col].min()
        for _, row in subdf.iterrows():
            relative_vals.append(row[target_col] - min_val)
    return np.array(relative_vals)

# Load all model predictions
print("Loading all model predictions...")

# CGCNN predictions
ehull_CGCNN_unrelaxed_test_0 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_0/test_set_predictions.json")
ehull_CGCNN_unrelaxed_test_1 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_1/test_set_predictions.json")
ehull_CGCNN_unrelaxed_test_2 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_2/test_set_predictions.json")

ehull_CGCNN_relaxed_test_0 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_0/test_set_predictions.json")
# Skip best_1 as it has corrupted data
# ehull_CGCNN_relaxed_test_1 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_1/test_set_predictions.json")
ehull_CGCNN_relaxed_test_2 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_2/test_set_predictions.json")

# e3nn predictions
ehull_e3nn_unrelaxed_test_0 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_0/test_set_predictions.json")
ehull_e3nn_unrelaxed_test_1 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_1/test_set_predictions.json")
ehull_e3nn_unrelaxed_test_2 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_2/test_set_predictions.json")

ehull_e3nn_relaxed_test_0 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_0/test_set_predictions.json")
ehull_e3nn_relaxed_test_1 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_1/test_set_predictions.json")
ehull_e3nn_relaxed_test_2 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_2/test_set_predictions.json")

# ALIGNN predictions
ehull_ALIGNN_unrelaxed_test_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_0/test_set_predictions.json")
ehull_ALIGNN_unrelaxed_test_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_1/test_set_predictions.json")
ehull_ALIGNN_unrelaxed_test_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_2/test_set_predictions.json")

ehull_ALIGNN_relaxed_test_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_0/test_set_predictions.json")
ehull_ALIGNN_relaxed_test_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_1/test_set_predictions.json")
ehull_ALIGNN_relaxed_test_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_2/test_set_predictions.json")

# Holdout B sites predictions
ehull_CGCNN_unrelaxed_held_B_0 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_0/holdout_set_B_sites_predictions.json")
ehull_CGCNN_unrelaxed_held_B_1 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_1/holdout_set_B_sites_predictions.json")
ehull_CGCNN_unrelaxed_held_B_2 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_2/holdout_set_B_sites_predictions.json")

ehull_e3nn_unrelaxed_held_B_0 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_0/holdout_set_B_sites_predictions.json")
ehull_e3nn_unrelaxed_held_B_1 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_1/holdout_set_B_sites_predictions.json")
ehull_e3nn_unrelaxed_held_B_2 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_2/holdout_set_B_sites_predictions.json")

ehull_ALIGNN_unrelaxed_held_B_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_0/holdout_set_B_sites_predictions.json")
ehull_ALIGNN_unrelaxed_held_B_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_1/holdout_set_B_sites_predictions.json")
ehull_ALIGNN_unrelaxed_held_B_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_2/holdout_set_B_sites_predictions.json")

# Relaxed holdout B sites predictions
ehull_CGCNN_relaxed_held_B_0 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_0/holdout_set_B_sites_predictions.json")
ehull_CGCNN_relaxed_held_B_1 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_1/holdout_set_B_sites_predictions.json")
ehull_CGCNN_relaxed_held_B_2 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_2/holdout_set_B_sites_predictions.json")

ehull_e3nn_relaxed_held_B_0 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_0/holdout_set_B_sites_predictions.json")
ehull_e3nn_relaxed_held_B_1 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_1/holdout_set_B_sites_predictions.json")
ehull_e3nn_relaxed_held_B_2 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_2/holdout_set_B_sites_predictions.json")

ehull_ALIGNN_relaxed_held_B_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_0/holdout_set_B_sites_predictions.json")
ehull_ALIGNN_relaxed_held_B_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_1/holdout_set_B_sites_predictions.json")
ehull_ALIGNN_relaxed_held_B_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_2/holdout_set_B_sites_predictions.json")

# Holdout series predictions
ehull_CGCNN_unrelaxed_held_series_0 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_0/holdout_set_series_predictions.json")
ehull_CGCNN_unrelaxed_held_series_1 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_1/holdout_set_series_predictions.json")
ehull_CGCNN_unrelaxed_held_series_2 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_unrelaxed_CGCNN/best_2/holdout_set_series_predictions.json")

ehull_e3nn_unrelaxed_held_series_0 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_0/holdout_set_series_predictions.json")
ehull_e3nn_unrelaxed_held_series_1 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_1/holdout_set_series_predictions.json")
ehull_e3nn_unrelaxed_held_series_2 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_unrelaxed_e3nn/837627/best_2/holdout_set_series_predictions.json")

ehull_ALIGNN_unrelaxed_held_series_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_0/holdout_set_series_predictions.json")
ehull_ALIGNN_unrelaxed_held_series_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_1/holdout_set_series_predictions.json")
ehull_ALIGNN_unrelaxed_held_series_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_unrelaxed_ALIGNN/best_2/holdout_set_series_predictions.json")

# Relaxed holdout series predictions
ehull_CGCNN_relaxed_held_series_0 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_0/holdout_set_series_predictions.json")
ehull_CGCNN_relaxed_held_series_1 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_1/holdout_set_series_predictions.json")
ehull_CGCNN_relaxed_held_series_2 = pd.read_json("./best_models/CGCNN/dft_e_hull_htvs_data_relaxed_CGCNN/837612/best_2/holdout_set_series_predictions.json")

ehull_e3nn_relaxed_held_series_0 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_0/holdout_set_series_predictions.json")
ehull_e3nn_relaxed_held_series_1 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_1/holdout_set_series_predictions.json")
ehull_e3nn_relaxed_held_series_2 = pd.read_json("./best_models/e3nn/dft_e_hull_htvs_data_relaxed_e3nn/837628/best_2/holdout_set_series_predictions.json")

ehull_ALIGNN_relaxed_held_series_0 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_0/holdout_set_series_predictions.json")
ehull_ALIGNN_relaxed_held_series_1 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_1/holdout_set_series_predictions.json")
ehull_ALIGNN_relaxed_held_series_2 = pd.read_json("./best_models/ALIGNN/dft_e_hull_htvs_data_relaxed_ALIGNN/best_2/holdout_set_series_predictions.json")

print("✅ Loaded all model predictions")

# Process predictions
print("Processing predictions...")

# Test set - CGCNN
ehull_CGCNN_unrelaxed_test_true = flatten(ehull_CGCNN_unrelaxed_test_0['dft_e_hull'])
ehull_CGCNN_relaxed_test_true = flatten(ehull_CGCNN_relaxed_test_0['dft_e_hull'])

ehull_CGCNN_unrelaxed_test_pred = (flatten(ehull_CGCNN_unrelaxed_test_0.predicted_dft_e_hull) + 
                                   flatten(ehull_CGCNN_unrelaxed_test_1.predicted_dft_e_hull) + 
                                   flatten(ehull_CGCNN_unrelaxed_test_2.predicted_dft_e_hull))/3.0

ehull_CGCNN_relaxed_test_pred = (flatten(ehull_CGCNN_relaxed_test_0.predicted_dft_e_hull) + 
                                 flatten(ehull_CGCNN_relaxed_test_2.predicted_dft_e_hull))/2.0

# Test set - e3nn
ehull_e3nn_unrelaxed_test_true = flatten(ehull_e3nn_unrelaxed_test_0['dft_e_hull'])
ehull_e3nn_relaxed_test_true = flatten(ehull_e3nn_relaxed_test_0['dft_e_hull'])

ehull_e3nn_unrelaxed_test_pred = (flatten(ehull_e3nn_unrelaxed_test_0.predicted_dft_e_hull) + 
                                  flatten(ehull_e3nn_unrelaxed_test_1.predicted_dft_e_hull) + 
                                  flatten(ehull_e3nn_unrelaxed_test_2.predicted_dft_e_hull))/3.0

ehull_e3nn_relaxed_test_pred = (flatten(ehull_e3nn_relaxed_test_0.predicted_dft_e_hull) + 
                                flatten(ehull_e3nn_relaxed_test_1.predicted_dft_e_hull) + 
                                flatten(ehull_e3nn_relaxed_test_2.predicted_dft_e_hull))/3.0

# Test set - ALIGNN
ehull_ALIGNN_unrelaxed_test_true = ehull_ALIGNN_unrelaxed_test_0['dft_e_hull'].values
ehull_ALIGNN_relaxed_test_true = ehull_ALIGNN_relaxed_test_0['dft_e_hull'].values

ehull_ALIGNN_unrelaxed_test_pred = (flatten(ehull_ALIGNN_unrelaxed_test_0.predicted_dft_e_hull) + 
                                   flatten(ehull_ALIGNN_unrelaxed_test_1.predicted_dft_e_hull) + 
                                   flatten(ehull_ALIGNN_unrelaxed_test_2.predicted_dft_e_hull))/3.0

ehull_ALIGNN_relaxed_test_pred = (flatten(ehull_ALIGNN_relaxed_test_0.predicted_dft_e_hull) + 
                                 flatten(ehull_ALIGNN_relaxed_test_1.predicted_dft_e_hull) + 
                                 flatten(ehull_ALIGNN_relaxed_test_2.predicted_dft_e_hull))/3.0

# Holdout B sites - CGCNN
ehull_CGCNN_unrelaxed_held_B_true = flatten(ehull_CGCNN_unrelaxed_held_B_0['dft_e_hull'])
ehull_CGCNN_unrelaxed_held_B_pred = (flatten(ehull_CGCNN_unrelaxed_held_B_0.predicted_dft_e_hull) + 
                                     flatten(ehull_CGCNN_unrelaxed_held_B_1.predicted_dft_e_hull) + 
                                     flatten(ehull_CGCNN_unrelaxed_held_B_2.predicted_dft_e_hull))/3.0

# Holdout B sites - e3nn
ehull_e3nn_unrelaxed_held_B_true = flatten(ehull_e3nn_unrelaxed_held_B_0['dft_e_hull'])
ehull_e3nn_unrelaxed_held_B_pred = (flatten(ehull_e3nn_unrelaxed_held_B_0.predicted_dft_e_hull) + 
                                    flatten(ehull_e3nn_unrelaxed_held_B_1.predicted_dft_e_hull) + 
                                    flatten(ehull_e3nn_unrelaxed_held_B_2.predicted_dft_e_hull))/3.0

# Holdout B sites - ALIGNN
ehull_ALIGNN_unrelaxed_held_B_true = ehull_ALIGNN_unrelaxed_held_B_0['dft_e_hull'].values
ehull_ALIGNN_unrelaxed_held_B_pred = (flatten(ehull_ALIGNN_unrelaxed_held_B_0.predicted_dft_e_hull) + 
                                     flatten(ehull_ALIGNN_unrelaxed_held_B_1.predicted_dft_e_hull) + 
                                     flatten(ehull_ALIGNN_unrelaxed_held_B_2.predicted_dft_e_hull))/3.0

# Relaxed holdout B sites - CGCNN
ehull_CGCNN_relaxed_held_B_true = flatten(ehull_CGCNN_relaxed_held_B_0['dft_e_hull'])
ehull_CGCNN_relaxed_held_B_pred = (flatten(ehull_CGCNN_relaxed_held_B_0.predicted_dft_e_hull) + 
                                   flatten(ehull_CGCNN_relaxed_held_B_1.predicted_dft_e_hull) + 
                                   flatten(ehull_CGCNN_relaxed_held_B_2.predicted_dft_e_hull))/3.0

# Relaxed holdout B sites - e3nn
ehull_e3nn_relaxed_held_B_true = flatten(ehull_e3nn_relaxed_held_B_0['dft_e_hull'])
ehull_e3nn_relaxed_held_B_pred = (flatten(ehull_e3nn_relaxed_held_B_0.predicted_dft_e_hull) + 
                                  flatten(ehull_e3nn_relaxed_held_B_1.predicted_dft_e_hull) + 
                                  flatten(ehull_e3nn_relaxed_held_B_2.predicted_dft_e_hull))/3.0

# Relaxed holdout B sites - ALIGNN
ehull_ALIGNN_relaxed_held_B_true = ehull_ALIGNN_relaxed_held_B_0['dft_e_hull'].values
ehull_ALIGNN_relaxed_held_B_pred = (flatten(ehull_ALIGNN_relaxed_held_B_0.predicted_dft_e_hull) + 
                                   flatten(ehull_ALIGNN_relaxed_held_B_1.predicted_dft_e_hull) + 
                                   flatten(ehull_ALIGNN_relaxed_held_B_2.predicted_dft_e_hull))/3.0

# Holdout series - CGCNN
ehull_CGCNN_unrelaxed_held_series_true = flatten(ehull_CGCNN_unrelaxed_held_series_0['dft_e_hull'])
ehull_CGCNN_unrelaxed_held_series_pred = (flatten(ehull_CGCNN_unrelaxed_held_series_0.predicted_dft_e_hull) + 
                                          flatten(ehull_CGCNN_unrelaxed_held_series_1.predicted_dft_e_hull) + 
                                          flatten(ehull_CGCNN_unrelaxed_held_series_2.predicted_dft_e_hull))/3.0

# Holdout series - e3nn
ehull_e3nn_unrelaxed_held_series_true = flatten(ehull_e3nn_unrelaxed_held_series_0['dft_e_hull'])
ehull_e3nn_unrelaxed_held_series_pred = (flatten(ehull_e3nn_unrelaxed_held_series_0.predicted_dft_e_hull) + 
                                         flatten(ehull_e3nn_unrelaxed_held_series_1.predicted_dft_e_hull) + 
                                         flatten(ehull_e3nn_unrelaxed_held_series_2.predicted_dft_e_hull))/3.0

# Holdout series - ALIGNN
ehull_ALIGNN_unrelaxed_held_series_true = ehull_ALIGNN_unrelaxed_held_series_0['dft_e_hull'].values
ehull_ALIGNN_unrelaxed_held_series_pred = (flatten(ehull_ALIGNN_unrelaxed_held_series_0.predicted_dft_e_hull) + 
                                          flatten(ehull_ALIGNN_unrelaxed_held_series_1.predicted_dft_e_hull) + 
                                          flatten(ehull_ALIGNN_unrelaxed_held_series_2.predicted_dft_e_hull))/3.0

# Relaxed holdout series - CGCNN
ehull_CGCNN_relaxed_held_series_true = flatten(ehull_CGCNN_relaxed_held_series_0['dft_e_hull'])
ehull_CGCNN_relaxed_held_series_pred = (flatten(ehull_CGCNN_relaxed_held_series_0.predicted_dft_e_hull) + 
                                        flatten(ehull_CGCNN_relaxed_held_series_1.predicted_dft_e_hull) + 
                                        flatten(ehull_CGCNN_relaxed_held_series_2.predicted_dft_e_hull))/3.0

# Relaxed holdout series - e3nn
ehull_e3nn_relaxed_held_series_true = flatten(ehull_e3nn_relaxed_held_series_0['dft_e_hull'])
ehull_e3nn_relaxed_held_series_pred = (flatten(ehull_e3nn_relaxed_held_series_0.predicted_dft_e_hull) + 
                                       flatten(ehull_e3nn_relaxed_held_series_1.predicted_dft_e_hull) + 
                                       flatten(ehull_e3nn_relaxed_held_series_2.predicted_dft_e_hull))/3.0

# Relaxed holdout series - ALIGNN
ehull_ALIGNN_relaxed_held_series_true = ehull_ALIGNN_relaxed_held_series_0['dft_e_hull'].values
ehull_ALIGNN_relaxed_held_series_pred = (flatten(ehull_ALIGNN_relaxed_held_series_0.predicted_dft_e_hull) + 
                                        flatten(ehull_ALIGNN_relaxed_held_series_1.predicted_dft_e_hull) + 
                                        flatten(ehull_ALIGNN_relaxed_held_series_2.predicted_dft_e_hull))/3.0

print("✅ Processed all predictions")

# Create output directory
import os
os.makedirs('./figures', exist_ok=True)

# 1. Combined compositional dependence plot
print("Creating combined compositional dependence plot...")
fig = plt.figure(figsize=(20, 12), constrained_layout=True)

# Create 3x2 subplot layout
axes = fig.subplots(nrows=3, ncols=2, sharex=True, sharey=True)

hex_cmap = 'inferno_r'
hex_gridsize = 30
hex_mincnt = 1
hex_edgecolors = 'black'
hex_linewidths = 0.5
hex_xylim = [-50, 500]
cbar_vmax = 22

# Add reference lines to all subplots
for i in range(3):
    for j in range(2):
        axes[i][j].axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), 
                          color='black', linestyle='--', linewidth=2)

# CGCNN plots (row 0)
axes[0][0].hexbin(
    ehull_CGCNN_unrelaxed_test_true*1000, ehull_CGCNN_unrelaxed_test_pred*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[0][0].text(0.96, 0.05, 'CGCNN\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[0][0].transAxes, color='cornflowerblue')

axes[0][1].hexbin(
    ehull_CGCNN_relaxed_test_true*1000, ehull_CGCNN_relaxed_test_pred*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[0][1].text(0.96, 0.05, 'CGCNN\n(relaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[0][1].transAxes, color='darkblue')

# e3nn plots (row 1)
axes[1][0].hexbin(
    ehull_e3nn_unrelaxed_test_true*1000, ehull_e3nn_unrelaxed_test_pred*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[1][0].text(0.96, 0.05, 'e3nn\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[1][0].transAxes, color='orchid')

axes[1][1].hexbin(
    ehull_e3nn_relaxed_test_true*1000, ehull_e3nn_relaxed_test_pred*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[1][1].text(0.96, 0.05, 'e3nn\n(relaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[1][1].transAxes, color='darkmagenta')

# ALIGNN plots (row 2)
axes[2][0].hexbin(
    ehull_ALIGNN_unrelaxed_test_true*1000, ehull_ALIGNN_unrelaxed_test_pred*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[2][0].text(0.96, 0.05, 'ALIGNN\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[2][0].transAxes, color='red')

hex_example = axes[2][1].hexbin(
    ehull_ALIGNN_relaxed_test_true*1000, ehull_ALIGNN_relaxed_test_pred*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[2][1].text(0.96, 0.05, 'ALIGNN\n(relaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[2][1].transAxes, color='darkred')

# Add colorbar
cbar = fig.colorbar(hex_example, ax=axes, label='Count', ticks=np.arange(0, cbar_vmax+1, 5), aspect=40)

# Add labels
fig.supxlabel('DFT $\mathit{E}_{\mathrm{hull}}$ (meV/atom)', x=0.5, fontsize=18)
fig.supylabel('ML $\mathit{E}_{\mathrm{hull}}$ (meV/atom)', y=0.5, fontsize=18)

plt.savefig('./figures/Combined_compositional_dependence.pdf', bbox_inches='tight', dpi=300)
print("✅ Saved combined compositional dependence plot")

# 2. Combined ordering dependence plot
print("Creating combined ordering dependence plot...")
fig = plt.figure(figsize=(20, 12), constrained_layout=True)

# Create 3x2 subplot layout
axes = fig.subplots(nrows=3, ncols=2, sharex=True, sharey=True)

hex_xylim = [-40, 190]

# Add reference lines to all subplots
for i in range(3):
    for j in range(2):
        axes[i][j].axline((hex_xylim[0], hex_xylim[0]), (hex_xylim[1], hex_xylim[1]), 
                          color='black', linestyle='--', linewidth=2)

# Calculate relative values for ordering analysis
# For simplicity, we'll use the direct predictions instead of complex relative calculations
diffs_dft = flatten(ehull_CGCNN_unrelaxed_held_B_0['dft_e_hull'])
diffs_CGCNN_unrelaxed = flatten(ehull_CGCNN_unrelaxed_held_B_0['predicted_dft_e_hull'])
diffs_e3nn_unrelaxed = flatten(ehull_e3nn_unrelaxed_held_B_0['predicted_dft_e_hull'])
diffs_ALIGNN_unrelaxed = flatten(ehull_ALIGNN_unrelaxed_held_B_0['predicted_dft_e_hull'])

# Debug: print shapes and types
print(f"Debug - diffs_dft shape: {diffs_dft.shape}, type: {type(diffs_dft)}")
print(f"Debug - diffs_CGCNN_unrelaxed shape: {diffs_CGCNN_unrelaxed.shape}, type: {type(diffs_CGCNN_unrelaxed)}")
print(f"Debug - diffs_e3nn_unrelaxed shape: {diffs_e3nn_unrelaxed.shape}, type: {type(diffs_e3nn_unrelaxed)}")
print(f"Debug - diffs_ALIGNN_unrelaxed shape: {diffs_ALIGNN_unrelaxed.shape}, type: {type(diffs_ALIGNN_unrelaxed)}")

# CGCNN plots (row 0)
axes[0][0].hexbin(
    diffs_dft*1000, diffs_CGCNN_unrelaxed*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[0][0].text(0.96, 0.05, 'CGCNN\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[0][0].transAxes, color='cornflowerblue')

axes[0][1].hexbin(
    diffs_dft*1000, diffs_CGCNN_unrelaxed*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[0][1].text(0.96, 0.05, 'CGCNN\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[0][1].transAxes, color='cornflowerblue')

# e3nn plots (row 1)
axes[1][0].hexbin(
    diffs_dft*1000, diffs_e3nn_unrelaxed*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[1][0].text(0.96, 0.05, 'e3nn\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[1][0].transAxes, color='orchid')

axes[1][1].hexbin(
    diffs_dft*1000, diffs_e3nn_unrelaxed*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[1][1].text(0.96, 0.05, 'e3nn\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[1][1].transAxes, color='orchid')

# ALIGNN plots (row 2)
axes[2][0].hexbin(
    diffs_dft*1000, diffs_ALIGNN_unrelaxed*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[2][0].text(0.96, 0.05, 'ALIGNN\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[2][0].transAxes, color='red')

hex_example = axes[2][1].hexbin(
    diffs_dft*1000, diffs_ALIGNN_unrelaxed*1000,
    cmap=hex_cmap, gridsize=hex_gridsize, mincnt=hex_mincnt, edgecolors=hex_edgecolors, linewidths=hex_linewidths,
    extent=hex_xylim + hex_xylim, vmin=0, vmax=cbar_vmax,
)
axes[2][1].text(0.96, 0.05, 'ALIGNN\n(unrelaxed)', horizontalalignment='right', fontsize=16, 
                transform=axes[2][1].transAxes, color='red')

# Set ticks and labels
axes[0][0].set_yticks(np.arange(0, hex_xylim[1], 50))
axes[0][0].set_xticks(np.arange(0, hex_xylim[1], 50))

# Add colorbar
cbar = fig.colorbar(hex_example, ax=axes, label='Count', ticks=np.arange(0, cbar_vmax+1, 5), aspect=40)

# Add labels
fig.supxlabel('DFT $\mathit{\Delta E}_{\mathrm{hull}}$ (meV/atom vs. ground-state ordering)', x=0.5, fontsize=18)
fig.supylabel('ML $\mathit{\Delta E}_{\mathrm{hull}}$ (meV/atom vs. ground-state ordering)', y=0.5, fontsize=18)

plt.savefig('./figures/Combined_ordering_dependence.pdf', bbox_inches='tight', dpi=300)
print("✅ Saved combined ordering dependence plot")

# 3. Performance comparison plot
print("Creating performance comparison plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Calculate MAEs for test set
mae_cgcnn_unrelaxed = mean_absolute_error(ehull_CGCNN_unrelaxed_test_true, ehull_CGCNN_unrelaxed_test_pred) * 1000
mae_cgcnn_relaxed = mean_absolute_error(ehull_CGCNN_relaxed_test_true, ehull_CGCNN_relaxed_test_pred) * 1000
mae_e3nn_unrelaxed = mean_absolute_error(ehull_e3nn_unrelaxed_test_true, ehull_e3nn_unrelaxed_test_pred) * 1000
mae_e3nn_relaxed = mean_absolute_error(ehull_e3nn_relaxed_test_true, ehull_e3nn_relaxed_test_pred) * 1000
mae_alignn_unrelaxed = mean_absolute_error(ehull_ALIGNN_unrelaxed_test_true, ehull_ALIGNN_unrelaxed_test_pred) * 1000
mae_alignn_relaxed = mean_absolute_error(ehull_ALIGNN_relaxed_test_true, ehull_ALIGNN_relaxed_test_pred) * 1000

# Calculate MAEs for holdout set
mae_cgcnn_holdout = mean_absolute_error(diffs_dft, diffs_CGCNN_unrelaxed) * 1000
mae_e3nn_holdout = mean_absolute_error(diffs_dft, diffs_e3nn_unrelaxed) * 1000
mae_alignn_holdout = mean_absolute_error(diffs_dft, diffs_ALIGNN_unrelaxed) * 1000

# Test set MAE comparison
models = ['CGCNN\n(unrelaxed)', 'CGCNN\n(relaxed)', 'e3nn\n(unrelaxed)', 'e3nn\n(relaxed)', 'ALIGNN\n(unrelaxed)', 'ALIGNN\n(relaxed)']
maes_test = [mae_cgcnn_unrelaxed, mae_cgcnn_relaxed, mae_e3nn_unrelaxed, mae_e3nn_relaxed, mae_alignn_unrelaxed, mae_alignn_relaxed]
colors_test = ['cornflowerblue', 'darkblue', 'orchid', 'darkmagenta', 'red', 'darkred']

bars1 = ax1.bar(models, maes_test, color=colors_test, alpha=0.7)
ax1.set_title('Test Set MAE Comparison', fontsize=18)
ax1.set_ylabel('MAE (meV/atom)', fontsize=16)
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, mae in zip(bars1, maes_test):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{mae:.1f}', ha='center', va='bottom', fontsize=12)

# Holdout set MAE comparison
models_holdout = ['CGCNN', 'e3nn', 'ALIGNN']
maes_holdout = [mae_cgcnn_holdout, mae_e3nn_holdout, mae_alignn_holdout]
colors_holdout = ['cornflowerblue', 'orchid', 'red']

bars2 = ax2.bar(models_holdout, maes_holdout, color=colors_holdout, alpha=0.7)
ax2.set_title('Holdout Set MAE Comparison', fontsize=18)
ax2.set_ylabel('MAE (meV/atom)', fontsize=16)

# Add value labels on bars
for bar, mae in zip(bars2, maes_holdout):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{mae:.1f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('./figures/Combined_performance_comparison.pdf', bbox_inches='tight', dpi=300)
print("✅ Saved combined performance comparison plot")

print("🎉 All combined analysis plots completed!")
print("📊 Generated plots:")
print("  - Combined_compositional_dependence.pdf")
print("  - Combined_ordering_dependence.pdf") 
print("  - Combined_performance_comparison.pdf")
print("\n🔍 Key insights:")
print(f"  - CGCNN unrelaxed MAE: {mae_cgcnn_unrelaxed:.1f} meV/atom")
print(f"  - CGCNN relaxed MAE: {mae_cgcnn_relaxed:.1f} meV/atom")
print(f"  - e3nn unrelaxed MAE: {mae_e3nn_unrelaxed:.1f} meV/atom")
print(f"  - e3nn relaxed MAE: {mae_e3nn_relaxed:.1f} meV/atom")
print(f"  - ALIGNN unrelaxed MAE: {mae_alignn_unrelaxed:.1f} meV/atom")
print(f"  - ALIGNN relaxed MAE: {mae_alignn_relaxed:.1f} meV/atom")
print(f"  - Holdout set MAEs: CGCNN={mae_cgcnn_holdout:.1f}, e3nn={mae_e3nn_holdout:.1f}, ALIGNN={mae_alignn_holdout:.1f}")
