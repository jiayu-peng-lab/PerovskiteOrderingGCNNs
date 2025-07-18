import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from pymatgen.core import Structure

# from training.sigopt_utils import build_sigopt_name  # Original SigOpt utils (commented out)
from training.wandb_utils import build_wandb_name  # Wandb utils (active)
from inference.select_best_models import get_experiment_id
from matplotlib.patches import FancyArrowPatch


def get_datapoint(target_prop, model, interp, training_fraction, struct):
    model_params = {}
    model_params["data"]= "data/"
    model_params["struct_type"] = struct
    model_params["interpolation"] = interp
    model_params["model_type"] = model
    model_params["contrastive_weight"]= 1.0
    model_params["training_fraction"]=training_fraction
    model_params["long_range"]=False
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    directory = "./best_models/" + model_params["model_type"] + "/" + wandb_name + "/best_"

    data_0 = pd.read_json(directory + "0" + "/test_set_predictions.json")
    data_1 = pd.read_json(directory + "1" + "/test_set_predictions.json")
    data_2 = pd.read_json(directory + "2" + "/test_set_predictions.json")
    
    pred_0 = np.asarray(flatten(list(data_0["predicted_" + target_prop]))).reshape(-1).flatten()
    pred_1 = np.asarray(flatten(list(data_1["predicted_" + target_prop]))).reshape(-1).flatten()
    pred_2 = np.asarray(flatten(list(data_2["predicted_" + target_prop]))).reshape(-1).flatten()
    
    errors = [np.mean(np.abs(pred_0 - data_0[target_prop])), 
              np.mean(np.abs(pred_1 - data_1[target_prop])), 
              np.mean(np.abs(pred_2 - data_2[target_prop]))
             ]
    errors = np.asarray(errors)
    
    return errors.mean(), errors.std()


def get_series(prop, model, interp, struct):
    training_fraction = [1.0, 0.5, 0.25, 0.125]  
    means = []
    stds = []
    
    for frac in training_fraction:
        curr_mean, curr_std = get_datapoint(prop, model, interp, frac, struct)
        means.append(curr_mean)
        stds.append(curr_std)
        
    means = np.asarray(means)
    stds = np.asarray(stds)
    return means, stds


def get_property(prop, struct):
    CGCNN = get_series(prop, "CGCNN", False, struct)
    # e3nn = get_series(prop, "e3nn", False, struct)  # Commented out as per new requirements
    
    return CGCNN, None


def flatten(matrix):
    matrix = list(matrix)
    if isinstance(matrix[0],list):
        out = [item for row in matrix for item in row]
    else:
        out = matrix
    return np.asarray(out)


def get_relative_vals(dataframe, vals):  
    vals_gs = {}
    out_diffs = []
    
    for i in range(len(dataframe)):
        curr_formula = dataframe.iloc[i].formula
        if curr_formula in vals_gs:
            if vals[i] < vals_gs[curr_formula]:
                vals_gs[curr_formula] = vals[i]
        else:
            vals_gs[curr_formula] = vals[i]
        
    for i in range(len(dataframe)):
        curr_formula = dataframe.iloc[i].formula
        curr_val = vals[i] - vals_gs[curr_formula]
        out_diffs.append(curr_val)
              
    return np.asarray(out_diffs)


def check_if_rocksalt(struct, B_elements):
    struct = Structure.from_dict(struct) 
    for i in range(len(struct.species)):
        if str(struct.species[i]) in B_elements:
            for j in range(len(struct)):
                diff = struct[i].coords - struct[j].coords
                dist = np.sqrt(np.sum(diff*diff))
                if abs(dist - 4.0) < 0.1 and struct.species[j]==struct.species[i]:
                    return False               
    return True


def get_is_rocksalt(df):
    is_rocksalt = []    
    for i in range(len(df)):
        is_rocksalt.append(check_if_rocksalt(df.iloc[i]["unrelaxed_struct"], df.iloc[i].composition["sites"]["B"]))
    return is_rocksalt


def check_if_layered(struct, B_elements):
    struct = Structure.from_dict(struct)
    for i in range(len(struct.species)):
        if str(struct.species[i]) in B_elements:
            count_NN = 0
            count_Next_NN = 0
            for j in range(len(struct)):
                diff = struct[i].coords - struct[j].coords
                dist = np.sqrt(np.sum(diff*diff))
                if struct.species[j] == struct.species[i]:
                    if abs(dist - 4.0)<0.01:
                        count_NN += 1
                    elif abs(dist - 5.6568)<0.01:
                        count_Next_NN += 1                        
            if count_NN != 2 or count_Next_NN != 1:
                return False
    return True


def get_is_layered(df):
    is_layered = []    
    for i in range(len(df)):
        is_layered.append(check_if_layered(df.iloc[i]["unrelaxed_struct"], df.iloc[i].composition["sites"]["B"]))
    return is_layered


def check_if_column(struct, B_elements):
    struct = Structure.from_dict(struct)
    for i in range(len(struct.species)):
        if str(struct.species[i]) in B_elements:
            count_NN = 0
            count_Next_NN = 0
            for j in range(len(struct)):
                diff = struct[i].coords - struct[j].coords
                dist = np.sqrt(np.sum(diff*diff))
                if struct.species[j] == struct.species[i]:
                    if abs(dist - 4.0)<0.01:
                        count_NN += 1
                    elif abs(dist - 5.6568)<0.01:
                        count_Next_NN += 1   
            if count_NN != 1 or count_Next_NN != 1:
                return False
    return True


def get_is_column(df):
    is_col = []    
    for i in range(len(df)):
        is_col.append(check_if_column(df.iloc[i]["unrelaxed_struct"], df.iloc[i].composition["sites"]["B"]))
    return is_col


def embeddings_from_file(model_params, test):
    target_prop = "dft_e_hull"
    idx = 0
    exp_id = get_experiment_id(model_params, "dft_e_hull")
    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"], contrastive_weight=model_params["contrastive_weight"], training_fraction=model_params["training_fraction"], long_range=False)
    # directory = "./best_models/" + model_params["model_type"] + "/" + sigopt_name + "/" + str(exp_id) + "/" + "best_" + str(idx)
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"], contrastive_weight=model_params["contrastive_weight"], training_fraction=model_params["training_fraction"], long_range=False)
    directory = "./best_models/" + model_params["model_type"] + "/" + wandb_name + "/" + str(exp_id) + "/" + "best_" + str(idx)
    data = pd.read_json(directory + '/' + test + "_embeddings"+"_" + str(0) + ".json")
    return data


def get_np_embedding(df):
    curr_embeddings = df["embedding_0"]
    arr_list = []
    for embedding in curr_embeddings:
        arr_list.append(np.asarray(embedding[0]))
    np_embedding = np.asarray(arr_list)
    return np_embedding


def find_ind(val_list, target):
    dist = 99999
    curr_index = None
    for i in range(len(val_list)):
        if abs(val_list[i] - target) < dist:
            dist = abs(val_list[i] - target)
            curr_index = i
    return curr_index


def get_rs_vec(embeddings, pca):
    rs_vec = []
    forms = set(embeddings.formula)
    embeddings["is_rocksalt"] = get_is_rocksalt(embeddings)
    
    for form in forms:       
        curr_embeddings = embeddings[embeddings["formula"] == form]
        np_embedding = get_np_embedding(curr_embeddings)
        projection = pca.transform(np_embedding)        
        ref = projection.mean(axis=0)

        for i in range(len(curr_embeddings)):
            if curr_embeddings.iloc[i].is_rocksalt:
                rs_pos = projection[i,:]

        curr_dist = rs_pos - ref
        rs_vec.append(curr_dist)
        
    rs_vec = np.asarray(rs_vec)
    return rs_vec


def get_layered_vec(embeddings, pca):
    lay_vec = []
    forms = set(embeddings.formula)
    embeddings["is_layered"] = get_is_layered(embeddings)
    
    for form in forms:
        curr_embeddings = embeddings[embeddings["formula"] == form]
        np_embedding = get_np_embedding(curr_embeddings)
        projection = pca.transform(np_embedding)
        ref = projection.mean(axis=0)
        
        for i in range(len(curr_embeddings)):
            if curr_embeddings.iloc[i].is_layered:
                lay_pos = projection[i,:]

        curr_dist = lay_pos - ref
        lay_vec.append(curr_dist)
   
    lay_vec = np.asarray(lay_vec)
    return lay_vec


def get_col_vec(embeddings, pca):
    col_vec = []
    forms = set(embeddings.formula)
    embeddings["is_column"] = get_is_column(embeddings)
    
    for form in forms:
        curr_embeddings = embeddings[embeddings["formula"] == form]
        np_embedding = get_np_embedding(curr_embeddings)
        projection = pca.transform(np_embedding)
        ref = projection.mean(axis=0)
        
        for i in range(len(curr_embeddings)):
            if curr_embeddings.iloc[i].is_column:
                col_pos = projection[i,:]

        curr_dist = col_pos - ref
        col_vec.append(curr_dist)

    col_vec = np.asarray(col_vec)
    return col_vec


def plot_pca_embedding(ax, model_type, struct_type, highlight=False):
    param = {"struct_type": struct_type, "model_type": model_type, "data": "data/", "interpolation": False, "contrastive_weight": 1.0, "training_fraction": 1.0, "long_range": False}
    embeddings = embeddings_from_file(param, "holdout_set_B_sites")
    np_embedding = get_np_embedding(embeddings)
    pca = PCA(n_components=2)
    pca.fit(np_embedding)
    projection = pca.transform(np_embedding)
    mean_pos_all = projection.mean(axis=0)
    
    norm_dist = []
    for i in range(projection.shape[0]):
        norm_dist.append(np.linalg.norm(projection[i,:] - mean_pos_all)) 
    norm_dist = np.mean(np.asarray(norm_dist))

    stored_comps = []
    pca_spread = []
    for form in set(embeddings["formula"]):
        stored_comps.append(form)
        filtered_embeddings = embeddings[embeddings["formula"] == form]
        np_filtered_embeddings = get_np_embedding(filtered_embeddings)
        curr_projection = pca.transform(np_filtered_embeddings)
        mean_pos = curr_projection.mean(axis=0)
        tmp_pca_vals = []
        for k in range(len(curr_projection)):
            tmp_pca_vals.append(np.sqrt(np.sum((curr_projection[k,:] - mean_pos)**2)))
        pca_spread.append(np.asarray(tmp_pca_vals).mean()/norm_dist)

    if not highlight:
        selected_forms = []
    else:
        selected_forms = ['O24Ru4Sr8Y4', 'Cr4Fe4La8O24', 'Ca8Mn4O24Ta4', 'Mn4O24Pr8Rh4']

        projection_featured = []        
        for i in range(len(selected_forms)):
            curr_form = selected_forms[i]
            filtered_embeddings = embeddings[embeddings["formula"] == curr_form]
            np_filtered_embeddings = get_np_embedding(filtered_embeddings)
            curr_projection = pca.transform(np_filtered_embeddings)
            projection_featured.append(curr_projection)

    background_embeddings = embeddings[~embeddings["formula"].isin(selected_forms)]
    np_background_embeddings = get_np_embedding(background_embeddings)
    projection_background = pca.transform(np_background_embeddings)
    ax.scatter(-projection_background[:,0], -projection_background[:,1], s=30, color='black', alpha=0.1, linewidth=0)

    if highlight:
        highlight_colors = ['orangered', 'crimson', 'darkgoldenrod', 'olivedrab']
        for i in range(len(projection_featured)):
            ax.scatter(-projection_featured[i][:,0], -projection_featured[i][:,1], s=600, color=highlight_colors[i], alpha=0.8, linewidth=1, edgecolors='black', marker='*')
            pass

    avg_rs_vec = get_rs_vec(embeddings, pca).mean(axis=0)
    avg_lay_vec = get_layered_vec(embeddings, pca).mean(axis=0)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrange = xlim[1] - xlim[0]
    yrange = ylim[1] - ylim[0]

    initial_pos = (xlim[0] + 0.3*xrange, ylim[0] + 0.3*yrange)
    arrow_rs_vec = FancyArrowPatch(initial_pos, initial_pos + avg_rs_vec*2, color='black', arrowstyle='->', mutation_scale=15, linewidth=2)
    arrow_lay_vec = FancyArrowPatch(initial_pos, initial_pos + avg_lay_vec*2, color='black', arrowstyle='->', mutation_scale=15, linewidth=2)

    ax.add_patch(arrow_rs_vec)
    ax.add_patch(arrow_lay_vec)

    return pca_spread


def plot_pca_embedding_additional(ax, model_type, struct_type, selected_form):
    param = {"struct_type": struct_type, "model_type": model_type, "data": "data/", "interpolation": False, "contrastive_weight": 1.0, "training_fraction": 1.0, "long_range": False}
    embeddings = embeddings_from_file(param, "holdout_set_B_sites")
    np_embedding = get_np_embedding(embeddings)
    pca = PCA(n_components=2)
    pca.fit(np_embedding)
    projection = pca.transform(np_embedding)
    mean_pos_all = projection.mean(axis=0)
    
    norm_dist = []
    for i in range(projection.shape[0]):
        norm_dist.append(np.linalg.norm(projection[i,:] - mean_pos_all)) 
    norm_dist = np.mean(np.asarray(norm_dist))

    stored_comps = []
    pca_spread = []
    for form in set(embeddings["formula"]):
        stored_comps.append(form)
        filtered_embeddings = embeddings[embeddings["formula"] == form]
        np_filtered_embeddings = get_np_embedding(filtered_embeddings)
        curr_projection = pca.transform(np_filtered_embeddings)
        mean_pos = curr_projection.mean(axis=0)
        tmp_pca_vals = []
        for k in range(len(curr_projection)):
            tmp_pca_vals.append(np.sqrt(np.sum((curr_projection[k,:] - mean_pos)**2)))
        pca_spread.append(np.asarray(tmp_pca_vals).mean()/norm_dist)

    selected_forms = [selected_form]

    projection_featured = []        
    for i in range(len(selected_forms)):
        curr_form = selected_forms[i]
        filtered_embeddings = embeddings[embeddings["formula"] == curr_form]
        np_filtered_embeddings = get_np_embedding(filtered_embeddings)
        curr_projection = pca.transform(np_filtered_embeddings)
        projection_featured.append(curr_projection)

    background_embeddings = embeddings[~embeddings["formula"].isin(selected_forms)]
    np_background_embeddings = get_np_embedding(background_embeddings)
    projection_background = pca.transform(np_background_embeddings)
    ax.scatter(-projection_background[:,0], -projection_background[:,1], s=30, color='black', alpha=0.1, linewidth=0)

    for i in range(len(projection_featured)):
        ax.scatter(-projection_featured[i][:,0], -projection_featured[i][:,1], s=600, facecolors="white", alpha=0.8, linewidth=1, edgecolors='black', marker='*')

