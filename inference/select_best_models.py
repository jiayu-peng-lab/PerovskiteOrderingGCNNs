import os
import shutil
import json
# import sigopt  # Original SigOpt import (commented out)
import math
import torch
import random
import numpy as np
import pandas as pd
import wandb  # Wandb import (active)
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties,select_structures
from training.evaluate import evaluate_model
from processing.interpolation.Interpolation import *
from training.loss import contrastive_loss
# from training.sigopt_utils import build_sigopt_name  # Original SigOpt utils (commented out)
from training.wandb_utils import build_wandb_name  # Wandb utils (active)
from processing.create_model.create_model import create_model

saved_models_path = "./saved_models/"

def get_experiment_id(model_params, target_prop):
    """
    Get wandb experiment ID - equivalent to SigOpt experiment ID
    """
    f = open('inference/experiment_ids.json')
    settings_to_id = json.load(f)
    f.close()

    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    # print(sigopt_name)
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    print(wandb_name)

    if wandb_name in settings_to_id:
        return settings_to_id[wandb_name]
    else:
        raise ValueError('These model parameters have not been studied')

def find_wandb_folder(parent_dir):
    """
    Find the unique 'wandb-*' folder in the given parent directory.
    Raises an error if none or more than one is found.
    Returns the full path to the folder.
    """
    wandb_folders = [f for f in os.listdir(parent_dir) if f.startswith('wandb-') and os.path.isdir(os.path.join(parent_dir, f))]
    if len(wandb_folders) == 0:
        raise FileNotFoundError(f"No 'wandb-*' folder found in {parent_dir}")
    if len(wandb_folders) > 1:
        raise RuntimeError(f"Multiple 'wandb-*' folders found in {parent_dir}: {wandb_folders}")
    return os.path.join(parent_dir, wandb_folders[0])

def load_model(gpu_num, train_loader, target_prop, model_params, observ_folder_name, job_idx, per_site):
    """
    Load model - equivalent to SigOpt version but uses wandb naming
    """
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    parent_dir = saved_models_path + model_params["model_type"] + "/" + wandb_name
    wandb_dir = find_wandb_folder(parent_dir)
    directory = os.path.join(wandb_dir, observ_folder_name)

    if model_params["model_type"] == "Painn":
        model = torch.load(directory + "/best_model", map_location=device)
        normalizer = None
    else:
        with open(directory + "/hyperparameters.json") as json_file:
            assignments = json.load(json_file)
        model, normalizer = create_model(model_params["model_type"],train_loader,model_params["interpolation"],target_prop,hyperparameters=assignments,per_site=per_site)
        model.to(device)
        model.load_state_dict(torch.load(directory + "/best_model.torch", map_location=device)['state'])
    
    return model, normalizer


def reverify_wandb_models(model_params, gpu_num, target_prop="dft_e_hull"):
    """
    Reverify wandb models - equivalent to reverify_sigopt_models
    """
    model_params["data"] = "data/"
    model_params["interpolation"] = False
    model_params["contrastive_weight"] = 1.0
    model_params["long_range"] = False

    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    
    interpolation = model_params["interpolation"]
    model_type = model_params["model_type"]    
    data_name = model_params["data"]
    struct_type = model_params["struct_type"]

    if data_name == "data/":
        training_data = pd.read_json(data_name + 'training_set.json')
        training_data = training_data.sample(frac=model_params["training_fraction"],replace=False,random_state=0)
        validation_data = pd.read_json(data_name + 'validation_set.json')
        edge_data = pd.read_json(data_name + 'edge_dataset.json')

        if not interpolation:
            training_data = pd.concat((training_data,edge_data))
    else:
        print("Specified Data Directory Does Not Exist!")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print("Loaded data")

    data = [training_data, validation_data]
    processed_data = []

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)
        dataset = select_structures(dataset,struct_type)
        if interpolation:
            dataset = apply_interpolation(dataset,target_prop)
        processed_data.append(dataset)

    print("Completed data processing")
    
    train_data = processed_data[0]
    validation_data = processed_data[1]
    
    per_site = False
    if "per_site" in target_prop:
        per_site = True

    train_loader = get_dataloader(train_data,target_prop,model_type,1,interpolation,per_site=per_site,long_range=model_params["long_range"])
    val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation,per_site=per_site,long_range=model_params["long_range"])       

    reverify_wandb_models_results = pd.DataFrame(columns=['observ_folder', 'reverified_loss'])

    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    parent_dir = saved_models_path + model_params["model_type"] + "/" + wandb_name
    wandb_dir = find_wandb_folder(parent_dir)
    observ_folders = [f for f in os.listdir(wandb_dir) if f.startswith('observ_') and os.path.isdir(os.path.join(wandb_dir, f))]
    observ_folders.sort()  # sort alphabetically, or sort by another criteria if needed
    num_folders = len(observ_folders)

    for idx, folder_name in enumerate(observ_folders):
        job_idx = num_folders - idx - 1
        print('Reverifying wandb model #' + str(idx) + ' (' + folder_name + ')')
        directory = os.path.join(wandb_dir, folder_name)

        model, normalizer = load_model(gpu_num, train_loader, target_prop, model_params, folder_name, job_idx,per_site=per_site)
        
        if "contrastive" in model_type:
            loss_fn = contrastive_loss 
            is_contrast = True
        else:
            loss_fn = torch.nn.L1Loss()
            is_contrast = False
            
        _, _, best_loss = evaluate_model(model, normalizer, model_type, val_loader, loss_fn, gpu_num,is_contrastive=is_contrast,contrastive_weight=model_params["contrastive_weight"])

        if model_type == "Painn":
            reverified_loss = best_loss
        else:
            reverified_loss = best_loss[0]

        new_row = pd.DataFrame([[folder_name, reverified_loss]], columns=['observ_folder', 'reverified_loss'])

        reverify_wandb_models_results = pd.concat([
            reverify_wandb_models_results,
            new_row
        ], ignore_index=True)

    reverify_wandb_models_results.to_csv(os.path.join(wandb_dir, "reverify_wandb_models_results.csv"))


def keep_the_best_few_models(model_params, num_best_models=3, target_prop="dft_e_hull"):
    """
    Keep the best few models - equivalent to SigOpt version but uses wandb naming
    """
    model_params["data"] = "data/"
    model_params["interpolation"] = False
    model_params["contrastive_weight"] = 1.0
    model_params["long_range"] = False

    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    parent_dir = saved_models_path + model_params["model_type"] + "/" + wandb_name
    wandb_dir = find_wandb_folder(parent_dir)
    old_directory_prefix = wandb_dir

    # Use experiment ID (from inference/experiment_ids.json) as an intermediate folder when available
    try:
        exp_id = get_experiment_id(model_params, target_prop)
    except Exception:
        exp_id = None

    if isinstance(exp_id, str) and exp_id.lower() == "none":
        exp_id = None

    if exp_id is None:
        new_directory_prefix = "./best_models/" + model_params["model_type"] + "/" + wandb_name
    else:
        new_directory_prefix = "./best_models/" + model_params["model_type"] + "/" + wandb_name + "/" + str(exp_id)

    # Load the results CSV from the wandb_dir
    reverify_wandb_models_results = pd.read_csv(os.path.join(wandb_dir, 'reverify_wandb_models_results.csv'), index_col=0)

    if not os.path.exists(new_directory_prefix):
        os.makedirs(new_directory_prefix)

    best_rows = reverify_wandb_models_results.sort_values(by=['reverified_loss']).head(num_best_models)
    for i, row in enumerate(best_rows.itertuples()):
        folder_name = row[1]  # row[0] is the index, row[1] is 'observ_folder'
        old_directory = os.path.join(old_directory_prefix, folder_name)
        new_directory = os.path.join(new_directory_prefix, f"best_{i}")
        if os.path.exists(old_directory):
            shutil.copytree(old_directory, new_directory)
            print(f"Copied model {folder_name} to best_{i}")
        else:
            print(f"Model {folder_name} not found, skipping...")

    print(f"Kept the best {num_best_models} models in {new_directory_prefix}")
