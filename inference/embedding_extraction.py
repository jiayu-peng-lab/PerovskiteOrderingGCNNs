import os
import json
import torch
import pandas as pd
import numpy as np
import random
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
# from training.sigopt_utils import build_sigopt_name  # Original SigOpt utils (commented out)
from training.wandb_utils import build_wandb_name  # Wandb utils (active)
from processing.create_model.create_model import create_model
from inference.select_best_models import get_experiment_id
from inference.test_model_prediction import evaluate_model_with_tracked_ids, load_model
from nff.train.loss import build_mae_loss
from nff.train.evaluate import evaluate
from torch.autograd import Variable


def _resolve_best_models_root(model_params, target_prop, wandb_name):
    base_dir = os.path.join("./best_models", model_params["model_type"], wandb_name)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Best-model base directory not found: {base_dir}")

    try:
        exp_id = get_experiment_id(model_params, target_prop)
    except Exception:
        exp_id = None

    if isinstance(exp_id, str) and exp_id.lower() == "none":
        exp_id = None

    if exp_id is not None:
        exp_dir = os.path.join(base_dir, str(exp_id))
        if os.path.isdir(exp_dir):
            return exp_dir

    if os.path.isdir(os.path.join(base_dir, "best_0")):
        return base_dir

    candidate_dirs = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and os.path.isdir(os.path.join(full_path, "best_0")):
            candidate_dirs.append(full_path)

    if len(candidate_dirs) == 1:
        return candidate_dirs[0]
    if len(candidate_dirs) > 1:
        raise RuntimeError(
            f"Multiple best-model roots found under {base_dir}: {candidate_dirs}"
        )
    raise FileNotFoundError(
        f"No best-model root found under {base_dir} (expected 'best_0' folder)."
    )


def get_all_embeddings(model_params, gpu_num, num_best_models=3, target_prop="dft_e_hull", depth=0):
    
    model_params["data"] = "data/"
    model_params["interpolation"] = False
    model_params["contrastive_weight"] = 1.0
    model_params["long_range"] = False

    for test_set_type in ["test_set", "holdout_set_B_sites", "holdout_set_series"]:
        get_model_embedding(test_set_type, model_params, gpu_num, num_best_models, target_prop, depth)
        print("Completed embedding extraction for " + test_set_type)


def get_model_embedding(test_set_type, model_params, gpu_num, num_best_models, target_prop, depth):

    if model_params["model_type"] == "Painn":
        print("Embeddings not implemented for Painn.")
        return None

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
        test_data = pd.read_json(data_name + test_set_type + '.json')
        edge_data = pd.read_json(data_name + 'edge_dataset.json')

        if not interpolation:
            training_data = pd.concat((training_data,edge_data))

    elif data_name == "pretrain_data/":

        training_data = pd.read_json(data_name + 'training_set.json')
        test_data = pd.read_json(data_name + 'test_set.json')

    else:
        print("Specified Data Directory Does Not Exist!")
   

    print("Loaded data")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    data = [training_data, test_data]
    processed_data = []

    for dataset in data:
        dataset = filter_data_by_properties(dataset,target_prop)
        dataset = select_structures(dataset,model_params["struct_type"])

        if interpolation:
            dataset = apply_interpolation(dataset,target_prop)

        processed_data.append(dataset)

    print("Completed data processing")

    train_data = processed_data[0]
    test_data = processed_data[1]

    train_loader = get_dataloader(train_data,target_prop,model_type,1,interpolation)
    test_loader = get_dataloader(test_data,target_prop,model_type,1,interpolation)       

    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    # exp_id = get_experiment_id(model_params, target_prop)
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"])
    best_models_root = _resolve_best_models_root(model_params, target_prop, wandb_name)

    for idx in range(num_best_models):
        directory = os.path.join(best_models_root, "best_" + str(idx))
        model, normalizer = load_model(gpu_num, train_loader, model_params, directory, target_prop,per_site=False)

        activation = {}

        def hook(model, input, output):
            if "embedding" not in activation:
                activation["embedding"] = [input[0].detach()]
            else:
                activation["embedding"].append(input[0].detach())

        model_layer = get_model_layer(model,model_params["model_type"],depth)
        model_layer.register_forward_hook(hook)
        prediction,ids = evaluate_model_with_tracked_ids(model, normalizer, gpu_num, test_loader, model_params, return_ids=True)
        embeddings = activation['embedding']
        sorted_embeddings = []
        infer_embedding = test_data.copy()
        infer_embedding.drop(columns=['structure', 'ase_structure'], inplace=True)
        if model_params["model_type"] == "e3nn":
            infer_embedding.drop(columns=['datapoint'], inplace=True)
            
        for index, _ in infer_embedding.iterrows():
            for j in range(len(ids)):
                if ids[j] == index:
                    sorted_embeddings.append(embeddings[j].cpu().numpy())

        infer_embedding["embedding"+"_"+str(depth)] = sorted_embeddings

        infer_embedding.to_json(os.path.join(directory, test_set_type + "_embeddings"+"_"+str(depth)+".json"))


def get_model_layer(model,model_type,depth):
    if "e3nn" in model_type:

        if depth == 0:
            if hasattr(model, "conv_to_fc"):
                return model.conv_to_fc
            else:
                return model.fc_out

        elif depth <= len(model.fcs):
            return model.fcs[depth-1]

        elif depth == len(model.fcs)+1:
            return model.fcs_out

        else:
            print("Depth Not Supported")
            return None

    elif "ALIGNN" in model_type:

        # For ALIGNN, use the pooled graph representation right before the final FC.
        # Hooking on model.fc will provide its input (the embedding vector) via input[0].
        if hasattr(model, "fc"):
            return model.fc
        else:
            print("ALIGNN model does not have attribute 'fc'")
            return None

    elif "CGCNN" in model_type:

        if depth == 0:
            return model.conv_to_fc

        elif depth <= len(model.fcs):
            return model.fcs[depth-1]

        elif depth == len(model.fcs)+1:
            return model.fcs_out

        else:
            print("Depth Not Supported")
            return None


    else:
        print("Model Type not Supported")
        return None





        
