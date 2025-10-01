import json
import torch
import pandas as pd
import numpy as np
import random
import time
from processing.dataloader.dataloader import get_dataloader
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
# from training.sigopt_utils import build_sigopt_name  # Original SigOpt utils (commented out)
from training.wandb_utils import build_wandb_name  # Wandb utils (active)
from processing.create_model.create_model import create_model
from inference.select_best_models import get_experiment_id
from nff.train.loss import build_mae_loss
from nff.train.evaluate import evaluate
from torch.autograd import Variable


def get_all_model_predictions(model_params, gpu_num, target_prop="dft_e_hull", num_best_models=3):
    
    model_params["data"] = "data/"
    model_params["interpolation"] = False
    model_params["contrastive_weight"] = 1.0
    model_params["long_range"] = False
    
    for test_set_type in ["test_set", "holdout_set_B_sites", "holdout_set_series"]:
        get_model_prediction(test_set_type, model_params, gpu_num, target_prop, num_best_models)
        print("Completed model prediction for " + test_set_type)


def get_model_prediction(test_set_type, model_params, gpu_num, target_prop, num_best_models):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    
    start = time.time()

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
            
    # elif data_name == "data_per_site/":
    #     training_data = pd.read_json(data_name + 'training_set.json')
    #     training_data = training_data.sample(frac=model_params["training_fraction"],replace=False,random_state=0)
    #     test_data = pd.read_json(data_name + test_set_type + '.json')
    #     edge_data = pd.read_json(data_name + 'edge_dataset.json')

    #     if not interpolation:
    #         training_data = pd.concat((training_data,edge_data))

    # elif data_name == "pretrain_data/":

    #     training_data = pd.read_json(data_name + 'training_set.json')
    #     test_data = pd.read_json(data_name + 'test_set.json')

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
    
    per_site = False
    if "per_site" in target_prop:
        per_site = True

    train_loader = get_dataloader(train_data,target_prop,model_type,1,interpolation,per_site=per_site,long_range=model_params["long_range"])
    
    start_2 = time.time()
    test_loader = get_dataloader(test_data,target_prop,model_type,1,interpolation,per_site=per_site,long_range=model_params["long_range"])       
    end_2 = time.time()
    
    # Original SigOpt name building (commented out)
    # sigopt_name = build_sigopt_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])
    # exp_id = get_experiment_id(model_params, target_prop)
    
    # Wandb name building (active)
    wandb_name = build_wandb_name(model_params["data"], target_prop, model_params["struct_type"], model_params["interpolation"], model_params["model_type"],contrastive_weight=model_params["contrastive_weight"],training_fraction=model_params["training_fraction"],long_range=model_params["long_range"])

    # Try to get experiment id from inference/experiment_ids.json; if "none" or missing, omit it from the path
    try:
        exp_id = get_experiment_id(model_params, target_prop)
    except Exception:
        exp_id = None

    if isinstance(exp_id, str) and exp_id.lower() == "none":
        exp_id = None

    for idx in range(num_best_models):
        # Prefer path with exp_id when available
        if exp_id is None:
            directory = "./best_models/" + model_params["model_type"] + "/" + wandb_name + "/" + "best_" + str(idx)
        else:
            directory = "./best_models/" + model_params["model_type"] + "/" + wandb_name + "/" + str(exp_id) + "/" + "best_" + str(idx)
        model, normalizer = load_model(gpu_num, train_loader, model_params, directory, target_prop,per_site=per_site)
        start_3 = time.time()
        prediction = evaluate_model_with_tracked_ids(model, normalizer, gpu_num, test_loader, model_params)
        end_3 = time.time()
        print("Timing...")
        print(end_3-start_3)
        print(end_3-start_3 + (end_2-start_2))

        sorted_prediction = []
        infer_data = test_data.copy()
        infer_data.drop(columns=['structure', 'ase_structure'], inplace=True)
        if model_params["model_type"] == "e3nn":
            infer_data.drop(columns=['datapoint'], inplace=True)
            
        for index, _ in infer_data.iterrows():
            sorted_prediction.append(prediction[index])

        if interpolation:
            infer_data["predicted_"+target_prop+"_diff"] = sorted_prediction
            infer_data["predicted_" + target_prop] = infer_data["predicted_"+target_prop+"_diff"] + infer_data[target_prop + '_interp']
        else:
            infer_data["predicted_"+target_prop] = sorted_prediction

        infer_data.to_json(directory + '/' + test_set_type + "_predictions.json")

        
def load_model(gpu_num, train_loader, model_params, directory, target_prop,per_site):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)

    if model_params["model_type"] == "Painn":
        model = torch.load(directory + "/best_model", map_location=device)
        normalizer = None
    else:
        with open(directory + "/hyperparameters.json") as json_file:
                assignments = json.load(json_file)
        model, normalizer = create_model(model_params["model_type"], train_loader,model_params["interpolation"],target_prop,hyperparameters=assignments,per_site=per_site)
        model.to(device)
        model.load_state_dict(torch.load(directory + "/best_model.torch", map_location=device)['state'])
    
    return model, normalizer


def evaluate_model_with_tracked_ids(model, normalizer, gpu_num, test_loader, model_params,return_ids = False):
    device_name = "cuda:" + str(gpu_num)
    device = torch.device(device_name)
    predictions = {}

    if model_params["model_type"] == "Painn":
        prop_names = model.output_keys
        loss_fn_painn = build_mae_loss(loss_coef = {target_prop: 1.0 for target_prop in prop_names})
        results, targets, _ = evaluate(model, 
                                          test_loader, 
                                          loss_fn_painn, 
                                          device=gpu_num)
        
        prop_name = prop_names[0]
        out = [float(entry) for entry in results[prop_name]]
        ids = list([int(i) for i in targets['crystal_id']])
        for i in range(len(ids)):
            predictions[ids[i]] = out[i]

        if return_ids:
            return predictions, ids
        return predictions

    else:
        model.eval()    
        all_crys_ids = []
        with torch.no_grad():
            for j, d in enumerate(test_loader):
                if model_params["model_type"] == "CGCNN":
                    input_struct = d[0]
                    target = d[1]
                    input_var = (Variable(input_struct[0].cuda(non_blocking=True)),
                                 Variable(input_struct[1].cuda(non_blocking=True)),
                                 input_struct[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True) for crys_idx in input_struct[3]])
                    output = model(*input_var).view(-1)
                    target = Variable(target.cuda(non_blocking=True))
                    crys_idx = d[2]
                elif model_params["model_type"] == "ALIGNN":
                    # Handle ALIGNN data - it should be a tuple from our custom collate function
                    if isinstance(d, tuple) and len(d) == 3:
                        # New format: (input_data, targets, crystal_ids)
                        input_data = d[0]  # (batched_graph, batched_line_graph, lattices_tensor)
                        target = d[1]      # labels_tensor
                        crys_idx = d[2]    # crystal indices
                        
                        # Extract graphs for model input - ALIGNN expects (graph, line_graph, lattice)
                        batched_graph, batched_line_graph, lattices_tensor = input_data
                        output = model((batched_graph, batched_line_graph, lattices_tensor))
                    else:
                        # Fallback: old format where d might be a different structure
                        # For ALIGNN, d should be: (batched_graph, batched_line_graph, lattices_tensor, labels_tensor)
                        batched_graph, batched_line_graph, lattices_tensor, labels_tensor = d
                        output = model((batched_graph, batched_line_graph, lattices_tensor))
                        # Generate batch indices as fallback
                        batch_size = labels_tensor.shape[0]
                        crys_idx = list(range(j * batch_size, (j + 1) * batch_size))
                else:
                    d.to(device)
                    output = model(d)
                    crys_idx = d.idx
                    
                try:
                    crys_idx = crys_idx.detach().cpu().numpy().reshape(-1)
                except:
                    crys_idx = np.array(crys_idx)

                predictions_iter = normalizer.denorm(output).detach().cpu().numpy().reshape(crys_idx.shape[0],-1)
                
                for i in range(len(crys_idx)):
                    predictions[crys_idx[i]] = predictions_iter[i]

        if return_ids:
            return predictions, list(predictions.keys())
        return predictions
