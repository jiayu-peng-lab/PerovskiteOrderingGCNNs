# import json
# import sys
# import pandas as pd
# import argparse
# import pickle as pkl
# import torch
# import numpy as np
# import random
# import shutil
# import os
# import wandb
# from processing.utils import filter_data_by_properties,select_structures
# from processing.interpolation.Interpolation import *
# from processing.dataloader.dataloader import get_dataloader
# from processing.dataloader.dataloader_gpu_optimized import get_dataloader_gpu_optimized
# from processing.create_model.create_model import create_model
# from training.hyperparameters.wandb_parameters import *
# from training.model_training.trainer_gpu_hybrid import trainer_gpu_hybrid
# from training.wandb_utils import build_wandb_name
# from training.evaluate import *


# def run_wandb_experiment_gpu(struct_type,model_type,gpu_num,experiment_id=None,parallel_band=1,obs_budget=50,training_fraction=1.0,data_name="data/",target_prop="dft_e_hull",interpolation=False,contrastive_weight=1.0,training_seed=0,nickname="",resume_sweep_id=None):
#     """Run wandb hyperparameter optimization experiment with GPU optimization for DGL"""
    
#     if data_name == "data/":
#         training_data = pd.read_json(data_name + 'training_set.json')
#         training_data = training_data.sample(frac=training_fraction,replace=False,random_state=training_seed)
#         validation_data = pd.read_json(data_name + 'validation_set.json')
#         edge_data = pd.read_json(data_name + 'edge_dataset.json')
#         if not interpolation:
#             training_data = pd.concat((training_data,edge_data))
#     else:
#         print("Specified Data Directory Does Not Exist!")

#     torch.manual_seed(0)
#     random.seed(0)
#     np.random.seed(0)
#     print("Loaded data")

#     data = [training_data, validation_data]
#     processed_data = []

#     for dataset in data:
#         dataset = filter_data_by_properties(dataset,target_prop)
#         dataset = select_structures(dataset,struct_type)
#         if interpolation:
#             dataset = apply_interpolation(dataset,target_prop)
#         processed_data.append(dataset)

#     print("Completed data processing")

#     # Create wandb sweep configuration
#     sweep_config = create_wandb_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,obs_budget)
#     wandb_name = build_wandb_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    
#     print(f"Created wandb sweep configuration for '{wandb_name}'")
#     print(f"Model type: {model_type}")
#     print(f"Structure type: {struct_type}")
#     print(f"Project name from config: {sweep_config.get('project', 'NOT_FOUND')}")

#     # Initialize sweep with dynamic project name
#     project_name = sweep_config['project']
#     print(f"Using project name: {project_name}")
    
#     # Remove project from sweep_config to avoid conflicts
#     sweep_config_without_project = sweep_config.copy()
#     del sweep_config_without_project['project']
    
#     # Check if we're resuming an existing sweep
#     if resume_sweep_id is not None:
#         try:
#             # Try to get the existing sweep
#             api = wandb.Api()
#             sweep = api.sweep(f"{project_name}/{resume_sweep_id}")
            
#             # Count completed runs
#             completed_runs = len([run for run in sweep.runs if run.state == "finished"])
#             print(f"Found existing sweep with ID: {resume_sweep_id}")
#             print(f"Completed runs: {completed_runs}")
#             print(f"Original budget: {obs_budget}")
            
#             # Calculate remaining budget
#             remaining_budget = obs_budget - completed_runs
#             if remaining_budget <= 0:
#                 print(f"All {obs_budget} runs have been completed. No more runs needed.")
#                 return
            
#             print(f"Resuming sweep with {remaining_budget} remaining runs")
#             sweep_id = resume_sweep_id
            
#         except Exception as e:
#             print(f"Error accessing sweep {resume_sweep_id}: {e}")
#             print("Creating new sweep instead...")
#             sweep_id = wandb.sweep(sweep_config_without_project, project=project_name)
#             print(f"Created new wandb sweep with ID: {sweep_id}")
#     else:
#         # Create new sweep
#         sweep_id = wandb.sweep(sweep_config_without_project, project=project_name)
#         print(f"Created new wandb sweep with ID: {sweep_id}")
#         remaining_budget = obs_budget
    
#     print(f"Project: {project_name}")

#     # Define the training function for the sweep
#     def train_function():
#         # Initialize wandb run
#         wandb.init()
        
#         # Get hyperparameters from wandb
#         hyperparameters = dict(wandb.config)
        
#         # Convert hyperparameters to expected format
#         hyperparameters = convert_hyperparameters(hyperparameters)
        
#         # Train model with GPU optimization
#         val_loss = wandb_evaluate_model_gpu(data_name,hyperparameters,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,sweep_id,obs_budget,gpu_num,nickname)
        
#         # Log final validation loss
#         wandb.log({"val_mae": val_loss})
        
#         # Save model files permanently (equivalent to sigopt storage)
#         run_id = wandb.run.id
#         model_save_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(sweep_id) + '/observ_' + str(run_id)
#         model_tmp_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(run_id) + '/' + nickname + '_tmp' + str(gpu_num)
        
#         # Ensure the temporary directory exists
#         if not os.path.exists(model_tmp_dir):
#             os.makedirs(model_tmp_dir)
        
#         # Copy best model to permanent storage
#         if os.path.exists(model_tmp_dir + '/best_model.torch'):
#             if not os.path.exists(model_save_dir):
#                 os.makedirs(model_save_dir)
#             shutil.copy2(model_tmp_dir + '/best_model.torch', model_save_dir + '/best_model.torch')
#             print(f"Saved best model to {model_save_dir}")
        
#         # Clean up temporary directory
#         shutil.rmtree(model_tmp_dir)
#         print(f"Cleaned up temporary directory: {model_tmp_dir}")

#     # Start the sweep
#     wandb.agent(sweep_id, train_function, count=remaining_budget)
#     print(f"Completed sweep with ID: {sweep_id}")


# def wandb_evaluate_model_gpu(data_name,hyperparameters,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,experiment_id,observation_count,gpu_num,nickname):
#     """Evaluate model for wandb experiment with GPU optimization for DGL"""
    
#     # Set device
#     device = f"cuda:{gpu_num}" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")
    
#     train_data = processed_data[0]
#     validation_data = processed_data[1]

#     per_site = False
#     if "per_site" in target_prop:
#         per_site = True

#     # Ensure hyperparameters is a dict and convert if needed
#     if hyperparameters is None:
#         hyperparameters = {}
#     elif not isinstance(hyperparameters, dict):
#         hyperparameters = dict(hyperparameters)
    
#     # Convert hyperparameters to expected format
#     hyperparameters = convert_hyperparameters(hyperparameters)

#     # Use GPU-optimized dataloader for ALIGNN, regular dataloader for others
#     if model_type == "ALIGNN":
#         print("Using GPU-optimized dataloader for ALIGNN")
#         train_loader = get_dataloader_gpu_optimized(train_data, target_prop, model_type, hyperparameters["batch_size"], interpolation, per_site=per_site, device=device)
#         val_loader = get_dataloader_gpu_optimized(validation_data, target_prop, model_type, 1, interpolation, per_site=per_site, device=device)
#         train_eval_loader = None
#     else:
#         # Use regular dataloader for other models
#         train_loader = get_dataloader(train_data, target_prop, model_type, hyperparameters["batch_size"], interpolation, per_site=per_site)
#         train_eval_loader = None

#         if "e3nn" in model_type and "pretrain" not in data_name and "per_site" not in target_prop:
#             train_eval_loader = get_dataloader(train_data, target_prop, "e3nn_contrastive", 1, interpolation, per_site=per_site)
#             val_loader = get_dataloader(validation_data, target_prop, "e3nn_contrastive", 1, interpolation, per_site=per_site)
#         else:
#             val_loader = get_dataloader(validation_data, target_prop, model_type, 1, interpolation, per_site=per_site)
    
#     # Pass hyperparameters as positional argument
#     model, normalizer = create_model(model_type, train_loader, interpolation, target_prop, hyperparameters, per_site=per_site)
    
#     wandb_name = build_wandb_name(data_name, target_prop, struct_type, interpolation, model_type, contrastive_weight, training_fraction, training_seed)
    
#     # Get run ID safely
#     run_id = "unknown"
#     if wandb.run is not None:
#         run_id = wandb.run.id
    
#     model_tmp_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(run_id) + '/' + nickname + '_tmp' + str(gpu_num)
#     if os.path.exists(model_tmp_dir):
#         shutil.rmtree(model_tmp_dir)
#     os.makedirs(model_tmp_dir) 

#     best_model, loss_fn = trainer_gpu_hybrid(model, normalizer, model_type, train_loader, val_loader, hyperparameters, model_tmp_dir, gpu_num, train_eval_loader=train_eval_loader, contrastive_weight=contrastive_weight)
    
#     is_contrastive = False
#     if "contrastive" in model_type:
#         is_contrastive = True
#     _, _, best_loss = evaluate_model(best_model, normalizer, model_type, val_loader, loss_fn, gpu_num, is_contrastive=is_contrastive, contrastive_weight=contrastive_weight)

#     if model_type == "Painn":
#         return best_loss
#     else:
#         return best_loss[0]


# def create_wandb_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,obs_budget):
#     """Create wandb sweep configuration"""

#     # Wandb equivalent (active)
#     wandb_name = build_wandb_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    
#     if model_type == "Painn":
#         sweep_config = get_painn_hyperparameter_range()
#     elif model_type == "CGCNN":
#         sweep_config = get_cgcnn_hyperparameter_range()
#     elif model_type == "ALIGNN":
#         sweep_config = get_alignn_hyperparameter_range()
#     else:
#         sweep_config = get_e3nn_hyperparameter_range()
    
#     # Create project name based on model type and structure type
#     # 6 different project names: CGCNN-unrelaxed, CGCNN-relaxed, e3nn-unrelaxed, e3nn-relaxed, Painn-unrelaxed, Painn-relaxed
#     if struct_type in ["unrelaxed", "relaxed"]:
#         project_name = f"perovskite-ordering-{model_type.lower()}-{struct_type}"
#     else:
#         # For other structure types, use a generic name
#         project_name = f"perovskite-ordering-{model_type.lower()}-{struct_type}"
    
#     # Add project and program info
#     sweep_config.update({
#         'project': project_name,
#         'name': wandb_name
#     })
    
#     return sweep_config


# def convert_hyperparameters(hyperparameters):
#     """Convert wandb hyperparameters to the format expected by the model"""
#     # This function should convert the hyperparameters from wandb format
#     # to the format expected by the create_model function
#     # You may need to adjust this based on your specific hyperparameter structure
#     return hyperparameters 