import json
import sys
import pandas as pd
import argparse
import pickle as pkl
import torch
import numpy as np
import random
import shutil
import os
import wandb
from processing.utils import filter_data_by_properties,select_structures
from processing.interpolation.Interpolation import *
from processing.dataloader.dataloader import get_dataloader
from processing.create_model.create_model import create_model
from training.hyperparameters.wandb_parameters import *
from training.model_training.trainer import *
from training.wandb_utils import build_wandb_name
from training.evaluate import *


def run_wandb_experiment(struct_type,model_type,gpu_num,experiment_id=None,parallel_band=1,obs_budget=50,training_fraction=1.0,data_name="data/",target_prop="dft_e_hull",interpolation=False,contrastive_weight=1.0,training_seed=0,nickname="",resume_sweep_id=None):
    """Run wandb hyperparameter optimization experiment"""
    
    if data_name == "data/":
        training_data = pd.read_json(data_name + 'training_set.json')
        training_data = training_data.sample(frac=training_fraction,replace=False,random_state=training_seed)
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

    # Create wandb sweep configuration
    sweep_config = create_wandb_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,obs_budget)
    wandb_name = build_wandb_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    
    print(f"Created wandb sweep configuration for '{wandb_name}'")
    print(f"Model type: {model_type}")
    print(f"Structure type: {struct_type}")
    print(f"Project name from config: {sweep_config.get('project', 'NOT_FOUND')}")

    # Initialize sweep with dynamic project name
    project_name = sweep_config['project']
    print(f"Using project name: {project_name}")
    
    # Remove project from sweep_config to avoid conflicts
    sweep_config_without_project = sweep_config.copy()
    del sweep_config_without_project['project']
    
    # Check if we're resuming an existing sweep
    if resume_sweep_id is not None:
        try:
            # Try to get the existing sweep
            api = wandb.Api()
            sweep = api.sweep(f"{project_name}/{resume_sweep_id}")
            
            # Count completed runs
            completed_runs = len([run for run in sweep.runs if run.state == "finished"])
            print(f"Found existing sweep with ID: {resume_sweep_id}")
            print(f"Completed runs: {completed_runs}")
            print(f"Original budget: {obs_budget}")
            
            # Calculate remaining budget
            remaining_budget = obs_budget - completed_runs
            if remaining_budget <= 0:
                print(f"All {obs_budget} runs have been completed. No more runs needed.")
                return
            
            print(f"Resuming sweep with {remaining_budget} remaining runs")
            sweep_id = resume_sweep_id
            
        except Exception as e:
            print(f"Error accessing sweep {resume_sweep_id}: {e}")
            print("Creating new sweep instead...")
            sweep_id = wandb.sweep(sweep_config_without_project, project=project_name)
            print(f"Created new wandb sweep with ID: {sweep_id}")
    else:
        # Create new sweep
        sweep_id = wandb.sweep(sweep_config_without_project, project=project_name)
        print(f"Created wandb sweep with ID: {sweep_id}")
        remaining_budget = obs_budget
    
    print(f"Project: {project_name}")

    # Define the training function for the sweep
    def train_function():
        # Initialize wandb run
        wandb.init()
        
        # Get hyperparameters from wandb
        hyperparameters = dict(wandb.config)
        
        # Convert hyperparameters to expected format
        hyperparameters = convert_hyperparameters(hyperparameters)
        
        # Train model
        val_loss = wandb_evaluate_model(data_name,hyperparameters,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,sweep_id,obs_budget,gpu_num,nickname)
        
        # Log final validation loss
        wandb.log({"val_mae": val_loss})
        
        # Save model files permanently (equivalent to sigopt storage)
        run_id = wandb.run.id
        model_save_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(sweep_id) + '/observ_' + str(run_id)
        model_tmp_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(run_id) + '/' + nickname + '_tmp' + str(gpu_num)
        
        # Ensure the temporary directory exists
        if not os.path.exists(model_tmp_dir):
            os.makedirs(model_tmp_dir)
        
        # Save hyperparameters and training results
        training_results = {"validation_loss": val_loss}
        with open(model_tmp_dir + '/hyperparameters.json', 'w') as file:
            json.dump(hyperparameters, file)
        with open(model_tmp_dir + '/training_results.json', 'w') as file:
            json.dump(training_results, file)
        
        # Create permanent save directory
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        # Copy contents of tmp file to permanent location
        possible_file_names = ["best_model", "best_model.pth.tar", "best_model.torch",
                               "final_model.torch","final_model","final_model.pth.tar",
                               "log_human_read.csv","checkpoints/checkpoint-100.pth.tar",
                               'hyperparameters.json','training_results.json']
        for file_name in possible_file_names:
            if os.path.isfile(model_tmp_dir + "/" + file_name):
                if file_name == "checkpoints/checkpoint-100.pth.tar":
                    shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + "checkpoint-100.pth.tar")
                else:
                    shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + file_name)
        
        # Clean up tmp directory
        shutil.rmtree(model_tmp_dir)
        torch.cuda.empty_cache()
    
    # Run the sweep with remaining budget
    wandb.agent(sweep_id, train_function, count=remaining_budget)
    
    print(f"Completed wandb sweep with {remaining_budget} observations")


def wandb_evaluate_model(data_name,hyperparameters,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,experiment_id,observation_count,gpu_num,nickname):
    """Evaluate model for wandb experiment"""

    # Wandb equivalent (active)
    device = "cuda:" + str(gpu_num)
    
    train_data = processed_data[0]
    validation_data = processed_data[1]

    per_site = False
    if "per_site" in target_prop:
        per_site = True

    # Ensure hyperparameters is a dict and convert if needed
    if hyperparameters is None:
        hyperparameters = {}
    elif not isinstance(hyperparameters, dict):
        hyperparameters = dict(hyperparameters)
    
    # Convert hyperparameters to expected format
    hyperparameters = convert_hyperparameters(hyperparameters)

    train_loader = get_dataloader(train_data,target_prop,model_type,hyperparameters["batch_size"],interpolation,per_site=per_site,device=device)
    train_eval_loader = None

    if "e3nn" in model_type and "pretrain" not in data_name and "per_site" not in target_prop:
        train_eval_loader = get_dataloader(train_data,target_prop,"e3nn_contrastive",1,interpolation,per_site=per_site,device=device)
        val_loader = get_dataloader(validation_data,target_prop,"e3nn_contrastive",1,interpolation,per_site=per_site,device=device)
    else:
        val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation,per_site=per_site,device=device)
    
    # Pass hyperparameters as positional argument
    model, normalizer = create_model(model_type, train_loader, interpolation, target_prop, hyperparameters, per_site=per_site)
    
    wandb_name = build_wandb_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    
    # Get run ID safely
    run_id = "unknown"
    if wandb.run is not None:
        run_id = wandb.run.id
    
    model_tmp_dir = './saved_models/'+ model_type + '/' + wandb_name + '/wandb-' + str(run_id) + '/' + nickname + '_tmp' + str(gpu_num)
    if os.path.exists(model_tmp_dir):
        shutil.rmtree(model_tmp_dir)
    os.makedirs(model_tmp_dir) 

    best_model,loss_fn = trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,model_tmp_dir,gpu_num,train_eval_loader=train_eval_loader,contrastive_weight=contrastive_weight)
    
    is_contrastive = False
    if "contrastive" in model_type:
        is_contrastive = True
    _, _, best_loss = evaluate_model(best_model, normalizer, model_type, val_loader, loss_fn, gpu_num,is_contrastive=is_contrastive, contrastive_weight=contrastive_weight)

    if model_type == "Painn":
        return best_loss
    else:
        return best_loss[0]


def create_wandb_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,obs_budget):
    """Create wandb sweep configuration"""

    # Wandb equivalent (active)
    wandb_name = build_wandb_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
    
    if model_type == "Painn":
        sweep_config = get_painn_hyperparameter_range()
    elif model_type == "CGCNN":
        sweep_config = get_cgcnn_hyperparameter_range()
    elif model_type == "ALIGNN":
        sweep_config = get_alignn_hyperparameter_range()
    else:
        sweep_config = get_e3nn_hyperparameter_range()
    
    # Create project name based on model type, structure type, and training fraction
    # 6 base project names: CGCNN-unrelaxed, CGCNN-relaxed, e3nn-unrelaxed, e3nn-relaxed, Painn-unrelaxed, Painn-relaxed
    if struct_type in ["unrelaxed", "relaxed"]:
        project_name = f"perovskite-ordering-{model_type.lower()}-{struct_type}"
    else:
        # For other structure types, use a generic name
        project_name = f"perovskite-ordering-{model_type.lower()}-{struct_type}"

    # Append training fraction to project name for disambiguation when not full data
    if training_fraction is not None and float(training_fraction) != 1.0:
        project_name = f"{project_name}-frac-{training_fraction}"
    
    # Add project and program info
    sweep_config.update({
        'project': project_name,
        'name': wandb_name
    })
    
    return sweep_config


def convert_hyperparameters(hyperparameters):
    """Convert wandb hyperparameters to the format expected by the model"""
    # This function should convert the hyperparameters from wandb format
    # to the format expected by the create_model function
    # You may need to adjust this based on your specific hyperparameter structure
    return hyperparameters


def list_existing_sweeps(project_name=None, model_type=None, struct_type=None):
    """List existing sweeps that can be resumed"""
    try:
        api = wandb.Api()
        
        # If no project specified, try to construct it from model and struct type
        if project_name is None and model_type is not None and struct_type is not None:
            if struct_type in ["unrelaxed", "relaxed"]:
                project_name = f"perovskite-ordering-{model_type.lower()}-{struct_type}"
            else:
                project_name = f"perovskite-ordering-{model_type.lower()}-{struct_type}"
        
        if project_name is None:
            print("Please specify project_name or provide model_type and struct_type")
            return
        
        print(f"Searching for sweeps in project: {project_name}")
        
        # Get all sweeps in the project
        sweeps = api.sweeps(project_name)
        
        if not sweeps:
            print(f"No sweeps found in project: {project_name}")
            return
        
        print(f"\nFound {len(sweeps)} sweeps:")
        print("-" * 80)
        
        for sweep in sweeps:
            # Count runs by state
            finished_runs = len([run for run in sweep.runs if run.state == "finished"])
            running_runs = len([run for run in sweep.runs if run.state == "running"])
            failed_runs = len([run for run in sweep.runs if run.state == "failed"])
            total_runs = len(sweep.runs)
            
            print(f"Sweep ID: {sweep.id}")
            print(f"Name: {sweep.name}")
            print(f"State: {sweep.state}")
            print(f"Runs: {finished_runs} finished, {running_runs} running, {failed_runs} failed (total: {total_runs})")
            print(f"Created: {sweep.created_at}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error accessing wandb API: {e}")
        print("Make sure you're logged in to wandb: wandb login")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for perovskite ordering GCNNs using wandb')
    parser.add_argument('--data_name', default = "data/", type=str, metavar='name',
                        help="the source of the data")
    parser.add_argument('--prop', default = "dft_e_hull", type=str, metavar='name',
                        help="the property to predict (default: dft_e_hull; other options: Op_band_center)")
    parser.add_argument('--struct_type', default = 'unrelaxed', type=str, metavar='struct_type',
                        help="using which structure representation (default: unrelaxed; other options: relaxed, M3Gnet_relaxed)")
    parser.add_argument('--interpolation', default = 'no', type=str, metavar='yes/no',
                        help="using interpolation (default: no; other options: yes)")
    parser.add_argument('--model', default = "CGCNN", type=str, metavar='model',
                        help="the neural network to use (default: CGCNN; other options: Painn, e3nn, e3nn_contrastive)")
    parser.add_argument('--contrastive_weight', default = 1.0, type=float, metavar='loss_parameters',
                        help="the weighting applied to the contrastive loss term (default: 1.0)")
    parser.add_argument('--training_fraction', default = 1.0, type=float, metavar='training_set',
                        help="fraction of the total training set used (default 1.0)")
    parser.add_argument('--training_seed', default = 0, type=int, metavar='training_set',
                        help="the random seed for selecting fraction of training set (default 0)")
    parser.add_argument('--gpu', default = 0, type=int, metavar='device',
                        help="the gpu to use (default: 0)")
    parser.add_argument('--nickname', default = "", type=str, metavar='device',
                        help="nickname for temporary folder")
    parser.add_argument('--budget', default = 50, type=int, metavar='wandb_props',
                        help="budget of wandb sweep (default: 50)")
    parser.add_argument('--resume_sweep_id', default=None, type=str, metavar='sweep_id',
                        help="ID of the sweep to resume (optional)")
    parser.add_argument('--list_sweeps', action='store_true',
                        help="list existing sweeps that can be resumed")
    args = parser.parse_args()

    # Handle list sweeps option
    if args.list_sweeps:
        list_existing_sweeps(model_type=args.model, struct_type=args.struct_type)
        sys.exit(0)

    data_name = args.data_name
    target_prop = args.prop
    model_type = args.model
    gpu_num = args.gpu
    nickname = args.nickname
    struct_type = args.struct_type
    contrastive_weight = args.contrastive_weight
    training_fraction = args.training_fraction
    training_seed = args.training_seed
    obs_budget = args.budget
    resume_sweep_id = args.resume_sweep_id
    
    if struct_type not in ["unrelaxed", "relaxed", "spud", "M3Gnet_relaxed"]:
        raise ValueError('struct type is not available')
    
    if args.interpolation == 'yes':
        interpolation = True
    elif args.interpolation == 'no':
        interpolation = False
    else:
        raise ValueError('interpolation needs to be yes or no')    
    
    run_wandb_experiment(struct_type,model_type,gpu_num,None,1,obs_budget,training_fraction,data_name,target_prop,interpolation,contrastive_weight,training_seed,nickname,resume_sweep_id) 