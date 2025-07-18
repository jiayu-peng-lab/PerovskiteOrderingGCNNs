# # Original SigOpt imports
# import sigopt
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
# from processing.utils import filter_data_by_properties,select_structures
# from processing.interpolation.Interpolation import *
# from processing.dataloader.dataloader import get_dataloader
# from processing.create_model.create_model import create_model
# from training.hyperparameters.sigopt_parameters import *
# from training.model_training.trainer import *
# from training.sigopt_utils import build_sigopt_name
# from training.evaluate import *


# def run_sigopt_experiment(struct_type,model_type,gpu_num,experiment_id=None,parallel_band=1,obs_budget=50,training_fraction=1.0,data_name="data/",target_prop="dft_e_hull",interpolation=False,contrastive_weight=1.0,training_seed=0,nickname=""):

#     if experiment_id == None:
#         sigopt_settings={"parallel_band": parallel_band,"obs_budget": obs_budget}

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

#     conn = sigopt.Connection(driver="lite")
#     sigopt_name = build_sigopt_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)

#     if experiment_id == None:
#         experiment = create_sigopt_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,sigopt_settings,conn)
#         print("Created a new SigOpt experiment '" + sigopt_name + "' with ID: " + str(experiment.id))
#     else:
#         experiment = conn.experiments(experiment_id).fetch()
#         print("Continuing a prior SigOpt experiment '" + sigopt_name + "' with ID: " + str(experiment.id))

#     while experiment.progress.observation_count < experiment.observation_budget:
#         print('\n========================\nSigopt experiment count #', experiment.progress.observation_count)
#         suggestion = conn.experiments(experiment.id).suggestions().create()
#         hyperparameters = suggestion.assignments

#         value = sigopt_evaluate_model(data_name,suggestion.assignments,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,experiment.id,experiment.progress.observation_count,gpu_num,nickname)    
#         training_results = {"validation_loss": value}

#         conn.experiments(experiment.id).observations().create(
#             suggestion=suggestion.id,
#             values=[{"name": "val_mae", "value": value}],
#         )

#         experiment = conn.experiments(experiment.id).fetch()
#         observation_id = experiment.progress.observation_count - 1

#         model_save_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(experiment.id) + '/observ_' + str(observation_id)
#         model_tmp_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(experiment_id) + '/' + nickname + '_tmp' + str(gpu_num)

#         with open(model_tmp_dir + '/hyperparameters.json', 'w') as file:
#             json.dump(hyperparameters, file)
#         with open(model_tmp_dir + '/training_results.json', 'w') as file:
#             json.dump(training_results, file)

#         if not os.path.exists(model_save_dir):
#             os.makedirs(model_save_dir)

#         ### Copy contents of tmp file
#         possible_file_names = ["best_model", "best_model.pth.tar", "best_model.torch",
#                                "final_model.torch","final_model","final_model.pth.tar",
#                                "log_human_read.csv","checkpoints/checkpoint-100.pth.tar",
#                                'hyperparameters.json','training_results.json']
#         for file_name in possible_file_names:
#             if os.path.isfile(model_tmp_dir + "/" + file_name):
#                 if file_name == "checkpoints/checkpoint-100.pth.tar":
#                     shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + "checkpoint-100.pth.tar")
#                 else:
#                     shutil.move(model_tmp_dir + "/" + file_name, model_save_dir + "/" + file_name)
        
#         ### Empty tmp file
#         shutil.rmtree(model_tmp_dir)
#         torch.cuda.empty_cache()


# def sigopt_evaluate_model(data_name,hyperparameters,processed_data,target_prop,interpolation,struct_type,model_type,contrastive_weight,training_fraction,training_seed,experiment_id,observation_count,gpu_num,nickname):
#     device = "cuda:" + str(gpu_num)
#     train_data = processed_data[0]
#     validation_data = processed_data[1]
#     per_site = False
#     if "per_site" in target_prop:
#         per_site = True
#     train_loader = get_dataloader(train_data,target_prop,model_type,hyperparameters["batch_size"],interpolation,per_site=per_site)
#     train_eval_loader = None
#     if "e3nn" in model_type and "pretrain" not in data_name and "per_site" not in target_prop:
#         train_eval_loader = get_dataloader(train_data,target_prop,"e3nn_contrastive",1,interpolation,per_site=per_site)
#         val_loader = get_dataloader(validation_data,target_prop,"e3nn_contrastive",1,interpolation,per_site=per_site)
#     else:
#         val_loader = get_dataloader(validation_data,target_prop,model_type,1,interpolation,per_site=per_site)
#     model, normalizer = create_model(model_type,train_loader,interpolation,target_prop,hyperparameters=hyperparameters,per_site=per_site)
#     sigopt_name = build_sigopt_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
#     model_tmp_dir = './saved_models/'+ model_type + '/' + sigopt_name + '/' + str(experiment_id) + '/' + nickname + '_tmp' + str(gpu_num)
#     if os.path.exists(model_tmp_dir):
#         shutil.rmtree(model_tmp_dir)
#     os.makedirs(model_tmp_dir) 
#     best_model,loss_fn = trainer(model,normalizer,model_type,train_loader,val_loader,hyperparameters,model_tmp_dir,gpu_num,train_eval_loader=train_eval_loader,contrastive_weight=contrastive_weight)
#     is_contrastive = False
#     if "contrastive" in model_type:
#         is_contrastive = True
#     _, _, best_loss = evaluate_model(best_model, normalizer, model_type, val_loader, loss_fn, gpu_num,is_contrastive=is_contrastive, contrastive_weight=contrastive_weight)
#     if model_type == "Painn":
#         return best_loss
#     else:
#         return best_loss[0]


# def create_sigopt_experiment(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed,sigopt_settings,conn):
#     sigopt_name = build_sigopt_name(data_name,target_prop,struct_type,interpolation,model_type,contrastive_weight,training_fraction,training_seed)
#     if model_type == "Painn":
#         curr_parameters = get_painn_hyperparameter_range()
#     elif model_type == "CGCNN":
#         curr_parameters = get_cgcnn_hyperparameter_range()
#     else:
#         curr_parameters = get_e3nn_hyperparameter_range()
#     experiment = conn.experiments().create(
#         name=sigopt_name, 
#         parameters = curr_parameters,
#         metrics=[dict(name="val_mae", objective="minimize", strategy="optimize")],
#         observation_budget=sigopt_settings["obs_budget"], 
#         parallel_bandwidth=sigopt_settings["parallel_band"],
#     )
#     return experiment


# ####### DEFINE MAIN

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Hyperparameter optimization for perovksite ordering GCNNs')
#     parser.add_argument('--data_name', default = "data/", type=str, metavar='name',
#                         help="the source of the data")
#     parser.add_argument('--prop', default = "dft_e_hull", type=str, metavar='name',
#                         help="the property to predict (default: dft_e_hull; other options: Op_band_center)")
#     parser.add_argument('--struct_type', default = 'unrelaxed', type=str, metavar='struct_type',
#                         help="using which structure representation (default: unrelaxed; other options: relaxed, M3Gnet_relaxed)")
#     parser.add_argument('--interpolation', default = 'no', type=str, metavar='yes/no',
#                         help="using interpolation (default: no; other options: yes)")
#     parser.add_argument('--model', default = "CGCNN", type=str, metavar='model',
#                         help="the neural network to use (default: CGCNN; other options: Painn, e3nn, e3nn_contrastive)")
#     parser.add_argument('--contrastive_weight', default = 1.0, type=float, metavar='loss_parameters',
#                         help="the weighting applied to the contrastive loss term (default: 1.0)")
#     parser.add_argument('--training_fraction', default = 1.0, type=float, metavar='training_set',
#                         help="fraction of the total training set used (default 1.0)")
#     parser.add_argument('--training_seed', default = 0, type=int, metavar='training_set',
#                         help="the random seed for selecting fraction of training set (default 0)")
#     parser.add_argument('--gpu', default = 0, type=int, metavar='device',
#                         help="the gpu to use (default: 0)")
#     parser.add_argument('--nickname', default = "", type=str, metavar='device',
#                         help="nickname for temporary folder")
#     parser.add_argument('--id', default = -1, type=int, metavar='sigopt_props',
#                         help="id for sigopt experiment (default: -1)")
#     parser.add_argument('--parallel', default = 1, type=int, metavar='sigopt_props',
#                         help="bandwidth of sigopt (default: 1)")
#     parser.add_argument('--budget', default = 50, type=int, metavar='sigopt_props',
#                         help="budget of sigopt (default: 50)")
#     args = parser.parse_args()

#     data_name = args.data_name
#     target_prop = args.prop
#     model_type = args.model
#     gpu_num = args.gpu
#     nickname = args.nickname
#     struct_type = args.struct_type
#     contrastive_weight = args.contrastive_weight
#     training_fraction = args.training_fraction
#     training_seed = args.training_seed
    
#     if struct_type not in ["unrelaxed", "relaxed", "spud", "M3Gnet_relaxed"]:
#         raise ValueError('struct type is not available')
    
#     if args.interpolation == 'yes':
#         interpolation = True
#     elif args.interpolation == 'no':
#         interpolation = False
#     else:
#         raise ValueError('interpolation needs to be yes or no')    
        
#     if args.id == -1:
#         experiment_id = None
#         parallel_band = args.parallel
#         obs_budget = args.budget
#     else:
#         experiment_id = args.id
#         parallel_band = None
#         obs_budget = None
    
#     run_sigopt_experiment(struct_type,model_type,gpu_num,experiment_id,parallel_band,obs_budget,training_fraction,data_name,target_prop,interpolation,contrastive_weight,training_seed,nickname)

