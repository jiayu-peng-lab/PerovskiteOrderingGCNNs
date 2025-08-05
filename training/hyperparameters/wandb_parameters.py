# Wandb equivalents (active)
def get_cgcnn_hyperparameter_range():
    """Get wandb sweep configuration for CGCNN model"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization (equivalent to SigOpt's default)
        'metric': {
            'name': 'val_mae',
            'goal': 'minimize'
        },
        'parameters': {
            'MaxEpochs': {
                'value': 100
            },
            'batch_size': {
                'min': 4,
                'max': 16,
                'distribution': 'int_uniform'
            },
            'log_lr': {
                'min': -5,
                'max': -2,
                'distribution': 'int_uniform'
            },
            'reduceLR_patience': {
                'min': 10,
                'max': 30,
                'distribution': 'int_uniform'
            },
            'atom_fea_len': {
                'min': 32,
                'max': 256,
                'distribution': 'int_uniform'
            },
            'n_conv': {
                'min': 2,
                'max': 5,
                'distribution': 'int_uniform'
            },
            'h_fea_len': {
                'min': 32,
                'max': 256,
                'distribution': 'int_uniform'
            },
            'n_h': {
                'min': 1,
                'max': 4,
                'distribution': 'int_uniform'
            }
        }
    }
    return sweep_config


def get_painn_hyperparameter_range():
    """Get wandb sweep configuration for Painn model"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_mae',
            'goal': 'minimize'
        },
        'parameters': {
            'MaxEpochs': {
                'value': 100
            },
            'batch_size': {
                'min': 4,
                'max': 16,
                'distribution': 'int_uniform'
            },
            'log_lr': {
                'min': -5,
                'max': -3,
                'distribution': 'int_uniform'
            },
            'reduceLR_patience': {
                'min': 10,
                'max': 30,
                'distribution': 'int_uniform'
            },
            'log2_feat_dim': {
                'min': 5,
                'max': 9,
                'distribution': 'int_uniform'
            },
            'activation': {
                'values': ['swish', 'learnable_swish', 'ReLU', 'LeakyReLU']
            },
            'num_conv': {
                'min': 1,
                'max': 6,
                'distribution': 'int_uniform'
            }
        }
    }
    return sweep_config


def get_e3nn_hyperparameter_range():
    """Get wandb sweep configuration for e3nn model"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_mae',
            'goal': 'minimize'
        },
        'parameters': {
            'MaxEpochs': {
                'value': 100
            },
            'batch_size': {
                'min': 4,
                'max': 12,
                'distribution': 'int_uniform'
            },
            'log_lr': {
                'min': -5,
                'max': -2,
                'distribution': 'int_uniform'
            },
            'reduceLR_patience': {
                'min': 10,
                'max': 30,
                'distribution': 'int_uniform'
            },
            'len_embedding_feature_vector': {
                'values': [32, 64, 128]
            },
            'num_hidden_feature': {
                'values': [32, 64, 128]
            },
            'num_hidden_layer': {
                'min': 0,
                'max': 2,
                'distribution': 'int_uniform'
            },
            'multiplicity_irreps': {
                'values': [16, 32, 64]
            },
            'num_conv': {
                'min': 1,
                'max': 4,
                'distribution': 'int_uniform'
            },
            'num_radical_basis': {
                'values': [5, 10, 20]
            },
            'num_radial_neurons': {
                'values': [32, 64, 128]
            }
        }
    }
    return sweep_config


def get_alignn_hyperparameter_range():
    """Get wandb sweep configuration for ALIGNN model"""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_mae',
            'goal': 'minimize'
        },
        'parameters': {
            'MaxEpochs': {
                'value': 100
            },
            'batch_size': {
                'min': 4,
                'max': 12,
                'distribution': 'int_uniform'
            },
            'log_lr': {
                'min': -5,
                'max': -2,
                'distribution': 'int_uniform'
            },
            'reduceLR_patience': {
                'min': 10,
                'max': 30,
                'distribution': 'int_uniform'
            },
            'embedding_features': {
                'values': [32, 64, 128]
            },
            'hidden_features': {
                'values': [128, 256, 512]
            },
            'alignn_layers': {
                'min': 2,
                'max': 6,
                'distribution': 'int_uniform'
            },
            'gcn_layers': {
                'min': 2,
                'max': 6,
                'distribution': 'int_uniform'
            }
        }
    }
    return sweep_config


def convert_hyperparameters(hyperparameters):
    """Convert wandb hyperparameters to the format expected by the training code"""
    # Convert log_lr to actual learning rate but keep both
    if 'log_lr' in hyperparameters:
        hyperparameters['lr'] = 10 ** hyperparameters['log_lr']
        # Keep log_lr as well since trainer expects it
    
    # Convert log2_feat_dim to feat_dim for Painn
    if 'log2_feat_dim' in hyperparameters:
        hyperparameters['feat_dim'] = 2 ** hyperparameters['log2_feat_dim']
        # Keep log2_feat_dim as well since trainer expects it
    
    return hyperparameters 