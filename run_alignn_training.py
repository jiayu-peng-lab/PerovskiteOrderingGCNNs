#!/usr/bin/env python3
"""
Script to run ALIGNN training with fixed DGL CUDA support
"""

import os
import sys
import multiprocessing
import torch.multiprocessing as mp
from multiprocessing import freeze_support

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
# This is required when using CUDA with multiprocessing
mp.set_start_method('spawn', force=True)

from training.run_wandb_experiment_gpu import run_wandb_experiment_gpu

# Set environment variables for DGL GPU usage
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    """Run ALIGNN training with GPU support"""
    print("Starting ALIGNN training with GPU support...")
    print("DGL CUDA issue has been fixed!")
    
    # Run the experiment with ALIGNN model
    run_wandb_experiment_gpu(
        struct_type="unrelaxed",
        model_type="ALIGNN",
        gpu_num=0,
        obs_budget=1,  # Start with 1 observation for testing
        training_fraction=0.125,
        training_seed=0
    )

if __name__ == "__main__":
    freeze_support()
    main() 