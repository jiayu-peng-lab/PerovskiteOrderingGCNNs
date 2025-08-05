import multiprocessing
import torch.multiprocessing as mp
from multiprocessing import freeze_support

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
# This is required when using CUDA with multiprocessing
mp.set_start_method('spawn', force=True)

from training.run_wandb_experiment import run_wandb_experiment
import wandb

def main():
    wandb.login()

    #run_sigopt_experiment
    run_wandb_experiment(
        struct_type="unrelaxed",
        model_type="ALIGNN",
        gpu_num=0,
        obs_budget=1,
        training_fraction=0.125,
        training_seed=0
    )

if __name__ == "__main__":
    freeze_support()
    main()
