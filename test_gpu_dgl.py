from training.run_wandb_experiment_gpu import run_wandb_experiment_gpu
import wandb
import torch
import dgl
import os

# Set environment variables for DGL GPU usage
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Check GPU availability
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"DGL version: {dgl.__version__}")

if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

wandb.login()

# Run experiment with GPU-optimized settings
run_wandb_experiment_gpu(
    struct_type="unrelaxed",
    model_type="ALIGNN",
    gpu_num=0,
    obs_budget=1,
    training_fraction=0.125,
    training_seed=0
) 