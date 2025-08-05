from training.run_wandb_experiment import run_wandb_experiment
import wandb
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
