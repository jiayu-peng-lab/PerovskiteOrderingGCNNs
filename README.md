# PerovskiteOrderingGCNNs

[![arXiv](https://img.shields.io/badge/arXiv-2409.13851-red.svg)](https://arxiv.org/abs/2409.13851)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281/zenodo.13820311-blue.svg)](https://doi.org/10.5281/zenodo.13820311)
[![MDF](https://img.shields.io/badge/Materials_Data_Facility-10.18126/ncqt--rh18-purple.svg)](https://doi.org/10.18126/ncqt-rh18)
[![MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/license/mit)

Repo for our paper **"Learning Ordering in Crystalline Materials with Symmetry-Aware Graph Neural Networks"** ([preprint on arXiv](https://arxiv.org/abs/2409.13851)).

## 📥 Download

To start, clone this repo and all its submodules to your local directory or a workstation:
```
git clone --recurse-submodules git@github.com:jiayu-peng-lab/PerovskiteOrderingGCNNs.git
```
or
```
git clone git@github.com:jiayu-peng-lab/PerovskiteOrderingGCNNs.git
cd PerovskiteOrderingGCNNs
git submodule update --init
```

Our codes are built upon previous implementations of [CGCNN](https://github.com/-mit/PerovskiteOrderingGCNNs_cgcnn/tree/af4c0bf6606da1b46887ed8c29521d199d5e2798), [e3nn](https://github.com/learningmatter-mit/PerovskiteOrderingGCNNs_e3nn/tree/408b90e922a2a9c7bae2ad95433aae97d1a58494), [PaiNN](https://github.com/learningmatter-mit/PerovskiteOrderingGCNNs_painn/tree/e7980a52af4936addc5fb03dbc50d4fc74fe98fc), and [ALIGNN](https://github.com/jiayu-peng-lab/PerovskiteOrderingGCNNs_alignn/tree/805884f442cb7cee2c515bc7cbd15b4ef2f31ee2), which are included as submodules in this repo. If there are any changes in their corresponding GitHub repos, the following command will update the submodules in this repo:
```
git submodule update --remote --merge
```

---

## 🚀 Usage

### 🖥️ If working on a Linux workstation

To automatically download all required data and models and set up the conda environment, run:
```
bash scripts/setup_project.sh
```
This script will:
- Download and extract the datasets and model files from Zenodo.
- Set up the Conda environment (using the provided [environment.yml](environment.yml) if available).
- Ensure you are ready to run the notebooks or scripts.

Alternatively, you can download all our data and trained models manually; they are archived on Zenodo ([DOI: 10.5281/zenodo.13820311](https://doi.org/10.5281/zenodo.13820311)) and Materials Data Facility ([DOI: 10.18126/ncqt-rh18](https://doi.org/10.18126/ncqt-rh18)). Please place all data and model files in the corresponding directories and then refer to the following Jupyter notebooks to reproduce the results of our paper. Moreover, if you want to install the Conda environment manually, this repository requires the following packages to run correctly:
```
pandas            1.5.3
scipy             1.10.1
numpy             1.24.3
scikit-learn      1.2.2
matplotlib        3.7.1
seaborn           0.12.2
pymatgen          2023.5.10
ase               3.22.1
rdkit             2023.3.1
e3fp              1.2.5
pytorch           1.13.1
pytorch-cuda      11.7
pytorch-sparse    0.6.17
pytorch-scatter   2.1.1
pytorch-cluster   1.6.1
torchvision       0.14.1
torchaudio        0.13.1
pyg               2.3.0
e3nn              0.5.1
wandb             0.16.3
gdown             4.7.1
mscorefonts       0.0.1
boken             3.3.4
```

All these packages can be installed using the [`environment.yml`](environment.yml) file and Conda:
```
conda env create -f environment.yml
conda activate Perovskite_ML_Environment
```

### DGL (for ALIGNN only)

ALIGNN requires DGL. We keep `dgl/` untracked in Git; install DGL locally in your environment:

- If you installed PyTorch 1.13.1 with CUDA 11.7 (as specified above):

```bash
# Option A: pip (recommended)
pip install dgl-cu117 -f https://data.dgl.ai/wheels/torch-1.13/cu117/repo.html

# Option B: conda
conda install -c dglteam/label/cu117 dgl
```

Verify the install:

```bash
python -c "import dgl, torch; print('DGL', dgl.__version__, 'Torch', torch.__version__)"
```

If your CUDA/Torch versions differ, pick the matching wheel from the DGL wheel index (`https://www.dgl.ai/pages/start.html`).

Afterwards, you can run the following three notebooks to reproduce the main results of this paper:
- [`1_model_training.ipynb`](1_model_training.ipynb): Train GCNNs and conduct hyperparameter optimization.
- [`2_model_inference.ipynb`](2_model_inference.ipynb): Verify performance, select top models, compute predictions, and extract latent embeddings.
- [`3_model_analysis.ipynb`](3_model_analysis.ipynb): Reproduce all major figures in the manuscript.

---

### 🖥️ If working on an HPC cluster

If you're working on an HPC cluster with SLURM (such as [UB CCR](https://www.buffalo.edu/ccr.html)), you can use `salloc` for interactive sessions or `sbatch` for batch jobs with your conda environment.

#### Interactive Session with `salloc`

To get an interactive session with GPU access:

```bash
salloc --partition=general-compute --qos=general-compute --mem=64G --time=72:00:00 --gpus-per-node=1
```

Once you have the interactive session, follow these steps:

1. **Navigate to your project directory:**
   ```bash
   cd /path/to/PerovskiteOrderingGCNNs
   ```

2. **Activate your conda environment:**
   ```bash
   conda activate Perovskite_ML_Environment
   ```

3. **Run your experiment:**
   ```bash
   python training/run_wandb_experiment.py --struct_type relaxed --model CGCNN --gpu 0 --budget 50 --training_fraction 1 --training_seed 0
   ```

#### Batch Jobs with `sbatch`

Create a SLURM batch script (e.g., `run_training.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=perovskite_training
#SBATCH --output=logs/training_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Create log directory
mkdir -p logs

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Perovskite_ML_Environment

# Set environment variables (optional, for cache directories)
export MPLCONFIGDIR=$HOME/.cache/matplotlib
export PIP_CACHE_DIR=$HOME/.cache/pip

# Run your experiment
python training/run_wandb_experiment.py --struct_type relaxed --model CGCNN --gpu 0 --budget 50 --training_fraction 1 --training_seed 0
```

**Important notes for using conda with sbatch:**

- **Activate conda properly**: Use `source $(conda info --base)/etc/profile.d/conda.sh` before `conda activate` to ensure conda is properly initialized in the batch script environment.
- **GPU selection**: The `--gpu 0` argument in your Python script will use the GPU allocated by SLURM via `--gres=gpu:1`.
- **Resume sweeps**: To resume a previous WandB sweep, add `--resume_sweep_id <sweep_id>` to the Python command.
- **Monitor jobs**: Use `squeue -u $USER` to check job status, and `tail -f logs/training_<job_id>.log` to monitor output.

Submit the job with:
```bash
sbatch run_training.sh
```

---

## 📖 Citation

If you use our codes, data, and/or models, please cite the following paper:

```bibtex
@article{peng2024learning,
  title={Learning Ordering in Crystalline Materials with Symmetry-Aware Graph Neural Networks},
  author={Jiayu Peng and James Damewood and Jessica Karaguesian and Jaclyn R. Lunger and Rafael Gómez-Bombarelli},
  journal={arXiv:2409.13851},
  url = {https://arxiv.org/abs/2409.13851},
  year={2024}
}
```
