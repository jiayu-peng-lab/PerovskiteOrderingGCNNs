# CUDA-enabled Dockerfile for Perovskite ML (Python 3.11, CUDA 12.1)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential \
    git \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set workdir
WORKDIR /workspace

# Copy requirements and install core dependencies (except torch/torchvision/torchaudio)
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements (except torch-geometric and its extensions)
RUN pip install --no-cache-dir --upgrade \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    wandb \
    e3nn \
    ase \
    pymatgen \
    mp-api \
    plotly \
    tqdm

# Install PyTorch Geometric and its extensions for CUDA 12.1
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
RUN pip install torch-geometric

# Copy the rest of the codebase
COPY . .

# Set default command
CMD ["/bin/bash"] 