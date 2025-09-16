# Base image with CUDA 12.1, cuDNN 8, and PyTorch 2.4 preinstalled
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    DGLBACKEND=pytorch \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN python -m pip install --upgrade pip wheel setuptools

# Install DGL matching CUDA 12.1 and Torch 2.4 from official wheel repo
# If you need CPU-only, replace with: `pip install dgl`
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    dgl -f https://data.dgl.ai/wheels/cu121/repo.html

# Copy requirements and install
COPY requirements-alignn.txt /tmp/requirements-alignn.txt
RUN pip install -r /tmp/requirements-alignn.txt && rm -f /tmp/requirements-alignn.txt

# Create a work directory
WORKDIR /workspace

# Copy optional helper script
COPY train_alignn.py /usr/local/bin/train_alignn.py
RUN chmod +x /usr/local/bin/train_alignn.py

# Default command prints versions to validate runtime; override with your train command
CMD ["python", "/usr/local/bin/train_alignn.py", "--check"]


