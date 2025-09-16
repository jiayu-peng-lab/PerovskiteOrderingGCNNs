## ALIGNN training container for HPC

This repo includes a CUDA-enabled Docker image to run ALIGNN with DGL reliably on HPC. You can either run Docker on a GPU node or convert the image to a Singularity/Apptainer `.sif` that runs on your cluster.

### Contents
- `docker/Dockerfile`: CUDA 12.1 + PyTorch 2.4 runtime, installs DGL (CUDA 12.1 wheels) and Python deps.
- `requirements-alignn.txt`: Pinned Python packages used by ALIGNN training.
- `docker/train_alignn.py`: Minimal runner to verify environment and optionally invoke training.

### Build the image
```bash
cd /data/users/kritarth/PerovskiteOrderingGCNNs
docker build -t alignn-hpc -f docker/Dockerfile .
```

### Test locally (with GPU)
```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  alignn-hpc:latest python /usr/local/bin/train_alignn.py --check
```
You should see Torch/DGL versions, CUDA availability, and GPU count.

### Run training
Mount your data and outputs and pass your config:
```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/your/data:/workspace/data \
  -v /path/to/output:/workspace/output \
  alignn-hpc:latest \
  alignn-train --root /workspace/data --config /workspace/path/to/config.yaml --output /workspace/output
```

Alternatively, use the helper:
```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  alignn-hpc:latest \
  python /usr/local/bin/train_alignn.py --data /workspace/data --config /workspace/config.yaml --out /workspace/output
```

### Convert Docker image to Singularity/Apptainer
If your HPC uses Apptainer:
```bash
apptainer build alignn-hpc.sif docker-daemon://alignn-hpc:latest
```
Or with Singularity (older):
```bash
singularity build alignn-hpc.sif docker-daemon://alignn-hpc:latest
```

Run on HPC GPU node:
```bash
apptainer exec --nv alignn-hpc.sif \
  alignn-train --root /path/on/hpc/data --config /path/on/hpc/config.yaml --output /path/on/hpc/output
```

If you need internet-restricted nodes, build the Docker image locally, then copy `alignn-hpc.sif` to the cluster.

### Notes on DGL and CUDA
- The Dockerfile installs DGL from the official CUDA 12.1 wheels (`data.dgl.ai`). If your cluster has a different CUDA driver/toolkit, prefer the Apptainer/SIF flow (image carries its runtime).
- For CPU-only, replace the DGL install in the Dockerfile with `pip install dgl` and use a CPU PyTorch base.

### Troubleshooting
- If `--nv` isn’t available, request a GPU node or correct module environment.
- If `GLIBC`/`libcuda.so` errors occur on HPC, use the SIF image built locally and run with `apptainer exec --nv`.
- If `alignn-train` isn’t found, ensure `alignn` is installed in the image and `$PATH` is correct; try running `python -c "import alignn, dgl, torch; print('ok')"` inside the container.


