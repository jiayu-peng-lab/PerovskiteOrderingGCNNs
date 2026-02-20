#!/bin/bash

# Function to print status messages
echo_status() {
    echo -e "\n=============================="
    echo "$1"
    echo -e "==============================\n"
}

# Download URLs
data_url="https://zenodo.org/records/13820311/files/data_v1.1.tar.gz?download=1"
best_models_url="https://zenodo.org/records/18707688/files/best_models_v2.tar.gz?download=1"
saved_models_url="https://zenodo.org/records/13820311/files/saved_models.tar.gz?download=1"

# Output filenames
data_tar="data_v1.1.tar.gz"
best_models_tar="best_models_v2.tar.gz"
saved_models_tar="saved_models.tar.gz"

# Check if pv is installed for progress bar, else try to install it
PV_AVAILABLE=1
if ! command -v pv &> /dev/null; then
    echo_status "'pv' not found. Attempting to install for extraction progress bars..."
    apt-get update && apt-get install -y pv
    if ! command -v pv &> /dev/null; then
        echo_status "'pv' could not be installed. Extraction progress bars will be disabled."
        PV_AVAILABLE=0
    fi
fi

# Download and extract function with progress bar if available
download_and_extract() {
    url=$1
    tarfile=$2
    target_dir=$3
    
    echo_status "Downloading $tarfile ..."
    wget --show-progress -O "$tarfile" "$url"
    
    echo_status "Removing existing $target_dir ..."
    rm -rf "$target_dir"
    
    if [ "$PV_AVAILABLE" -eq 1 ]; then
        echo_status "Extracting $tarfile ... (progress below)"
        tar -xzvf "$tarfile" | pv -lep -s $(tar -tzf "$tarfile" | wc -l) > /dev/null
    else
        echo_status "Extracting $tarfile ... (no progress bar)"
        tar -xzvf "$tarfile"
    fi
    
    echo_status "$target_dir is ready."
}

# Set up conda environment in background
setup_conda_env() {
    echo_status "Setting up conda environment ..."
    if [ -f "models/PerovskiteOrderingGCNNs_painn/environment.yml" ]; then
        conda env create -f models/PerovskiteOrderingGCNNs_painn/environment.yml
    elif [ -f "wandb/latest-run/files/conda-environment.yaml" ]; then
        conda env create -f wandb/latest-run/files/conda-environment.yaml
    else
        echo "No environment.yml or conda-environment.yaml found!"
    fi
    echo_status "Conda environment setup complete."
}

# Run downloads and extraction in background
(download_and_extract "$data_url" "$data_tar" "data" && \
 download_and_extract "$best_models_url" "$best_models_tar" "best_models" && \
 download_and_extract "$saved_models_url" "$saved_models_tar" "saved_models" && \
 echo_status "All downloads and extractions complete.") &

# Run conda setup in background
(setup_conda_env) &

# Wait for all background jobs to finish
wait

echo_status "Setup complete! You are ready to use the project." 