#!/bin/bash

#=======================================================================
# SLURM CONFIGURATION
#=======================================================================
#SBATCH --job-name=openstl-train
#SBATCH --partition=frida
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=H100:1          # or A100:1 depending on your GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/job_%j_%x.out

#=======================================================================
# RUN THE JOB IN A CONTAINER
#====================================

echo "Submitting job $SLURM_JOB_ID in container..."

# Configuration
CONTAINER_IMAGE="docker://dorianhgn/openstl-mamba:latest"  
WORKSPACE_DIR="$HOME/OpenSTL"
DATASET="mmnist"                                    # Dataset name (e.g., mmnist, etc.)
CONFIG_FILE="configs/mmnist/simvp_gsta.py"
EXPERIMENT_NAME="mmnist_simvp_gsta"

# Créer le répertoire de logs s'il n'existe pas
mkdir -p logs

# Informations sur le job
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG_FILE"
echo "========================================="

srun \
  --container-image=$CONTAINER_IMAGE \
  --no-container-mount-home \
  --container-mounts=$WORKSPACE_DIR:/workspace \
  --container-workdir=/workspace \
  bash -lc '
    echo "--- Inside container: pwd=$(pwd) ---"

    # 1) Install OpenSTL in dev mode
    pip install -e .

    # 2) Start training
    python tools/train.py -d $DATASET -c $CONFIG_FILE --ex_name $EXPERIMENT_NAME


    echo "--- Training finished ---"
  '

echo "Job finished at $(date)"
