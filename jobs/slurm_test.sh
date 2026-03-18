#!/bin/bash

#=======================================================================
# SLURM CONFIGURATION
#=======================================================================
#SBATCH --job-name=openstl-test
#SBATCH --partition=frida
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1          # or A100:1 depending on your GPU
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
export DATASET="mmnist"                                    # Dataset name (e.g., mmnist, etc.)
export CONFIG_FILE="work_dirs/mmnist_jvm_ens_5/config_base.py"
export EXPERIMENT_NAME="mmnist_jvm_ens_5"
export CKPT_PATH="work_dirs/mmnist_jvm/checkpoints/best.ckpt"

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

    # 2) Start testing
    python tools/test.py -d $DATASET -c $CONFIG_FILE --ex_name $EXPERIMENT_NAME


    echo "--- Testing finished ---"
  '

echo "Job finished at $(date)"
