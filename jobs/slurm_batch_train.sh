#!/bin/bash

#=======================================================================
# SLURM CONFIGURATION
#=======================================================================
#SBATCH --job-name=open_stl_batch_train    # Name of the job
#SBATCH --partition=frida                  # The partition (queue) from your docs
#SBATCH --time=7-00:00:00                  # Wall time limit (D-HH:MM:SS)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=1                # Number of tasks per node
#SBATCH --gpus-per-node=1                  # 1 GPU only
#SBATCH --cpus-per-task=6                  # CPUs for data loading
#SBATCH --mem-per-gpu=96G                  # Total memory for the job
#SBATCH --output=logs/job_%j_%x.out        # Standard output file

#=======================================================================
# RUN BATCH JOBS IN A CONTAINER (single GPU, sequential)
#=======================================================================

echo "Submitting batch job $SLURM_JOB_ID in container..."

# Configuration
CONTAINER_IMAGE="docker://dorianhgn/openstl-mamba:latest"
CONTAINER_NAME="container_${SLURM_JOB_ID}"
export DATASET="mmnist"
export EXP_JSON_FILE="configs/mmnist/mini-exp.json"

mkdir -p logs

BIND="${PWD}:/workspace:rw"

echo "Node  = $(hostname)"
echo "Nodes = $SLURM_JOB_NODELIST  ($SLURM_NNODES nœud(s))"
echo "GPUs/node        = $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"

echo "→ Job $SLURM_JOB_ID : création du container et installation des dépendances…"

srun \
  --ntasks=1 \
  --container-image=${CONTAINER_IMAGE} \
  --container-name=${CONTAINER_NAME} \
  --container-writable \
  --no-container-mount-home \
  --container-mounts=${BIND} \
  --container-workdir=/workspace \
  bash -lc '
    set -euo pipefail
    pip install -e .
  '

if [ $? -ne 0 ]; then
  echo "❌ Échec de l installation des dépendances"
  exit 1
fi

echo "→ Lancement des trainings séquentiels sur 1 GPU…"

srun \
  --ntasks=1 \
  --container-name=${CONTAINER_NAME} \
  --no-container-mount-home \
  --container-mounts=${BIND} \
  --container-workdir=/workspace \
  bash -lc '
    set -euo pipefail

    echo "→ Parsing experiments from $EXP_JSON_FILE..."

    while IFS=":" read -r ex_name config_file; do
      ex_name=$(echo "$ex_name" | tr -d " \",{}")
      config_file=$(echo "$config_file" | tr -d " \",{}")

      if [ -z "$ex_name" ] || [ -z "$config_file" ]; then
        continue
      fi

      echo ""
      echo "======================================================================="
      echo "→ Starting experiment: $ex_name"
      echo "   Config: $config_file"
      echo "======================================================================="

      python tools/train.py \
        -d $DATASET \
        -c $config_file \
        --ex_name $ex_name

      TRAIN_EXIT_CODE=$?

      if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "❌ Experiment $ex_name failed with exit code $TRAIN_EXIT_CODE"
        echo "   Continuing with next experiment..."
      else
        echo "✅ Experiment $ex_name completed successfully"
      fi

    done < <(grep -o "\"[^\"]*\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" $EXP_JSON_FILE)

    echo ""
    echo "======================================================================="
    echo "✅ All batch trainings completed"
    echo "======================================================================="
  '

echo "→ Job $SLURM_JOB_ID terminé à $(date)"
