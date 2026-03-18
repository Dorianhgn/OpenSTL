#!/bin/bash

#=======================================================================
# SLURM CONFIGURATION
#=======================================================================
#SBATCH --job-name=openstl-sevir-download
#SBATCH --partition=frida
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/job_%j_%x.out

#=======================================================================
# RUN THE JOB IN A CONTAINER
#=======================================================================

echo "Submitting SEVIR download job $SLURM_JOB_ID in container..."

# Configuration
CONTAINER_IMAGE="docker://dorianhgn/openstl-mamba:latest"
WORKSPACE_DIR="$HOME/OpenSTL"

# Modalities to download: vis ir069 ir107 vil lght
# Example: export SEVIR_MODALITIES="vil ir107"
export SEVIR_MODALITIES="${SEVIR_MODALITIES:-vil}"

# Whether to also generate processed HDF5 files
export SEVIR_PROCESS="${SEVIR_PROCESS:-1}"

mkdir -p logs

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Modalities: $SEVIR_MODALITIES"
echo "Process raw data: $SEVIR_PROCESS"
echo "========================================="

srun \
	--container-image=$CONTAINER_IMAGE \
	--no-container-mount-home \
	--container-mounts=$WORKSPACE_DIR:/workspace \
	--container-workdir=/workspace \
	bash -lc '
		set -euo pipefail

		echo "--- Inside container: pwd=$(pwd) ---"

		# Install OpenSTL in dev mode so the helper script can import the local codebase
		pip install -e .

		# Download SEVIR raw files and optionally generate processed HDF5 files
		if [[ "${SEVIR_PROCESS:-1}" == "1" ]]; then
			bash tools/prepare_data/download_sevir.sh --modalities ${SEVIR_MODALITIES}
		else
			bash tools/prepare_data/download_sevir.sh --modalities ${SEVIR_MODALITIES} --no-process
		fi

		echo "--- SEVIR download finished ---"
	'

echo "Job finished at $(date)"
