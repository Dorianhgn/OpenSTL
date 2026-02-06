# 🐳 Docker/Singularity Setup for HPC

This directory contains everything you need to run OpenSTL with Mamba-SSM on HPC clusters.

## 📋 Quick Start

### Option 1: Use Pre-built Container (Recommended)

https://hub.docker.com/r/dorianhgn/openstl-mamba

#### Run docker locally 

```bash
docker pull dorianhgn/openstl-mamba:latest
git clone https://github.com/Dorianhgn/OpenSTL.git
cd OpenSTL
docker run -it --rm \
    --gpus all \
    -v $HOME/OpenSTL:/workspace \
    -w /workspace \
    dorianhgn/openstl-mamba:latest
```

#### Laucnhing interactive sessions

```bash
srun -p dev -c8 --mem=128G --gres=gpu:1 --time=0-12:00:00 \
   --container-image=dorianhgn/openstl-mamba:latest \
   --container-name=openstl_cont \
   --job-name openstl-work \
	--no-container-mount-home \
	--container-mounts=$HOME/OpenSTL:/workspace \
	--container-workdir=/workspace
	--pty bash
```

or

```bash
stunnel -p dev -c8 --mem=128G --gres=gpu:1 --time=0-12:00:00 \
   --container-image=dorianhgn/openstl-mamba:latest \
   --container-name=openstl_cont \
   --job-name openstl-work \
	--no-container-mount-home \
	--container-mounts=$HOME/OpenSTL:/workspace \
	--container-workdir=/workspace
```


#### Launching jobs

Refer to `jobs/slurm_train.sh` for an example of how to submit training jobs using the container.


### Option 2: Build Your Own Container

1. **Build Docker image locally**:
   ```bash
   docker build -t your_dockerhub_username/openstl-mamba:latest .
   docker push your_dockerhub_username/openstl-mamba:latest
   ```

2. **Update scripts**: Edit `scripts/slurm_train.sh` and `scripts/interactive_session.sh` to use your image.

3. **Run on HPC**: Submit jobs as in Option 1.

## 📁 Files

- `Dockerfile` - Container definition with all dependencies
- `HPC_CONTAINER_GUIDE.md` - Detailed setup instructions
- `scripts/slurm_train.sh` - Slurm job submission script
- `scripts/interactive_session.sh` - Interactive debugging session

## 🔑 Key Features

- **PyTorch Release 24.10** with CUDA 12.6 (compatible with H100/A100)
- **Ubuntu 22.04** including **Python 3.10**
- **mamba-ssm 2.2.2** with causal-conv1d 1.4.0 pre-installed
- **Lightning 2.2.1** for distributed training
- All OpenSTL dependencies included

## ⚡ Why Use Containers?

- ✅ No reinstallation needed (saves ~15 min per job)
- ✅ Reproducible environment
- ✅ Works on any HPC with Singularity/Apptainer
- ✅ Easier dependency management