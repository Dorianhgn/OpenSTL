# Prediction Saving & Comparison Visualization Scripts

This directory contains scripts for saving model predictions and creating comparison visualizations.

## Overview

Three scripts are provided:

1. **`save_stvmamba_predictions.py`** - Saves STVMamba predictions (deterministic, fast)
2. **`save_jvm_ensemble_predictions.py`** - Saves JvM/CFM ensemble predictions (ODE-based)
3. **`plot_comparison_vizuals.py`** - Creates comparison visualizations from saved tensors

## Workflow

### Step 1: Save Model Predictions

#### STVMamba (Deterministic)

```bash
# Saves full test set predictions
python scripts/phase5/save_stvmamba_predictions.py \
    --config experiments/duaod_final/stvmamba_fusion/4.5M/version_0_full_test/viz_config.yaml

# Or save only first batch (faster for testing)
python scripts/phase5/save_stvmamba_predictions.py \
    --config experiments/duaod_final/stvmamba_fusion/4.5M/version_0_full_test/viz_config.yaml
```

#### JvM / CFM (Ensemble ODE)

```bash
# JvM with 5 ensemble members
python scripts/phase5/save_jvm_ensemble_predictions.py \
    --config experiments/duaod_final/spadeJvM/small/version_1_test_full_ep27_n-ens5/viz_config.yaml

# CFM with 5 ensemble members  
python scripts/phase5/save_jvm_ensemble_predictions.py \
    --config experiments/duaod_final/spadeCFM_patches/small/version_0_test_full_ep11_n-ens5/viz_config.yaml

# Override ensemble count
python scripts/phase5/save_jvm_ensemble_predictions.py \
    --config path/to/viz_config.yaml --n_ens 10
```

### Step 2: Create Comparison Plots

```bash
# Create grid comparison (multiple frames)
python scripts/phase5/plot_comparison_vizuals.py \
    --config experiments/duaod_final/final_vizualisations/final_viz_config.yaml

# Create animation
python scripts/phase5/plot_comparison_vizuals.py \
    --config experiments/duaod_final/final_vizualisations/final_viz_config.yaml --animation

# Single frame comparison
python scripts/phase5/plot_comparison_vizuals.py \
    --config experiments/duaod_final/final_vizualisations/final_viz_config.yaml --frame_idx 12
```

## Configuration Files

### `viz_config.yaml` (for prediction saving)

```yaml
# STVMamba format
config: path/to/training/config.yaml
ckpt_path: path/to/checkpoint.ckpt
batch_idx: 0  # optional, if not set saves full test set

# JvM/CFM format (adds n_ens)
config: path/to/training/config.yaml
ckpt_path: path/to/checkpoint.ckpt
n_ens: 5  # number of ensemble predictions
batch_idx: 0  # which batch (only one sample per run for ensemble)
```

### `final_viz_config.yaml` (for comparison plots)

```yaml
output_dir: 'path/to/output/directory'

tensor_dirs:
  Model_A: 'path/to/model_a/tensors'
  Model_B: 'path/to/model_b/tensors'
  Model_C: 'path/to/model_c/tensors'

fig_size: [20, 12]
sample_idx: 0  # which sample to visualize
```

## Output Files

### Prediction Saving

**STVMamba (deterministic):**
- `tensors/predictions_batch{N}.pt` or `tensors/predictions_full.pt`
- `tensors/ground_truths_batch{N}.pt` or `tensors/ground_truths_full.pt`
- `tensors/inputs_batch{N}.pt` or `tensors/inputs_full.pt`
- `tensors/metadata_*.yaml`

**JvM/CFM (ensemble):**
- `tensors/ensemble_predictions_batch{B}_sample{S}_ens{N}.pt` - All ensemble members
- `tensors/ensemble_mean_batch{B}_sample{S}_ens{N}.pt` - Ensemble mean
- `tensors/ground_truth_batch{B}_sample{S}_ens{N}.pt` - Ground truth
- `tensors/condition_batch{B}_sample{S}_ens{N}.pt` - Input conditions
- `tensors/x_past_batch{B}_sample{S}_ens{N}.pt` - Past AOD
- `tensors/metadata_*.yaml`

### Comparison Visualization

- `comparison_grid.png` - Multi-frame grid (default)
- `comparison_frame{XX}.png` - Single frame comparisons
- `comparison_animation.mp4` - Animated comparison (with --animation flag)

## Requirements

- PyTorch
- PyTorch Lightning
- Cartopy (for geographic plots)
- Matplotlib
- NumPy
- YAML

Install cartopy:
```bash
conda install -c conda-forge cartopy
# or
pip install cartopy
```

For animations, ffmpeg is required:
```bash
conda install -c conda-forge ffmpeg
# or
apt-get install ffmpeg
```
