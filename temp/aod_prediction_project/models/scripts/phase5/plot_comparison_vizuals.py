#!/usr/bin/env python3
"""
Multi-Model Comparison Visualization Script

This script creates comparison plots for multiple models by loading pre-saved
prediction tensors and generating side-by-side visualizations.

Supports:
- Deterministic models (STVMamba): Single prediction per sample
- Ensemble models (JvM, CFM): Uses ensemble mean for comparison

Usage:
    python scripts/phase5/plot_comparison_vizuals.py \
        --config experiments/duaod_final/final_vizualisations/final_viz_config.yaml

The final_viz_config.yaml should contain:
    output_dir: 'experiments/duaod_final/final_vizualisations'
    
    tensor_dirs:
      stvmamba: 'experiments/duaod_final/stvmamba_fusion/4.5M/version_0_full_test/tensors'
      JvM_ens5: 'experiments/duaod_final/spadeJvM/small/version_1_test_full_ep27_n-ens5/tensors'
      patchCFM: 'experiments/duaod_final/spadeCFM_patches/small/version_0_test_full_ep11_n-ens5/tensors'
    
    fig_size: [15, 10]  # optional
    sample_idx: 0       # optional, which sample to plot
    frame_idx: 0        # optional, which forecast frame to plot (or 'all' for animation)
"""
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

# Import cartopy for geographic plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: Cartopy not available. Using simple matplotlib plots instead.")


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_tensor_data(tensor_dir: Path, model_type: str = "auto") -> Dict[str, torch.Tensor]:
    """
    Load tensors from a directory.
    
    Automatically detects whether it's a deterministic model (STVMamba) or
    ensemble model (JvM/CFM) based on available files.
    
    Args:
        tensor_dir: Directory containing saved tensors
        model_type: 'auto', 'deterministic', or 'ensemble'
    
    Returns:
        Dict with keys: 'predictions', 'ground_truth', optionally 'ensemble_all'
    """
    tensor_dir = Path(tensor_dir)
    
    if not tensor_dir.exists():
        raise FileNotFoundError(f"Tensor directory not found: {tensor_dir}")
    
    data = {}
    
    # List available files
    available_files = list(tensor_dir.glob("*.pt"))
    file_names = [f.name for f in available_files]
    
    # Auto-detect model type
    if model_type == "auto":
        if any("ensemble" in f for f in file_names):
            model_type = "ensemble"
        elif any("predictions" in f for f in file_names):
            model_type = "deterministic"
        else:
            raise ValueError(f"Cannot determine model type from files: {file_names}")
    
    if model_type == "deterministic":
        # STVMamba format: predictions_*.pt, ground_truths_*.pt
        pred_files = list(tensor_dir.glob("predictions*.pt"))
        gt_files = list(tensor_dir.glob("ground_truths*.pt"))
        
        if pred_files:
            data['predictions'] = torch.load(pred_files[0], weights_only=True)
        if gt_files:
            data['ground_truth'] = torch.load(gt_files[0], weights_only=True)
            
    elif model_type == "ensemble":
        # JvM/CFM format: ensemble_mean_*.pt, ensemble_predictions_*.pt, ground_truth_*.pt
        mean_files = list(tensor_dir.glob("ensemble_mean*.pt"))
        all_files = list(tensor_dir.glob("ensemble_predictions*.pt"))
        gt_files = list(tensor_dir.glob("ground_truth*.pt"))
        
        if mean_files:
            data['predictions'] = torch.load(mean_files[0], weights_only=True)
        if all_files:
            data['ensemble_all'] = torch.load(all_files[0], weights_only=True)
        if gt_files:
            data['ground_truth'] = torch.load(gt_files[0], weights_only=True)
    
    # Load metadata if available
    metadata_files = list(tensor_dir.glob("metadata*.yaml"))
    if metadata_files:
        with open(metadata_files[0], 'r') as f:
            data['metadata'] = yaml.safe_load(f)
    
    return data


def load_all_models_data(tensor_dirs: Dict[str, str]) -> Dict[str, Dict]:
    """
    Load data from all models specified in config.
    
    Args:
        tensor_dirs: Dict mapping model_name -> tensor_directory_path
    
    Returns:
        Dict mapping model_name -> data_dict
    """
    all_data = {}
    
    for model_name, tensor_dir in tensor_dirs.items():
        print(f"Loading {model_name} from {tensor_dir}...")
        try:
            data = load_tensor_data(Path(tensor_dir))
            all_data[model_name] = data
            
            if 'predictions' in data:
                print(f"  ✓ Predictions shape: {data['predictions'].shape}")
            if 'ground_truth' in data:
                print(f"  ✓ Ground truth shape: {data['ground_truth'].shape}")
            if 'ensemble_all' in data:
                print(f"  ✓ Ensemble all shape: {data['ensemble_all'].shape}")
        except Exception as e:
            print(f"  ✗ Error loading {model_name}: {e}")
    
    return all_data


# ============================================================================
# Visualization Functions
# ============================================================================

def get_extent_from_config(data_config_path: Optional[str] = None) -> List[float]:
    """
    Get map extent from data config or return default (Sahara region).
    
    Returns:
        [lon_min, lon_max, lat_min, lat_max]
    """
    # Default extent for Sahara dust region - matches animate_predictions.py
    default_extent = [-25, 65, 10, 65]  # Western Sahara to Middle East
    
    if data_config_path:
        try:
            with open(data_config_path, 'r') as f:
                config = yaml.safe_load(f)
            # Try to extract extent from config if available
            # For now, return default
        except:
            pass
    
    return default_extent


def create_comparison_plot_cartopy(
    all_data: Dict[str, Dict],
    output_dir: Path,
    sample_idx: int = 0,
    frame_idx: int = 0,
    extent: List[float] = None,
    figsize: Tuple[int, int] = (20, 8),
):
    """
    Create a comparison plot with cartopy for geographic visualization.
    
    Layout: Ground Truth | Model 1 | Model 2 | Model 3 | ...
    
    Args:
        all_data: Dict mapping model_name -> data_dict
        output_dir: Directory to save plot
        sample_idx: Which sample to plot (for multi-sample tensors)
        frame_idx: Which forecast frame to plot
        extent: Map extent [lon_min, lon_max, lat_min, lat_max]
        figsize: Figure size
    """
    if extent is None:
        extent = get_extent_from_config()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model names and data
    model_names = list(all_data.keys())
    n_models = len(model_names)
    
    # Find ground truth (use first available)
    gt_data = None
    for model_name, data in all_data.items():
        if 'ground_truth' in data:
            gt_tensor = data['ground_truth']
            # Handle different tensor shapes
            if gt_tensor.dim() == 5:  # (B, T, C, H, W)
                gt_data = gt_tensor[sample_idx, frame_idx, 0].numpy()
            elif gt_tensor.dim() == 4:  # (T, C, H, W) or (N, T, C, H, W) squeezed
                if gt_tensor.shape[0] > gt_tensor.shape[1]:  # Likely (N, T, C, H)
                    gt_data = gt_tensor[sample_idx, frame_idx, 0].numpy()
                else:  # Likely (T, C, H, W)
                    gt_data = gt_tensor[frame_idx, 0].numpy()
            elif gt_tensor.dim() == 3:  # (T, H, W)
                gt_data = gt_tensor[frame_idx].numpy()
            break
    
    if gt_data is None:
        print("Warning: No ground truth found. Will only plot predictions.")
    
    # Total columns: GT + all models
    n_cols = (1 if gt_data is not None else 0) + n_models
    
    # Create figure
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()} if CARTOPY_AVAILABLE else {}
    )
    
    if n_cols == 1:
        axes = [axes]
    
    # Calculate global vmax for consistent color scaling
    all_values = []
    if gt_data is not None:
        all_values.append(gt_data)
    
    for model_name, data in all_data.items():
        if 'predictions' in data:
            pred_tensor = data['predictions']
            if pred_tensor.dim() == 5:
                pred_data = pred_tensor[sample_idx, frame_idx, 0].numpy()
            elif pred_tensor.dim() == 4:
                if pred_tensor.shape[0] > pred_tensor.shape[1]:
                    pred_data = pred_tensor[sample_idx, frame_idx, 0].numpy()
                else:
                    pred_data = pred_tensor[frame_idx, 0].numpy()
            elif pred_tensor.dim() == 3:
                pred_data = pred_tensor[frame_idx].numpy()
            else:
                continue
            all_values.append(pred_data)
    
    # ========== COLOR INTENSITY CONTROLS ==========
    # OPTION 1: Set vmax directly (recommended for consistent colors)
    vmax = 0.10  # Set this to your desired max AOD value (e.g., 0.10, 0.15, 0.20)
    
    # OPTION 2: Use percentile-based vmax (comment out OPTION 1 and uncomment below)
    # percentile_val = 85  # Lower = darker colors
    # vmax = np.percentile(np.concatenate([v.flatten() for v in all_values]), percentile_val) if all_values else 0.1
    # ===============================================
    
    vmin = 0
    cmap = 'YlOrBr'
    print(f"  → Using vmax={vmax:.4f} for color scaling")
    
    col_idx = 0
    
    # Plot Ground Truth
    if gt_data is not None:
        ax = axes[col_idx]
        if CARTOPY_AVAILABLE:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            # Add gridlines like in animate_predictions.py
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                        linewidth=0.5, alpha=0.5)
            im = ax.imshow(
                gt_data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                origin='upper', transform=ccrs.PlateCarree()
            )
        else:
            im = ax.imshow(gt_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
        col_idx += 1
    
    # Plot each model's predictions
    for model_name, data in all_data.items():
        if 'predictions' not in data:
            continue
        
        ax = axes[col_idx]
        pred_tensor = data['predictions']
        
        # Extract prediction data based on tensor shape
        if pred_tensor.dim() == 5:
            pred_data = pred_tensor[sample_idx, frame_idx, 0].numpy()
        elif pred_tensor.dim() == 4:
            if pred_tensor.shape[0] > pred_tensor.shape[1]:
                pred_data = pred_tensor[sample_idx, frame_idx, 0].numpy()
            else:
                pred_data = pred_tensor[frame_idx, 0].numpy()
        elif pred_tensor.dim() == 3:
            pred_data = pred_tensor[frame_idx].numpy()
        else:
            print(f"Unknown tensor shape for {model_name}: {pred_tensor.shape}")
            continue
        
        if CARTOPY_AVAILABLE:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            # Add gridlines like in animate_predictions.py
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                        linewidth=0.5, alpha=0.5)
            im = ax.imshow(
                pred_data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                origin='upper', transform=ccrs.PlateCarree()
            )
        else:
            im = ax.imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
        
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        col_idx += 1
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', shrink=0.6, pad=0.08)
    cbar.set_label('AOD (550nm)', fontsize=12)
    
    # Main title
    hours_per_frame = 5  # 5-hourly data (24 frames = 120 hours)
    forecast_hour = (frame_idx + 1) * hours_per_frame
    fig.suptitle(
        f'Model Comparison - Forecast Hour: h+{forecast_hour}',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = output_dir / f"comparison_frame{frame_idx:02d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved: {output_path}")
    
    return output_path


def create_comparison_grid(
    all_data: Dict[str, Dict],
    output_dir: Path,
    sample_idx: int = 0,
    frame_indices: List[int] = None,
    extent: List[float] = None,
    figsize: Tuple[int, int] = (20, 16),
):
    """
    Create a grid comparison plot showing multiple frames.
    
    Layout: 
        Rows = frames (time steps)
        Columns = GT | Model 1 | Model 2 | ...
    
    Args:
        all_data: Dict mapping model_name -> data_dict
        output_dir: Directory to save plot
        sample_idx: Which sample to plot
        frame_indices: Which frames to include (default: [0, 8, 16, 23])
        extent: Map extent
        figsize: Figure size
    """
    if extent is None:
        extent = get_extent_from_config()
    
    if frame_indices is None:
        # Default: start, 1/3, 2/3, end of forecast horizon
        frame_indices = [0, 8, 16, 23]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_names = list(all_data.keys())
    n_models = len(model_names)
    n_frames = len(frame_indices)
    
    # Find ground truth
    gt_tensor = None
    for model_name, data in all_data.items():
        if 'ground_truth' in data:
            gt_tensor = data['ground_truth']
            break
    
    n_cols = (1 if gt_tensor is not None else 0) + n_models
    n_rows = n_frames
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()} if CARTOPY_AVAILABLE else {}
    )
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Calculate global vmax
    all_values = []
    for fi in frame_indices:
        if gt_tensor is not None:
            gt_data = _extract_frame(gt_tensor, sample_idx, fi)
            if gt_data is not None:
                all_values.append(gt_data)
        
        for model_name, data in all_data.items():
            if 'predictions' in data:
                pred_data = _extract_frame(data['predictions'], sample_idx, fi)
                if pred_data is not None:
                    all_values.append(pred_data)
    
    # ========== COLOR INTENSITY CONTROLS ==========
    # OPTION 1: Set vmax directly (recommended for consistent colors)
    vmax = 0.10  # Set this to your desired max AOD value (e.g., 0.10, 0.15, 0.20)
    
    # OPTION 2: Use percentile-based vmax (comment out OPTION 1 and uncomment below)
    # percentile_val = 85  # Lower = darker colors
    # vmax = np.percentile(np.concatenate([v.flatten() for v in all_values]), percentile_val) if all_values else 0.1
    # ===============================================
    
    vmin = 0
    cmap = 'YlOrBr'
    print(f"  → Using vmax={vmax:.4f} for color scaling")
    
    hours_per_frame = 5  # 5-hourly data (24 frames = 120 hours)
    
    for row, frame_idx in enumerate(frame_indices):
        col = 0
        
        # Ground truth column
        if gt_tensor is not None:
            ax = axes[row, col]
            gt_data = _extract_frame(gt_tensor, sample_idx, frame_idx)
            _plot_map(ax, gt_data, extent, cmap, vmin, vmax)
            if row == 0:
                ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
            forecast_hour = (frame_idx + 1) * hours_per_frame
            ax.set_ylabel(f'h+{forecast_hour}', fontsize=11, fontweight='bold')
            col += 1
        
        # Model columns
        for model_name, data in all_data.items():
            if 'predictions' not in data:
                continue
            
            ax = axes[row, col]
            pred_data = _extract_frame(data['predictions'], sample_idx, frame_idx)
            _plot_map(ax, pred_data, extent, cmap, vmin, vmax)
            if row == 0:
                ax.set_title(model_name, fontsize=12, fontweight='bold')
            col += 1
    
    # Add colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        ax=axes, orientation='vertical', shrink=0.8, pad=0.02
    )
    cbar.set_label('AOD (550nm)', fontsize=12)
    
    # Main title
    fig.suptitle('Multi-Model Comparison Across Forecast Horizons', fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / "comparison_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Saved grid comparison: {output_path}")
    
    return output_path


def _extract_frame(tensor: torch.Tensor, sample_idx: int, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a single frame from tensor, handling different shapes."""
    try:
        if tensor.dim() == 5:  # (B, T, C, H, W)
            return tensor[sample_idx, frame_idx, 0].numpy()
        elif tensor.dim() == 4:  # (T, C, H, W) or (N, T, C, H, W) squeezed
            if tensor.shape[1] == 1:  # (T, C=1, H, W)
                return tensor[frame_idx, 0].numpy()
            else:
                return tensor[frame_idx, 0].numpy()
        elif tensor.dim() == 3:  # (T, H, W)
            return tensor[frame_idx].numpy()
        else:
            return None
    except IndexError:
        return None


def _plot_map(ax, data: np.ndarray, extent: List[float], cmap: str, vmin: float, vmax: float):
    """Plot data on a map axis."""
    if data is None:
        ax.axis('off')
        return
    
    if CARTOPY_AVAILABLE:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', color='black', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.3)  # Simplified gridlines for grid view
        ax.imshow(
            data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
            origin='upper', transform=ccrs.PlateCarree()
        )
    else:
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')


def create_animation_comparison(
    all_data: Dict[str, Dict],
    output_dir: Path,
    sample_idx: int = 0,
    extent: List[float] = None,
    fps: int = 2,
    figsize: Tuple[int, int] = (20, 6),
):
    """
    Create an animated comparison of all models across all forecast frames.
    
    Args:
        all_data: Dict mapping model_name -> data_dict
        output_dir: Directory to save animation
        sample_idx: Which sample to animate
        extent: Map extent
        fps: Frames per second
        figsize: Figure size
    """
    if extent is None:
        extent = get_extent_from_config()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_names = list(all_data.keys())
    
    # Find forecast horizon from first available prediction
    # Need to correctly identify time dimension vs batch dimension
    n_frames = 24  # Default
    for model_name, data in all_data.items():
        if 'predictions' in data:
            pred_shape = data['predictions'].shape
            if len(pred_shape) == 5:  # (B, T, C, H, W)
                n_frames = pred_shape[1]  # Time is 2nd dimension
            elif len(pred_shape) == 4:  # (T, C, H, W) or (B, T, C, H)
                # Heuristic: time dimension is usually smaller than spatial
                # If dim0 >> dim1, likely (B, T, ...) else (T, C, ...)
                if pred_shape[0] > 50:  # Likely batch dimension
                    n_frames = pred_shape[1] if len(pred_shape) > 1 else pred_shape[0]
                else:
                    n_frames = pred_shape[0]  # Time dimension
            elif len(pred_shape) == 3:  # (T, H, W)
                n_frames = pred_shape[0]
            print(f"  → Detected {n_frames} time frames from {model_name}")
            break
    
    # Check if ground truth is available
    gt_tensor = None
    for model_name, data in all_data.items():
        if 'ground_truth' in data:
            gt_tensor = data['ground_truth']
            break
    
    n_cols = (1 if gt_tensor is not None else 0) + len(model_names)
    
    # Calculate global vmax
    all_values = []
    for fi in range(n_frames):
        if gt_tensor is not None:
            gt_data = _extract_frame(gt_tensor, sample_idx, fi)
            if gt_data is not None:
                all_values.append(gt_data)
        
        for model_name, data in all_data.items():
            if 'predictions' in data:
                pred_data = _extract_frame(data['predictions'], sample_idx, fi)
                if pred_data is not None:
                    all_values.append(pred_data)
    
    # ========== COLOR INTENSITY CONTROLS ==========
    # OPTION 1: Set vmax directly (recommended for consistent colors)
    vmax = 0.30  # Set this to your desired max AOD value (e.g., 0.10, 0.15, 0.20)
    
    # OPTION 2: Use percentile-based vmax (comment out OPTION 1 and uncomment below)
    # percentile_val = 85  # Lower = darker colors
    # vmax = np.percentile(np.concatenate([v.flatten() for v in all_values]), percentile_val) if all_values else 0.1
    # ===============================================
    
    vmin = 0
    cmap = 'YlOrBr'
    print(f"  → Using vmax={vmax:.4f} for color scaling")
    
    # Create figure
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()} if CARTOPY_AVAILABLE else {}
    )
    
    if n_cols == 1:
        axes = [axes]
    
    # Initialize plots
    images = []
    titles = []
    
    col = 0
    if gt_tensor is not None:
        ax = axes[col]
        if CARTOPY_AVAILABLE:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                        linewidth=0.5, alpha=0.5)
        gt_data = _extract_frame(gt_tensor, sample_idx, 0)
        if CARTOPY_AVAILABLE:
            im = ax.imshow(gt_data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                          origin='upper', transform=ccrs.PlateCarree(), animated=True)
        else:
            im = ax.imshow(gt_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', animated=True)
        images.append(('gt', im, gt_tensor))
        ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        col += 1
    
    for model_name, data in all_data.items():
        if 'predictions' not in data:
            continue
        ax = axes[col]
        if CARTOPY_AVAILABLE:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.coastlines(resolution='50m', color='black', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                        linewidth=0.5, alpha=0.5)
        pred_data = _extract_frame(data['predictions'], sample_idx, 0)
        if CARTOPY_AVAILABLE:
            im = ax.imshow(pred_data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                          origin='upper', transform=ccrs.PlateCarree(), animated=True)
        else:
            im = ax.imshow(pred_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', animated=True)
        images.append((model_name, im, data['predictions']))
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        col += 1
    
    # Add colorbar
    cbar = fig.colorbar(images[0][1], ax=axes, orientation='horizontal', shrink=0.6, pad=0.08)
    cbar.set_label('AOD (550nm)', fontsize=11)
    
    # Main title
    hours_per_frame = 3  # 3-hourly data (24 frames = 72 hours)
    title = fig.suptitle('', fontsize=14, fontweight='bold', y=0.98)
    
    def animate(frame_idx):
        forecast_hour = (frame_idx + 1) * hours_per_frame
        title.set_text(f'Model Comparison - Forecast Hour: h+{forecast_hour}')
        
        for name, im, tensor in images:
            if name == 'gt':
                data = _extract_frame(tensor, sample_idx, frame_idx)
            else:
                data = _extract_frame(tensor, sample_idx, frame_idx)
            if data is not None:
                im.set_array(data)
        
        return [im for _, im, _ in images] + [title]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=1000//fps, blit=False, repeat=True
    )
    
    # Save as MP4
    output_path = output_dir / "comparison_animation.mp4"
    print(f"Creating animation with {n_frames} frames at {fps} FPS...")
    
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='AOD Prediction'), bitrate=1800)
        anim.save(str(output_path), writer=writer, dpi=100)
        print(f"✅ Animation saved: {output_path}")
    except Exception as e:
        print(f"⚠️ Could not save animation (ffmpeg may not be installed): {e}")
        # Save as GIF instead
        output_path_gif = output_dir / "comparison_animation.gif"
        anim.save(str(output_path_gif), writer='pillow', fps=fps, dpi=80)
        print(f"✅ Animation saved as GIF: {output_path_gif}")
    
    plt.close()
    
    return output_path


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Create multi-model comparison visualizations")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to final_viz_config.yaml'
    )
    parser.add_argument(
        '--frame_idx',
        type=int,
        default=None,
        help='Single frame to plot (default: create grid with multiple frames)'
    )
    parser.add_argument(
        '--animation',
        action='store_true',
        help='Create animation instead of static plots'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help='Sample index to visualize'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config.get('output_dir', 'comparison_plots'))
    tensor_dirs = config.get('tensor_dirs', {})
    figsize = tuple(config.get('fig_size', [20, 10]))
    sample_idx = config.get('sample_idx', args.sample_idx)
    
    if not tensor_dirs:
        raise ValueError("No tensor_dirs specified in config!")
    
    # Load all model data
    print("\n" + "="*60)
    print("Loading model predictions...")
    print("="*60)
    all_data = load_all_models_data(tensor_dirs)
    
    if not all_data:
        raise ValueError("No model data could be loaded!")
    
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    
    if args.animation:
        create_animation_comparison(
            all_data=all_data,
            output_dir=output_dir,
            sample_idx=sample_idx,
            figsize=(figsize[0], figsize[1] // 2),
        )
    elif args.frame_idx is not None:
        create_comparison_plot_cartopy(
            all_data=all_data,
            output_dir=output_dir,
            sample_idx=sample_idx,
            frame_idx=args.frame_idx,
            figsize=figsize,
        )
    else:
        # Create grid comparison with multiple frames
        create_comparison_grid(
            all_data=all_data,
            output_dir=output_dir,
            sample_idx=sample_idx,
            frame_indices=[0, 8, 16, 23],  # h+3, h+27, h+51, h+72
            figsize=figsize,
        )
        
        # Also create single frame plots for key horizons
        for fi in [0, 11, 23]:  # h+3, h+36, h+72
            create_comparison_plot_cartopy(
                all_data=all_data,
                output_dir=output_dir,
                sample_idx=sample_idx,
                frame_idx=fi,
                figsize=(figsize[0], figsize[1] // 2),
            )
    
    print(f"\n🎉 All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
