import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import os

def load_video_data(save_dir: str):
    """
    Load preds.npy and trues.npy from the specified OpenSTL save directory.
    Returns:
        preds: ndarray of shape (N, T, C, H, W)
        trues: ndarray of shape (N, T, C, H, W) (should be identical across models for same test set)
    """
    preds_path = os.path.join(save_dir, 'preds.npy')
    trues_path = os.path.join(save_dir, 'trues.npy')
    
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Missing predictions at {preds_path}")
        
    preds = np.load(preds_path)
    trues = np.load(trues_path) if os.path.exists(trues_path) else None
    
    return preds, trues

def denormalize(tensor):
    """Simple min-max denormalization if needed, else returns as is."""
    tensor = tensor.astype(float)
    if tensor.max() <= 1.05 and tensor.min() >= -0.05:
        return np.clip(tensor, 0, 1)
    
    # Scale to 0-1
    tensor = tensor - tensor.min()
    if tensor.max() > 0:
        tensor = tensor / tensor.max()
    return tensor

def create_comparison_grid(
    models_data: dict,
    ground_truth: np.ndarray,
    output_dir: Path,
    sample_idx: int = 0,
    frame_indices: list = None,
    figsize=(15, 10)
):
    """
    Plots a multi-row visual comparison.
    Row 1: Ground Truth
    Row 2: Model 1
    ...
    Cols: Different time steps (frame_indices)
    """
    if frame_indices is None:
        T = ground_truth.shape[1]
        frame_indices = list(range(T))
        
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_names = list(models_data.keys())
    n_rows = 1 + len(model_names) # 1 for GS + 1 for each model
    n_cols = len(frame_indices)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
        
    # Helper to plot a specific frame onto an axis
    def _plot_frame(ax, data_sequence, f_idx, title=None, ylabel=None):
        if f_idx >= data_sequence.shape[1]:
            ax.axis('off')
            return
            
        img = data_sequence[sample_idx, f_idx]
        img = denormalize(img)
        
        # If shape is (C, H, W), convert to (H, W, C)
        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0] # Grayscale
                ax.imshow(img, cmap='gray')
            else:
                img = np.transpose(img, (1, 2, 0)) # RGB
                ax.imshow(img)
        elif img.ndim == 2:
            ax.imshow(img, cmap='gray')
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        if title:
            ax.set_title(title, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, rotation=0, labelpad=40, ha='right', va='center')
            
    # Row 0: Ground Truth
    for c, f_idx in enumerate(frame_indices):
        title = f"t = {f_idx + 1}" if c > 0 else f"Ground Truth\n(t = {f_idx + 1})"
        ylabel = "Ground Truth" if c == 0 else None
        _plot_frame(axes[0, c], ground_truth, f_idx, title=title, ylabel=ylabel)
        
    # Rows 1 to N: Models
    for r, model_name in enumerate(model_names, 1):
        preds = models_data[model_name]
        for c, f_idx in enumerate(frame_indices):
            title = f"{model_name}" if c == 0 else None
            ylabel = model_name if c == 0 else None
            _plot_frame(axes[r, c], preds, f_idx, title=title, ylabel=ylabel)
            
    plt.tight_layout()
    output_path = output_dir / f"comparison_grid_sample{sample_idx}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison grid: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate visual prediction comparisons across models from .npy files")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file with model mapping and output_dir'
    )
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    models_dict = config.get('models', {})
    if not models_dict:
        raise ValueError("Config must have a 'models' dictionary mapping model names to paths of 'saved/' directories")
        
    output_dir = Path(config.get('output_dir', 'comparison_plots_visuals'))
    figsize = tuple(config.get('figsize', [20, 10]))
    sample_idx = config.get('sample_idx', 0)
    
    # Load all numpy files
    print(f"📊 Loading predictions from {len(models_dict)} model(s)...")
    models_preds = {}
    ground_truth = None
    
    for model_name, save_dir in models_dict.items():
        if not os.path.exists(save_dir):
            print(f"⚠️  Warning: {save_dir} not found. Skipping {model_name}.")
            continue
            
        print(f"   Loading {model_name} from {save_dir}")
        try:
            preds, trues = load_video_data(save_dir)
            models_preds[model_name] = preds
            if ground_truth is None and trues is not None:
                ground_truth = trues
        except Exception as e:
            print(f"   Error: {e}")
            
    if not models_preds or ground_truth is None:
        raise ValueError("Failed to load sufficient data or ground truth.")
        
    T = ground_truth.shape[1]
    
    # Either parse frame_indices or default to [0, T//4, T//2, 3T//4, T-1]
    frame_indices = config.get('frame_indices', None)
    if frame_indices is None or frame_indices == 'all':
        frame_indices = list(range(T))
    elif isinstance(frame_indices, int):
        frame_indices = [frame_indices]
        
    print(f"🎨 Generating visual comparison grid for sample {sample_idx} over frames {frame_indices}...")
    create_comparison_grid(
        models_data=models_preds,
        ground_truth=ground_truth,
        output_dir=output_dir,
        sample_idx=sample_idx,
        frame_indices=frame_indices,
        figsize=figsize
    )
    
if __name__ == '__main__':
    main()
