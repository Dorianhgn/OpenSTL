import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import os

def load_metrics_data(npy_path: str) -> dict:
    """
    Load per_frame_metrics.npy which contains dict like {'mse': array, 'mse_std': array}.
    """
    return np.load(npy_path, allow_pickle=True).item()

def plot_metric_comparison(
    models_data: dict,
    metric: str,
    output_dir: Path,
    figsize=(12, 6),
    ylabel=None,
    title=None,
    plot_std=False
):
    """
    Plot comparison for a single metric across all models.
    
    Args:
        models_data: Dict mapping model_name -> dict of metrics
        metric: Metric name to plot (e.g., 'ssim', 'mse')
        output_dir: Directory to save the plot
        figsize: Figure size
        ylabel: Y-axis label (default: metric name)
        title: Plot title (default: auto-generated)
        plot_std: Boolean, whether to plot confidence intervals (std)
    """
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Check which models have this metric
    models_with_metric = {name: data for name, data in models_data.items() if metric in data}
    
    if not models_with_metric:
        print(f"⚠️  Metric '{metric}' not found in any model. Skipping.")
        return
        
    colors = sns.color_palette("deep", n_colors=len(models_with_metric))
    
    max_frames = 0
    
    # Plot each model
    for idx, (model_name, data) in enumerate(models_with_metric.items()):
        y_mean = data[metric]  # Array of shape (T,)
        x = np.arange(len(y_mean))
        max_frames = max(max_frames, len(y_mean))
        color = colors[idx]
        
        plt.plot(
            x, y_mean, 
            marker='o', 
            markersize=5, 
            linewidth=2, 
            label=model_name, 
            color=color
        )
        
        if plot_std and f"{metric}_std" in data:
            y_std = data[f"{metric}_std"]
            plt.fill_between(
                x, 
                y_mean - y_std, 
                y_mean + y_std, 
                color=color, 
                alpha=0.2
            )

    # Labels and formatting
    if title is None:
        std_suffix = " (with Std Dev)" if plot_std else ""
        title = f'{metric.upper()} per Frame Comparison{std_suffix}'
    if ylabel is None:
        ylabel = metric.upper()
        
    plt.xlabel('Forecast Horizon (Frames)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    
    # X-axis configuration
    plt.xticks(np.arange(max_frames), np.arange(1, max_frames + 1))
    plt.xlim(0, max_frames - 1)
        
    # Detailed grid configuration
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    std_prefix = "_with_std" if plot_std else ""
    output_path = output_dir / f'comparison_{metric}{std_prefix}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare per-frame metrics across multiple models from .npy files")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file with model mapping and output_dir'
    )
    
    args = parser.parse_args()
    
    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    models_dict = config.get('models', {})
    if not models_dict:
        raise ValueError("Config must have a 'models' dictionary mapping model names to paths of per_frame_metrics.npy")
        
    output_dir = Path(config.get('output_dir', 'comparison_plots_metrics'))
    figsize = tuple(config.get('figsize', [12, 6]))
    plot_std = config.get('plot_std', True)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load all numpy files
    print(f"\n📊 Loading {len(models_dict)} model(s)...")
    models_data = {}
    for model_name, npy_path in models_dict.items():
        if not os.path.exists(npy_path):
            print(f"⚠️  Warning: {npy_path} not found. Skipping {model_name}.")
            continue
        print(f"   Loading {model_name}: {npy_path}")
        models_data[model_name] = load_metrics_data(npy_path)
        
    if not models_data:
        raise ValueError("No valid .npy files loaded!")
        
    # Determine which metrics are available across all models
    all_metrics = set()
    for data in models_data.values():
        for k in data.keys():
            if not k.endswith('_std'):
                all_metrics.add(k)
                
    print(f"\n📈 Found metrics: {', '.join(sorted(all_metrics))}")
    
    # Define metric properties
    metric_info = {
        'ssim': {'ylabel': 'SSIM (higher is better)', 'title': 'Structural Similarity Index'},
        'mse': {'ylabel': 'MSE (lower is better)', 'title': 'Mean Squared Error'},
        'mae': {'ylabel': 'MAE (lower is better)', 'title': 'Mean Absolute Error'},
        'psnr': {'ylabel': 'PSNR (higher is better)', 'title': 'Peak Signal-to-Noise Ratio'},
        'pod': {'ylabel': 'POD (higher is better)', 'title': 'Probability of Detection'},
        'far': {'ylabel': 'FAR (lower is better)', 'title': 'False Alarm Ratio'},
        'csi': {'ylabel': 'CSI (higher is better)', 'title': 'Critical Success Index'},
    }
    
    # Generate comparison plots
    print(f"\n🎨 Generating comparison plots...")
    for metric in sorted(all_metrics):
        info = metric_info.get(metric, {})
        plot_metric_comparison(
            models_data=models_data,
            metric=metric,
            output_dir=output_dir,
            figsize=figsize,
            ylabel=info.get('ylabel'),
            title=info.get('title'),
            plot_std=plot_std
        )
        
    print(f"\n{'='*80}")
    print("Done! All metric comparisons plotted.")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
