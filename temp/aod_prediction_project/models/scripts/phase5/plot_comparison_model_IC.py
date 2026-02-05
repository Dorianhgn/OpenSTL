"""
Script to plot metrics with confidence intervals (Mean +/- Std) over time.
Comparisons between multiple models are supported on the same plot.

Usage:
    python scripts/phase5/plot_comparison_model_IC.py \
        --config experiments/duaod_final/final_presentation_plots/comparaison_config_IC.yaml

The CSV files used should be the aggregated ones (containing mean/std columns),
typically named 'test_per_frame_metrics.csv'.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import numpy as np

def load_aggregated_csv(csv_path: str) -> pd.DataFrame:
    """
    Load aggregated CSV with mean/std metrics.
    
    Args:
        csv_path: Path to test_per_frame_metrics.csv
    
    Returns:
        DataFrame
    """
    return pd.read_csv(csv_path)

def plot_metric_std(
    models_data: dict,
    metric: str,
    output_dir: Path,
    figsize=(8, 4),
    ylabel=None,
    title=None
):
    """
    Plot metric standard deviation curve for multiple models.
    
    Args:
        models_data: Dict mapping model_name -> DataFrame
        metric: Metric name to plot (e.g. 'ssim')
        output_dir: Directory to save the plot
        figsize: Figure size
        ylabel: Y-axis label
        title: Plot title
    """
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Check if any model has the metric
    valid_models = []
    for model_name, df in models_data.items():
        if f"{metric}_std" in df.columns:
            valid_models.append(model_name)
            
    if not valid_models:
        print(f"⚠️  Metric '{metric}_std' not found in any model. Skipping.")
        return

    # Color palette
    colors = sns.color_palette("deep", n_colors=len(valid_models))
    
    # Plot each model
    for idx, model_name in enumerate(valid_models):
        df = models_data[model_name]
        color = colors[idx]
        
        # Sort by frame_idx just in case
        df = df.sort_values('frame_idx')
        
        x = df['frame_idx']
        y_std = df[f"{metric}_std"]
        
        # Plot std line with filled area
        plt.plot(
            x, y_std, 
            marker='d',  # Diamond marker to distinguish from mean plots
            markersize=5, 
            linewidth=2, 
            label=f"{model_name}", 
            color=color
        )
        
        # Fill area under the curve
        plt.fill_between(
            x, 
            0, 
            y_std, 
            color=color, 
            alpha=0.1
        )

    # Labels and formatting
    if title is None:
        title = f'{metric.upper()} Standard Deviation'
    if ylabel is None:
        ylabel = f'{metric.upper()} Std Dev'
        
    plt.xlabel('Forecast Horizon (Hours)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # X-axis configuration: Convert frames to hours (24 frames = 120 hours -> 5h/frame)
    # We want a tick for every timestep found in the data
    all_frames = pd.concat([df['frame_idx'] for df in models_data.values()]).unique()
    all_frames.sort()
    
    if len(all_frames) > 0:
        # Assuming frame 0 is the first prediction (+5h)
        hours_per_frame = 5
        tick_labels = [(int(f) + 1) * hours_per_frame for f in all_frames]
        plt.xticks(all_frames, tick_labels)
        plt.xlim(all_frames[0], all_frames[-1])

    # Detailed grid configuration
    plt.minorticks_on()
    plt.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
    
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'comparison_std_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot metric Standard Deviation for multiple models")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file'
    )
    # Optional overrides
    parser.add_argument('--metric', type=str, help='Metric to plot (override config)')
    parser.add_argument('--output_dir', type=str, help='Output directory (override config)')
    
    args = parser.parse_args()
    
    print(f"📄 Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Read config
    eval_dirs = config.get('eval_dirs', {})
    if not isinstance(eval_dirs, dict) or not eval_dirs:
        raise ValueError("Config must have 'eval_dirs' dict mapping model names to csv paths")
        
    metric_to_plot = args.metric or config.get('metric')
    if not metric_to_plot:
        raise ValueError("Metric must be specified in config ('metric') or via --metric")
        
    output_dir = args.output_dir or config.get('output_dir', 'comparison_plots_std')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    figsize = tuple(config.get('figsize', [8, 4]))
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Loading {len(eval_dirs)} model(s)...")
    
    models_data = {}
    for model_name, csv_path in eval_dirs.items():
        path = Path(csv_path)
        if not path.exists():
            print(f"⚠️  Warning: {csv_path} not found. Skipping {model_name}.")
            continue
        
        print(f"   Loading {model_name}: {path}")
        models_data[model_name] = load_aggregated_csv(str(path))
        
    if not models_data:
        raise ValueError("No valid CSV files loaded!")
        
    # Plot
    metric_info = {
        'ssim': {'title': 'Structural Similarity Index (Std Dev)'},
        'mse': {'title': 'Mean Squared Error (Std Dev)'},
        'mae': {'title': 'Mean Absolute Error (Std Dev)'},
        'crps': {'title': 'CRPS (Std Dev)'},
        'pod': {'title': 'Probability of Detection (Std Dev)'},
        'far': {'title': 'False Alarm Ratio (Std Dev)'},
        'csi': {'title': 'Critical Success Index (Std Dev)'},
    }
    
    info = metric_info.get(metric_to_plot, {})
    
    print(f"🎨 Generating plot for metric: {metric_to_plot}")
    plot_metric_std(
        models_data=models_data,
        metric=metric_to_plot,
        output_dir=output_dir,
        figsize=figsize,
        ylabel=f'{metric_to_plot.upper()} Std Dev',
        title=info.get('title', f'{metric_to_plot.upper()} Std Dev per Frame')
    )

if __name__ == '__main__':
    main()
