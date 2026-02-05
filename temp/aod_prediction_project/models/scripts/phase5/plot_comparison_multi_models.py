"""
Multi-model comparison script for per-frame metrics.

Compares multiple models (deterministic and ensemble) by loading their
raw CSV files and generating comparison plots for each metric.

Usage:
    python scripts/phase5/plot_comparison_multi_models.py \
        --csv_files model1=path/to/test_per_frame_metrics_raw.csv \
                    model2=path/to/other_raw.csv \
        --output_dir comparison_plots/
        
Or with a config file:
    python scripts/phase5/plot_comparison_multi_models.py \
        --config experiments/duaod_final/final_presentation_plots/comparison_config.yaml
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV and compute mean metrics per frame_idx.
    
    Args:
        csv_path: Path to test_per_frame_metrics_raw.csv
    
    Returns:
        DataFrame with frame_idx and mean metrics
    """
    df = pd.read_csv(csv_path)
    
    # Group by frame_idx and compute mean across all batches
    metrics_cols = [col for col in df.columns if col not in ['frame_idx', 'batch_idx']]
    df_mean = df.groupby('frame_idx')[metrics_cols].mean().reset_index()
    
    return df_mean


def plot_metric_comparison(
    models_data: dict,
    metric: str,
    output_dir: Path,
    figsize=(12, 6),
    ylabel=None,
    title=None
):
    """
    Plot comparison for a single metric across all models.
    
    Args:
        models_data: Dict mapping model_name -> DataFrame
        metric: Metric name to plot
        output_dir: Directory to save the plot
        figsize: Figure size
        ylabel: Y-axis label (default: metric name)
        title: Plot title (default: auto-generated)
    """
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Check which models have this metric
    models_with_metric = {name: df for name, df in models_data.items() if metric in df.columns}
    
    if not models_with_metric:
        print(f"⚠️  Metric '{metric}' not found in any model. Skipping.")
        return
    
    # Plot each model
    for model_name, df in models_with_metric.items():
        plt.plot(
            df['frame_idx'],
            df[metric],
            marker='o',
            linestyle='-',
            label=model_name,
            linewidth=2,
            markersize=6
        )
    
    # Labels and formatting
    if title is None:
        title = f'{metric.upper()} Comparison Across Models'
    if ylabel is None:
        ylabel = metric.upper()
    
    plt.xlabel('Forecast Horizon (Hours)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    
    # X-axis configuration: Convert frames to hours (24 frames = 120 hours -> 5h/frame)
    # We want a tick for every timestep
    all_frames = pd.concat([df['frame_idx'] for df in models_with_metric.values()]).unique()
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
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'comparison_{metric}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare per-frame metrics across multiple models")
    parser.add_argument(
        '--csv_files',
        nargs='+',
        metavar='NAME=PATH',
        help='Model CSV files in format: model_name=path/to/csv (e.g., stvmamba=path.csv jvm=path2.csv)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='comparison_plots',
        help='Directory to save comparison plots'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML config file with csv_files and output_dir'
    )
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[12, 6],
        help='Figure size (width height)'
    )
    
    args = parser.parse_args()
    
    # Load config from YAML if provided
    csv_files_dict = {}
    output_dir = args.output_dir
    figsize = tuple(args.figsize)
    
    if args.config:
        print(f"📄 Loading config from: {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Parse eval_dirs from config
        if 'eval_dirs' in config:
            eval_dirs = config['eval_dirs']
            if isinstance(eval_dirs, dict):
                csv_files_dict = eval_dirs
            else:
                raise ValueError("eval_dirs in config must be a dictionary")
        
        if 'output_dir' in config:
            output_dir = config['output_dir']
        
        if 'figsize' in config:
            figsize = tuple(config['figsize'])
    
    # Parse CSV files from command line (overrides config if both provided)
    if args.csv_files:
        for item in args.csv_files:
            if '=' not in item:
                raise ValueError(f"Invalid format: {item}. Use format: name=path")
            name, path = item.split('=', 1)
            csv_files_dict[name] = path
    
    if not csv_files_dict:
        raise ValueError("No CSV files provided. Use --csv_files or --config")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load all CSV files
    print(f"\n📊 Loading {len(csv_files_dict)} model(s)...")
    models_data = {}
    for model_name, csv_path in csv_files_dict.items():
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"⚠️  Warning: {csv_path} not found. Skipping {model_name}.")
            continue
        
        print(f"   Loading {model_name}: {csv_path}")
        models_data[model_name] = load_csv_data(str(csv_path))
    
    if not models_data:
        raise ValueError("No valid CSV files loaded!")
    
    # Determine which metrics are available across all models
    all_metrics = set()
    for df in models_data.values():
        all_metrics.update([col for col in df.columns if col != 'frame_idx'])
    
    print(f"\n📈 Found metrics: {', '.join(sorted(all_metrics))}")
    
    # Define metric properties (for better labels)
    metric_info = {
        'ssim': {'ylabel': 'SSIM (higher is better)', 'title': 'Structural Similarity Index'},
        'mse': {'ylabel': 'MSE (lower is better)', 'title': 'Mean Squared Error'},
        'mae': {'ylabel': 'MAE (lower is better)', 'title': 'Mean Absolute Error'},
        'crps': {'ylabel': 'CRPS (lower is better)', 'title': 'Continuous Ranked Probability Score'},
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
            title=info.get('title')
        )
    
    # Generate summary table
    print(f"\n📋 Generating summary table...")
    summary_data = []
    for model_name, df in models_data.items():
        row = {'model': model_name}
        for metric in sorted(all_metrics):
            if metric in df.columns:
                # Compute average across all timesteps
                row[f'{metric}_mean'] = df[metric].mean()
                row[f'{metric}_std'] = df[metric].std()
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'summary_table.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Saved summary table: {summary_path}")
    
    # Print summary to console
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS (Mean ± Std across all timesteps)")
    print(f"{'='*80}")
    for _, row in summary_df.iterrows():
        print(f"\n{row['model']}:")
        for metric in sorted(all_metrics):
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col in row:
                print(f"  {metric.upper():8s}: {row[mean_col]:.6f} ± {row[std_col]:.6f}")
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
