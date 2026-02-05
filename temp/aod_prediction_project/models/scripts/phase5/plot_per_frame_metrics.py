"""
Utility to plot per-frame metrics from test CSV files.

Usage:
    python scripts/plot_per_frame_metrics.py --csv path/to/test_per_frame_metrics.csv --output paper_plots/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_per_frame_metrics(csv_path: str, output_dir: str = None, show: bool = False):
    """
    Plot per-frame metrics from CSV file.
    
    Args:
        csv_path: Path to the CSV file (test_per_frame_metrics.csv)
        output_dir: Directory to save plots (default: same as CSV)
        show: Whether to show plots interactively
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(csv_path).parent / 'plots'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # ===== Plot 1: SSIM over time =====
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['frame_idx'], df['ssim_mean'], 'o-', label='SSIM', linewidth=2, markersize=6)
    
    # Add error bands if std is available
    if 'ssim_std' in df.columns:
        ax.fill_between(
            df['frame_idx'],
            df['ssim_mean'] - df['ssim_std'],
            df['ssim_mean'] + df['ssim_std'],
            alpha=0.3,
            label='±1 std'
        )
    
    ax.set_xlabel('Frame Index (timestep)', fontsize=12)
    ax.set_ylabel('SSIM', fontsize=12)
    ax.set_title('Structural Similarity Index (SSIM) per Frame', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'ssim_per_frame.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'ssim_per_frame.png'}")
    if show:
        plt.show()
    plt.close()
    
    # ===== Plot 2: MSE and MAE over time =====
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # MSE subplot
    ax1.plot(df['frame_idx'], df['mse_mean'], 'o-', color='red', label='MSE', linewidth=2, markersize=6)
    if 'mse_std' in df.columns:
        ax1.fill_between(
            df['frame_idx'],
            df['mse_mean'] - df['mse_std'],
            df['mse_mean'] + df['mse_std'],
            color='red',
            alpha=0.3,
            label='±1 std'
        )
    ax1.set_xlabel('Frame Index (timestep)', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Mean Squared Error (MSE) per Frame', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # MAE subplot
    ax2.plot(df['frame_idx'], df['mae_mean'], 'o-', color='orange', label='MAE', linewidth=2, markersize=6)
    if 'mae_std' in df.columns:
        ax2.fill_between(
            df['frame_idx'],
            df['mae_mean'] - df['mae_std'],
            df['mae_mean'] + df['mae_std'],
            color='orange',
            alpha=0.3,
            label='±1 std'
        )
    ax2.set_xlabel('Frame Index (timestep)', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Mean Absolute Error (MAE) per Frame', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mse_mae_per_frame.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'mse_mae_per_frame.png'}")
    if show:
        plt.show()
    plt.close()
    
    # ===== Plot 3: Combined view with normalized metrics =====
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Normalize metrics to [0, 1] for comparison
    ssim_norm = df['ssim_mean']
    mse_norm = (df['mse_mean'] - df['mse_mean'].min()) / (df['mse_mean'].max() - df['mse_mean'].min())
    mae_norm = (df['mae_mean'] - df['mae_mean'].min()) / (df['mae_mean'].max() - df['mae_mean'].min())
    
    ax.plot(df['frame_idx'], ssim_norm, 'o-', label='SSIM', linewidth=2, markersize=6, color='green')
    ax.plot(df['frame_idx'], 1 - mse_norm, 's-', label='1 - MSE (normalized)', linewidth=2, markersize=6, color='red', alpha=0.7)
    ax.plot(df['frame_idx'], 1 - mae_norm, '^-', label='1 - MAE (normalized)', linewidth=2, markersize=6, color='orange', alpha=0.7)
    
    ax.set_xlabel('Frame Index (timestep)', fontsize=12)
    ax.set_ylabel('Normalized Metric (higher is better)', fontsize=12)
    ax.set_title('All Metrics per Frame (Normalized)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'all_metrics_normalized.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_dir / 'all_metrics_normalized.png'}")
    if show:
        plt.show()
    plt.close()
    
    # ===== Plot 4: CRPS over time (if available) =====
    if 'crps_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['frame_idx'], df['crps_mean'], 'o-', color='purple', label='CRPS', linewidth=2, markersize=6)
        
        if 'crps_std' in df.columns:
            ax.fill_between(
                df['frame_idx'],
                df['crps_mean'] - df['crps_std'],
                df['crps_mean'] + df['crps_std'],
                color='purple',
                alpha=0.3,
                label='±1 std'
            )
        
        ax.set_xlabel('Frame Index (timestep)', fontsize=12)
        ax.set_ylabel('CRPS', fontsize=12)
        ax.set_title('Continuous Ranked Probability Score (CRPS) per Frame', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'crps_per_frame.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_dir / 'crps_per_frame.png'}")
        if show:
            plt.show()
        plt.close()
    
    # ===== Plot 5: Detection metrics (POD, FAR, CSI) =====
    if 'pod_mean' in df.columns and 'far_mean' in df.columns and 'csi_mean' in df.columns:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # POD subplot
        ax1.plot(df['frame_idx'], df['pod_mean'], 'o-', color='green', label='POD', linewidth=2, markersize=6)
        if 'pod_std' in df.columns:
            ax1.fill_between(
                df['frame_idx'],
                df['pod_mean'] - df['pod_std'],
                df['pod_mean'] + df['pod_std'],
                color='green',
                alpha=0.3,
                label='±1 std'
            )
        ax1.set_xlabel('Frame Index (timestep)', fontsize=12)
        ax1.set_ylabel('POD', fontsize=12)
        ax1.set_title('Probability of Detection (POD)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # FAR subplot
        ax2.plot(df['frame_idx'], df['far_mean'], 'o-', color='red', label='FAR', linewidth=2, markersize=6)
        if 'far_std' in df.columns:
            ax2.fill_between(
                df['frame_idx'],
                df['far_mean'] - df['far_std'],
                df['far_mean'] + df['far_std'],
                color='red',
                alpha=0.3,
                label='±1 std'
            )
        ax2.set_xlabel('Frame Index (timestep)', fontsize=12)
        ax2.set_ylabel('FAR', fontsize=12)
        ax2.set_title('False Alarm Ratio (FAR)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # CSI subplot
        ax3.plot(df['frame_idx'], df['csi_mean'], 'o-', color='blue', label='CSI', linewidth=2, markersize=6)
        if 'csi_std' in df.columns:
            ax3.fill_between(
                df['frame_idx'],
                df['csi_mean'] - df['csi_std'],
                df['csi_mean'] + df['csi_std'],
                color='blue',
                alpha=0.3,
                label='±1 std'
            )
        ax3.set_xlabel('Frame Index (timestep)', fontsize=12)
        ax3.set_ylabel('CSI', fontsize=12)
        ax3.set_title('Critical Success Index (CSI)', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_metrics_per_frame.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_dir / 'detection_metrics_per_frame.png'}")
        if show:
            plt.show()
        plt.close()
    
    # ===== Print summary statistics =====
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nSSIM:")
    print(f"  Mean (across all frames): {df['ssim_mean'].mean():.4f}")
    print(f"  Start frame (0): {df['ssim_mean'].iloc[0]:.4f}")
    print(f"  Mid frame ({len(df)//2}): {df['ssim_mean'].iloc[len(df)//2]:.4f}")
    print(f"  End frame ({len(df)-1}): {df['ssim_mean'].iloc[-1]:.4f}")
    
    print(f"\nMSE:")
    print(f"  Mean (across all frames): {df['mse_mean'].mean():.6f}")
    print(f"  Start frame (0): {df['mse_mean'].iloc[0]:.6f}")
    print(f"  End frame ({len(df)-1}): {df['mse_mean'].iloc[-1]:.6f}")
    
    print(f"\nMAE:")
    print(f"  Mean (across all frames): {df['mae_mean'].mean():.6f}")
    print(f"  Start frame (0): {df['mae_mean'].iloc[0]:.6f}")
    print(f"  End frame ({len(df)-1}): {df['mae_mean'].iloc[-1]:.6f}")
    
    if 'crps_mean' in df.columns:
        print(f"\nCRPS:")
        print(f"  Mean (across all frames): {df['crps_mean'].mean():.6f}")
        print(f"  Start frame (0): {df['crps_mean'].iloc[0]:.6f}")
        print(f"  End frame ({len(df)-1}): {df['crps_mean'].iloc[-1]:.6f}")
    
    if 'pod_mean' in df.columns:
        print(f"\nDetection Metrics:")
        print(f"  POD (mean): {df['pod_mean'].mean():.4f}")
        print(f"  FAR (mean): {df['far_mean'].mean():.4f}")
        print(f"  CSI (mean): {df['csi_mean'].mean():.4f}")
    
    print("="*60)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Plot per-frame metrics from test CSV')
    parser.add_argument('--csv', type=str, required=True, help='Path to test_per_frame_metrics.csv')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    plot_per_frame_metrics(args.csv, args.output, args.show)


if __name__ == '__main__':
    main()
