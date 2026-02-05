"""
Utility for saving per-frame metrics during model evaluation.

Provides a unified interface for deterministic models (SimVP, STVMamba, etc.)
to save frame-by-frame metrics in the same format as generative models.
Compatible with scripts/phase5/plot_per_frame_metrics.py.
"""
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


def save_per_frame_metrics_to_csv(
    frame_metrics: List[Dict[str, float]],
    logger_log_dir: Optional[str],
    output_filename: str = "test_per_frame_metrics.csv",
    verbose: bool = True
):
    """
    Save per-frame metrics to CSV files (aggregated and raw).
    
    This function mimics the behavior of JvMBaseLightningModule.on_test_epoch_end()
    with test_method="full", creating CSV files that are compatible with
    scripts/phase5/plot_per_frame_metrics.py.
    
    Args:
        frame_metrics: List of dicts with keys like:
            - frame_idx: int (timestep index)
            - batch_idx: int (batch number)
            - ssim: float
            - mse: float
            - mae: float
            - (optional) crps, pod, far, csi for probabilistic models
        logger_log_dir: Directory from self.logger.log_dir (can be None)
        output_filename: Name for the aggregated CSV file
        verbose: Whether to print status messages
    
    Example:
        >>> frame_metrics = []
        >>> for batch_idx, batch in enumerate(test_dataloader):
        ...     y_pred = model(batch[0])
        ...     y_true = batch[1]
        ...     for t in range(forecast_horizon):
        ...         ssim = compute_ssim(y_pred[:, t], y_true[:, t])
        ...         mse = torch.mean((y_pred[:, t] - y_true[:, t]) ** 2).item()
        ...         mae = torch.mean(torch.abs(y_pred[:, t] - y_true[:, t])).item()
        ...         frame_metrics.append({
        ...             'frame_idx': t,
        ...             'batch_idx': batch_idx,
        ...             'ssim': ssim,
        ...             'mse': mse,
        ...             'mae': mae,
        ...         })
        >>> save_per_frame_metrics_to_csv(frame_metrics, trainer.logger.log_dir)
    """
    if not frame_metrics:
        if verbose:
            print("⚠️ No frame metrics to save.")
        return
    
    # Create DataFrame from accumulated metrics
    df = pd.DataFrame(frame_metrics)
    
    # Determine which metrics are available
    available_metrics = [col for col in df.columns if col not in ['frame_idx', 'batch_idx']]
    
    # Group by frame_idx and compute statistics across all batches
    agg_dict = {}
    for metric in available_metrics:
        agg_dict[metric] = ['mean', 'std', 'min', 'max']
    
    df_stats = df.groupby('frame_idx').agg(agg_dict).reset_index()
    
    # Flatten column names (e.g., 'ssim_mean', 'ssim_std')
    df_stats.columns = ['_'.join(col).strip('_') for col in df_stats.columns.values]
    
    # Determine output directory
    output_dir = Path(logger_log_dir) if logger_log_dir is not None else Path('.')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save aggregated stats
    output_path = output_dir / output_filename
    df_stats.to_csv(output_path, index=False)
    
    if verbose:
        print(f"\n✅ Per-frame metrics saved to: {output_path}")
        print(f"   Metrics: {', '.join(available_metrics)}")
        print(f"   Total frames: {len(df_stats)}, Total batches: {df['batch_idx'].nunique()}")
    
    # Save raw data (all batches)
    raw_filename = output_filename.replace('.csv', '_raw.csv')
    output_path_raw = output_dir / raw_filename
    df.to_csv(output_path_raw, index=False)
    
    if verbose:
        print(f"✅ Raw per-frame metrics saved to: {output_path_raw}")
