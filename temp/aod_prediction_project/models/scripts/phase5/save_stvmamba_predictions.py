#!/usr/bin/env python3
"""
STVMamba Fusion - Save Full Test Set Predictions

This script loads an STVMamba model from a checkpoint and saves the full test set
predictions as tensors. Since STVMamba is deterministic and fast, we save the entire
test set predictions.

Usage:
    python scripts/phase5/save_stvmamba_predictions.py \
        --config experiments/duaod_final/stvmamba_fusion/4.5M/version_0_full_test/viz_config.yaml

The viz_config.yaml should contain:
    config: path/to/training/config.yaml
    ckpt_path: path/to/checkpoint.ckpt
    batch_idx: 0  # which batch to visualize (optional, if you want single sample only)
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# Import PyTorch Lightning and project modules
import lightning.pytorch as pl


def load_model_and_datamodule(config_path: str, ckpt_path: str):
    """
    Load the STVMamba model and DataModule from config.
    
    Args:
        config_path: Path to training config.yaml
        ckpt_path: Path to model checkpoint
    
    Returns:
        model: Loaded STVMamba model
        datamodule: Configured DataModule
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model class and init args
    model_config = config['model']
    model_class_path = model_config['class_path']
    model_init_args = model_config.get('init_args', {})
    
    # Dynamically import model class
    module_path, class_name = model_class_path.rsplit('.', 1)
    import importlib
    module = importlib.import_module(module_path)
    ModelClass = getattr(module, class_name)
    
    # Load model from checkpoint
    print(f"Loading model from: {ckpt_path}")
    model = ModelClass.load_from_checkpoint(ckpt_path, **model_init_args)
    model.eval()
    
    # Get data class and init args
    data_config = config['data']
    try:
        data_class_path = data_config['class_path']
    except KeyError:
        data_class_path = "datasets.weather_datamodule.WeatherDataModule"
    data_init_args = data_config.get('init_args', {})
    
    # Dynamically import data class
    data_module_path, data_class_name = data_class_path.rsplit('.', 1)
    data_module = importlib.import_module(data_module_path)
    DataClass = getattr(data_module, data_class_name)
    
    # Create DataModule
    datamodule = DataClass(**data_init_args)
    datamodule.setup(stage='test')
    
    return model, datamodule


def save_predictions(
    model,
    datamodule,
    output_dir: Path,
    batch_idx: Optional[int] = None,
    device: str = 'cuda'
):
    """
    Run inference on test set and save predictions.
    
    Args:
        model: STVMamba model
        datamodule: DataModule with test set
        output_dir: Directory to save tensors
        batch_idx: If specified, only save predictions for this batch index
        device: Device to run inference on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    test_loader = datamodule.test_dataloader()
    
    all_predictions = []
    all_ground_truths = []
    all_inputs = []
    
    print(f"\n🚀 Running inference on test set...")
    print(f"   Total batches: {len(test_loader)}")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            # STVMamba expects (X, y) format
            x, y = batch  # x: (B, T_in, C_in, H, W), y: (B, T_out, C_out, H, W)
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            y_pred = model(x)  # (B, T_out, C_out, H, W)
            
            # If batch_idx specified, only save that batch
            if batch_idx is not None:
                if idx == batch_idx:
                    all_predictions.append(y_pred.cpu())
                    all_ground_truths.append(y.cpu())
                    all_inputs.append(x.cpu())
                    break
            else:
                all_predictions.append(y_pred.cpu())
                all_ground_truths.append(y.cpu())
                all_inputs.append(x.cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)  # (N, T_out, C_out, H, W)
    ground_truths = torch.cat(all_ground_truths, dim=0)  # (N, T_out, C_out, H, W)
    inputs = torch.cat(all_inputs, dim=0)  # (N, T_in, C_in, H, W)
    
    # Save tensors
    suffix = f"_batch{batch_idx}" if batch_idx is not None else "_full"
    
    pred_path = output_dir / f"predictions{suffix}.pt"
    gt_path = output_dir / f"ground_truths{suffix}.pt"
    inputs_path = output_dir / f"inputs{suffix}.pt"
    
    torch.save(predictions, pred_path)
    torch.save(ground_truths, gt_path)
    torch.save(inputs, inputs_path)
    
    print(f"\n✅ Predictions saved to: {output_dir}")
    print(f"   - Predictions shape: {predictions.shape}")
    print(f"   - Ground truths shape: {ground_truths.shape}")
    print(f"   - Inputs shape: {inputs.shape}")
    print(f"   - Files: {pred_path.name}, {gt_path.name}, {inputs_path.name}")
    
    # Save metadata
    metadata = {
        "predictions_shape": list(predictions.shape),
        "ground_truths_shape": list(ground_truths.shape),
        "inputs_shape": list(inputs.shape),
        "batch_idx": batch_idx,
        "num_samples": predictions.shape[0],
        "forecast_horizon": predictions.shape[1],
    }
    
    metadata_path = output_dir / f"metadata{suffix}.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    
    return predictions, ground_truths, inputs


def main():
    parser = argparse.ArgumentParser(description="Save STVMamba predictions for test set")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to viz_config.yaml'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Save full test set (override batch_idx in config)'
    )
    
    args = parser.parse_args()
    
    # Load viz config
    with open(args.config, 'r') as f:
        viz_config = yaml.safe_load(f)
    
    config_path = viz_config['config']
    ckpt_path = viz_config['ckpt_path']
    batch_idx = None if args.full else viz_config.get('batch_idx', None)
    
    # Determine output directory (same as config directory)
    config_dir = Path(args.config).parent
    output_dir = config_dir / "tensors"
    
    # Load model and datamodule
    model, datamodule = load_model_and_datamodule(config_path, ckpt_path)
    
    # Save predictions
    save_predictions(
        model=model,
        datamodule=datamodule,
        output_dir=output_dir,
        batch_idx=batch_idx,
        device=args.device,
    )
    
    print(f"\n🎉 Done! Tensors saved to: {output_dir}")


if __name__ == '__main__':
    main()
