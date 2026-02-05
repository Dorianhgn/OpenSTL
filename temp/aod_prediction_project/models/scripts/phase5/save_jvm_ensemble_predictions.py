#!/usr/bin/env python3
"""
JvM/CFM - Save Ensemble Predictions for Single Sample

This script loads a SPADEJvM or SPADECfM model and saves ensemble predictions
for a single sample. The model solves the ODE n_ens times with different noise
initializations to generate ensemble predictions.

Both JvM (prediction_mode="x") and CFM (prediction_mode="v") use the same
SPADEJvMLightningModule, only differing in the prediction_mode parameter.

Usage:
    python scripts/phase5/save_jvm_ensemble_predictions.py \
        --config experiments/duaod_final/spadeJvM/small/version_1_test_full_ep27_n-ens5/viz_config.yaml

The viz_config.yaml should contain:
    config: path/to/training/config.yaml
    ckpt_path: path/to/checkpoint.ckpt
    n_ens: 5  # number of ensemble predictions
    batch_idx: 0  # which batch index (default: 0)
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
    Load the JvM/CFM model and DataModule from config.
    
    Args:
        config_path: Path to training config.yaml
        ckpt_path: Path to model checkpoint
    
    Returns:
        model: Loaded SPADEJvM model
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
    data_class_path = data_config['class_path']
    data_init_args = data_config.get('init_args', {})
    
    # Dynamically import data class
    data_module_path, data_class_name = data_class_path.rsplit('.', 1)
    data_module = importlib.import_module(data_module_path)
    DataClass = getattr(data_module, data_class_name)
    
    # Create DataModule
    datamodule = DataClass(**data_init_args)
    datamodule.setup(stage='test')
    
    return model, datamodule, config


def save_ensemble_predictions(
    model,
    datamodule,
    output_dir: Path,
    n_ens: int = 5,
    batch_idx: int = 0,
    sample_idx: int = 0,
    num_steps: int = None,
    solver: str = None,
    device: str = 'cuda'
):
    """
    Run ensemble inference for a single sample and save predictions.
    
    Args:
        model: SPADEJvM model
        datamodule: DataModule with test set
        output_dir: Directory to save tensors (will create ens{n}/ subdirectory)
        n_ens: Number of ensemble predictions
        batch_idx: Which batch to use
        sample_idx: Which sample in the batch to use
        num_steps: Number of ODE steps (default: from model hparams)
        solver: ODE solver (default: from model hparams)
        device: Device to run inference on
    """
    # Create subdirectory for this ensemble count
    output_dir = Path(output_dir) / f"ens{n_ens}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    # Get the specific batch
    test_loader = datamodule.test_dataloader()
    
    target_batch = None
    for idx, batch in enumerate(test_loader):
        if idx == batch_idx:
            target_batch = batch
            break
    
    if target_batch is None:
        raise ValueError(f"Batch index {batch_idx} not found. Test loader has {len(test_loader)} batches.")
    
    # Extract data from batch (JvM format: dict with x_past, x_future, cond)
    if isinstance(target_batch, dict):
        x_past = target_batch["x_past"].to(device)      # (B, T_in, C_aod, H, W)
        x_future = target_batch["x_future"].to(device)  # (B, T_out, C_aod, H, W)
        cond = target_batch["cond"].to(device)          # (B, T_in, C_raw, H, W)
    else:
        raise ValueError(f"Expected dict batch format, got {type(target_batch)}")
    
    # Select single sample
    x_past_sample = x_past[sample_idx:sample_idx+1]      # (1, T_in, C_aod, H, W)
    x_future_sample = x_future[sample_idx:sample_idx+1]  # (1, T_out, C_aod, H, W)
    cond_sample = cond[sample_idx:sample_idx+1]          # (1, T_in, C_raw, H, W)
    
    # Use model's default num_steps and solver if not specified
    if num_steps is None:
        num_steps = model.hparams.num_inference_steps
    if solver is None:
        solver = model.hparams.ode_solver
    
    # Target shape for sampling
    target_shape = x_future_sample.shape  # (1, T_out, C_aod, H, W)
    
    print(f"\n🚀 Running ensemble inference...")
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Prediction mode: {model.hparams.prediction_mode}")
    print(f"   Number of ensembles: {n_ens}")
    print(f"   ODE steps: {num_steps}, Solver: {solver}")
    print(f"   Sample shape: cond={cond_sample.shape}, x_past={x_past_sample.shape}")
    print(f"   Target shape: {target_shape}")
    
    ensemble_predictions = []
    
    with torch.no_grad():
        for ens_idx in tqdm(range(n_ens), desc="Ensemble sampling"):
            # Each call to sample() starts from different random noise
            # The sample() method handles the ODE integration internally
            pred = model.sample(
                x_cond_raw=cond_sample,
                x_past=x_past_sample,
                target_shape=target_shape,
                num_steps=num_steps,
                solver=solver,
                return_trajectory=False,  # Only return final prediction
            )
            ensemble_predictions.append(pred.cpu())
    
    # Stack ensemble predictions: (n_ens, 1, T_out, C_aod, H, W) -> (n_ens, T_out, C_aod, H, W)
    ensemble_preds = torch.cat(ensemble_predictions, dim=0)  # (n_ens, T_out, C_aod, H, W)
    
    # Compute ensemble mean
    ensemble_mean = ensemble_preds.mean(dim=0, keepdim=True)  # (1, T_out, C_aod, H, W)
    
    # Save tensors
    suffix = f"_batch{batch_idx}_sample{sample_idx}_ens{n_ens}"
    
    pred_path = output_dir / f"ensemble_predictions{suffix}.pt"
    mean_path = output_dir / f"ensemble_mean{suffix}.pt"
    gt_path = output_dir / f"ground_truth{suffix}.pt"
    cond_path = output_dir / f"condition{suffix}.pt"
    x_past_path = output_dir / f"x_past{suffix}.pt"
    
    torch.save(ensemble_preds, pred_path)
    torch.save(ensemble_mean.squeeze(0), mean_path)
    torch.save(x_future_sample.squeeze(0).cpu(), gt_path)
    torch.save(cond_sample.squeeze(0).cpu(), cond_path)
    torch.save(x_past_sample.squeeze(0).cpu(), x_past_path)
    
    print(f"\n✅ Ensemble predictions saved to: {output_dir}")
    print(f"   - Ensemble predictions shape: {ensemble_preds.shape}")
    print(f"   - Ensemble mean shape: {ensemble_mean.squeeze(0).shape}")
    print(f"   - Ground truth shape: {x_future_sample.squeeze(0).shape}")
    print(f"   - Files saved:")
    print(f"     - {pred_path.name}")
    print(f"     - {mean_path.name}")
    print(f"     - {gt_path.name}")
    print(f"     - {cond_path.name}")
    print(f"     - {x_past_path.name}")
    
    # Save metadata
    metadata = {
        "ensemble_predictions_shape": list(ensemble_preds.shape),
        "ensemble_mean_shape": list(ensemble_mean.squeeze(0).shape),
        "ground_truth_shape": list(x_future_sample.squeeze(0).shape),
        "condition_shape": list(cond_sample.squeeze(0).shape),
        "x_past_shape": list(x_past_sample.squeeze(0).shape),
        "n_ens": n_ens,
        "batch_idx": batch_idx,
        "sample_idx": sample_idx,
        "num_steps": num_steps,
        "solver": solver,
        "prediction_mode": model.hparams.prediction_mode,
    }
    
    metadata_path = output_dir / f"metadata{suffix}.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    
    return ensemble_preds, ensemble_mean, x_future_sample.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(description="Save JvM/CFM ensemble predictions")
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
        '--n_ens',
        type=int,
        default=None,
        help='Override n_ens from config'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help='Sample index within the batch (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Load viz config
    with open(args.config, 'r') as f:
        viz_config = yaml.safe_load(f)
    
    config_path = viz_config['config']
    ckpt_path = viz_config['ckpt_path']
    n_ens = args.n_ens if args.n_ens is not None else viz_config.get('n_ens', 5)
    batch_idx = viz_config.get('batch_idx', 0)
    sample_idx = args.sample_idx
    
    # Determine output directory (same as config directory)
    config_dir = Path(args.config).parent
    output_dir = config_dir / "tensors"
    
    # Load model and datamodule
    model, datamodule, config = load_model_and_datamodule(config_path, ckpt_path)
    
    # Save ensemble predictions
    save_ensemble_predictions(
        model=model,
        datamodule=datamodule,
        output_dir=output_dir,
        n_ens=n_ens,
        batch_idx=batch_idx,
        sample_idx=sample_idx,
        device=args.device,
    )
    
    print(f"\n🎉 Done! Tensors saved to: {output_dir}")


if __name__ == '__main__':
    main()
