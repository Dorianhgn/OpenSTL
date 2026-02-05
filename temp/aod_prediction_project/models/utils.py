import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from pathlib import Path
from typing import Callable, Optional

class WarmupCosineLR(SequentialLR):
    def __init__(
        self, 
        optimizer, 
        warmup_epochs: int = 5, 
        max_epochs: int = 300, 
        warmup_start_factor: float = 1e-4, 
        eta_min: float = 1e-6
    ):
        # 1. Création du Warmup
        scheduler1 = LinearLR(
            optimizer, 
            start_factor=warmup_start_factor, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        
        # 2. Création du Cosine pour le reste du temps
        scheduler2 = CosineAnnealingLR(
            optimizer, 
            T_max=max_epochs - warmup_epochs, 
            eta_min=eta_min
        )
        
        # 3. Initialisation du parent
        super().__init__(
            optimizer, 
            schedulers=[scheduler1, scheduler2], 
            milestones=[warmup_epochs]
        )


def visualize_trajectory_evolution(
    sample_fn: Callable,
    batch: dict,
    viz_dir: Path,
    solver_configs: list[tuple[int, str]],
    z1_gt: torch.Tensor,
    viz_sample_noise: Optional[list[Optional[float]]] = None,
) -> None:
    """Unified trajectory evolution visualization for CFM and JvM models.
    
    Creates two types of videos:
    1. Trajectory evolution: Shows ODE integration steps at 3 temporal slices
    2. Final prediction: Shows full temporal sequence of the final prediction
    
    Args:
        sample_fn: Function that takes (num_steps, solver, return_trajectory, x_target, noise_t)
                   and returns trajectory or final prediction
        batch: Batch dictionary containing conditions and targets
        viz_dir: Directory to save visualization videos
        solver_configs: List of (num_steps, solver) tuples to test
        z1_gt: Ground truth tensor (B, T_out, C, H, W)
        viz_sample_noise: List of noise levels for visualization [None, 0.1, 0.5]
                         None = clean sampling, float = noisy starting point at t
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    
    # Setup
    plt.switch_backend('Agg')
    
    # Default: clean sampling only
    if viz_sample_noise is None:
        viz_sample_noise = [None]
    
    # Iterate over noise levels
    for noise_level in viz_sample_noise:
        noise_suffix = f"_noise_{noise_level}" if noise_level is not None else "_clean"
        print(f"\n{'='*60}")
        print(f"Visualization with noise level: {noise_level}")
        print(f"{'='*60}")
        
        # Store final predictions for each solver
        final_preds = {}
        
        # Iterate over solver configs
        for num_steps, solver in solver_configs:
            print(f"\nGenerating trajectory with {num_steps} steps and {solver} solver...")
            
            with torch.no_grad():
                # Get full trajectory: (Steps, B, T, C, H, W)
                trajectory = sample_fn(
                    num_steps=num_steps,
                    solver=solver,
                    return_trajectory=True,
                    x_target=z1_gt if noise_level is not None else None,
                    noise_t=noise_level,
                )
            
            # Store final prediction
            final_preds[f"{num_steps}_{solver}"] = trajectory[-1, 0]  # (T, C, H, W)
            
            # === 1. Trajectory Evolution Video (3 temporal slices) ===
            n_steps, _, T, C, H, W = trajectory.shape
            t_indices = [0, T // 2, T - 1]
            
            for c in range(C):
                print(f"  Creating trajectory video for channel {c}/{C}...")
                
                # Create figure with manual layout
                fig = plt.figure(figsize=(10, 12))
                gs = fig.add_gridspec(3, 2, width_ratios=[1, 1], right=0.85)
                
                axes = []
                for r in range(3):
                    row_axes = []
                    for col in range(2):
                        row_axes.append(fig.add_subplot(gs[r, col]))
                    axes.append(row_axes)
                axes = np.array(axes)
                
                title = f"Channel {c} | Left: Pred | Right: GT\nSolver: {solver}{noise_suffix}"
                fig.suptitle(title)
                
                ims_pred = []
                slice_data_pred = []
                
                for i, t_idx in enumerate(t_indices):
                    # Prediction flow: (Steps, H, W)
                    data_steps = trajectory[:, 0, t_idx, c, :, :].cpu().numpy()
                    slice_data_pred.append(data_steps)
                    
                    # Ground Truth: (H, W)
                    gt_img = z1_gt[0, t_idx, c, :, :].cpu().numpy()
                    
                    # Color scale based on GT
                    vmin = np.percentile(gt_img, 2)
                    vmax = np.percentile(gt_img, 98)
                    
                    # Plot prediction (animated)
                    ax_pred = axes[i, 0]
                    im_pred = ax_pred.imshow(
                        data_steps[0],
                        cmap='viridis',
                        vmin=vmin,
                        vmax=vmax,
                        animated=True
                    )
                    ax_pred.set_title(f"Pred T={t_idx} (Step 0/{n_steps})")
                    ax_pred.axis('off')
                    ims_pred.append(im_pred)
                    
                    # Plot ground truth (static)
                    ax_gt = axes[i, 1]
                    im_gt = ax_gt.imshow(
                        gt_img,
                        cmap='viridis',
                        vmin=vmin,
                        vmax=vmax
                    )
                    ax_gt.set_title(f"Ground Truth T={t_idx}")
                    ax_gt.axis('off')
                
                # Add colorbar
                cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
                fig.colorbar(im_gt, cax=cbar_ax)
                
                # Update function
                def update(frame_idx):
                    for i, im in enumerate(ims_pred):
                        im.set_data(slice_data_pred[i][frame_idx])
                        axes[i, 0].set_title(f"Pred T={t_indices[i]} (Step {frame_idx}/{n_steps})")
                    return ims_pred
                
                # Save animation
                anim = animation.FuncAnimation(
                    fig,
                    update,
                    frames=n_steps,
                    interval=50,
                    blit=False
                )
                
                video_name = f"channel_{c}_steps_{num_steps}_{solver}{noise_suffix}_trajectory.mp4"
                video_path = viz_dir / video_name
                
                try:
                    writer = animation.FFMpegWriter(fps=20, bitrate=3000)
                    anim.save(str(video_path), writer=writer)
                    print(f"    Saved: {video_path}")
                except Exception as e:
                    print(f"    Error saving video: {e}")
                finally:
                    plt.close(fig)
        
        # === 2. Final Prediction Videos (full temporal sequence) ===
        print(f"\nCreating final prediction videos{noise_suffix}...")
        
        for solver_key, final_pred in final_preds.items():
            num_steps_str, solver_name = solver_key.rsplit('_', 1)
            
            for c in range(C):
                print(f"  Creating final video for {solver_key}, channel {c}/{C}...")
                
                # Create figure
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"Final Prediction - Channel {c} | {solver_key}{noise_suffix}")
                
                # Get temporal data
                gt_temporal = z1_gt[0, :, c, :, :].cpu().numpy()  # (T, H, W)
                pred_temporal = final_pred[:, c, :, :].cpu().numpy()  # (T, H, W)
                
                # Color scale
                vmin = np.percentile(gt_temporal, 2)
                vmax = np.percentile(gt_temporal, 98)
                
                # Initialize images
                im_pred = axes[0].imshow(pred_temporal[0], cmap='viridis', vmin=vmin, vmax=vmax, animated=True)
                axes[0].set_title(f"Prediction (t=0/{T})")
                axes[0].axis('off')
                
                im_gt = axes[1].imshow(gt_temporal[0], cmap='viridis', vmin=vmin, vmax=vmax, animated=True)
                axes[1].set_title(f"Ground Truth (t=0/{T})")
                axes[1].axis('off')
                
                # Add colorbar
                fig.colorbar(im_gt, ax=axes, fraction=0.046, pad=0.04)
                
                # Update function
                def update_temporal(frame):
                    im_pred.set_data(pred_temporal[frame])
                    im_gt.set_data(gt_temporal[frame])
                    axes[0].set_title(f"Prediction (t={frame}/{T})")
                    axes[1].set_title(f"Ground Truth (t={frame}/{T})")
                    return [im_pred, im_gt]
                
                # Create animation
                anim_temporal = animation.FuncAnimation(
                    fig,
                    update_temporal,
                    frames=T,
                    interval=200,
                    blit=False
                )
                
                # Save
                video_name = f"final_prediction_channel_{c}_{solver_key}{noise_suffix}.mp4"
                video_path = viz_dir / video_name
                
                try:
                    writer = animation.FFMpegWriter(fps=5, bitrate=3000)
                    anim_temporal.save(str(video_path), writer=writer)
                    print(f"    Saved: {video_path}")
                except Exception as e:
                    print(f"    Error saving video: {e}")
                finally:
                    plt.close(fig)