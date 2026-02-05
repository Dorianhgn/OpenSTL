"""
JvM Lightning Modules - Training modules for Just video Mamba architectures.

Implements the "Just Image" philosophy from:
"Back to Basics: Let Denoising Generative Models Denoise" (Li & He, 2026)
https://arxiv.org/abs/2511.13720

Key principle: Models predict clean images (x) directly, not noise (epsilon) or velocity (v).
This keeps predictions on the natural image manifold and significantly improves generation quality.

Modules:
- JvMBaseLightningModule: Base class with all common CFM logic
- JvMLightningModule: Pure Mamba version (no external context encoder)
- JvTMLightningModule: Contextual version with ViTEncoder3D for conditions
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchdiffeq import odeint
from typing import Literal, Optional
from pathlib import Path
from torchmetrics.image import StructuralSimilarityIndexMeasure
from einops import rearrange

from models.jvm import JvMModel, JvTMModel
from models.cfm.vit_encoder_3d import ViTEncoder3D
from models.cfm.context_cnn import ContextNet

# Import OT sampler
try:
    from torchcfm.optimal_transport import OTPlanSampler
except ImportError:
    import warnings
    warnings.warn("OTPlanSampler not found in torchcfm, OT sampling disabled")
    OTPlanSampler = None


# ==============================================================================
# Base Lightning Module
# ==============================================================================

class JvMBaseLightningModule(pl.LightningModule):
    """
    Base Lightning Module for JvM architectures.
    
    Implements "Just Image" approach from Li & He (2026):
    - Models predict clean image x directly (NOT epsilon or velocity)
    - Loss can be computed in velocity space (v-loss) or pixel space (x-loss)
    - v-loss provides better results with implicit time weighting
    
    Paper findings:
    - Predicting epsilon: FID > 96 (fails - off-manifold)
    - Predicting x with x-loss: FID ~10.14 (good)
    - Predicting x with v-loss: FID ~8.62 (best - implicit weighting)
    
    Implements all common functionality:
    - Conditional Flow Matching (CFM)
    - Training step with configurable v-loss or x-loss
    - Validation/Test steps
    - ODE sampling
    - Optimizer configuration
    - Visualization
    
    Subclasses must implement:
    - _build_velocity_net(): Create the velocity network
    - _prepare_model_inputs(): Prepare inputs for the specific model type
    
    Args:
        # Data dimensions
        in_channels: Input condition channels (raw weather data)
        latent_channels: Latent channels (output prediction channels)
        t_in: Input sequence length
        t_out: Output sequence length
        
        # Training
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        sigma_min: Minimum noise level for CFM
        loss_type: "v" for v-loss (best), "x" for x-loss
        
        # Inference
        num_inference_steps: Number of ODE steps for sampling
        ode_solver: ODE solver ("dopri5", "rk4", "euler")
        
        # Visualization
        visualization_params: List of (num_steps, solver) for trajectory visualization
    
    References:
        Li & He (2026). Back to Basics: Let Denoising Generative Models Denoise.
        arXiv:2511.13720
    """
    
    def __init__(
        self,
        # Data dimensions
        in_channels: int = 27,
        latent_channels: int = 4,
        t_in: int = 8,
        t_out: int = 24,
        
        # Training
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        sigma_min: float = 0.0,
        prediction_mode: Literal["x", "epsilon", "v"] = "x", # "x": JvM, "epsilon": diffusion, "v": CFM
        loss_type: str = "v",  # "v" for v-loss, "x" for x-loss
        
        # Time sampling distribution (Li & He 2026 + David Bertoin advice)
        sample_t_distrib: str = "uniform",  # "uniform" or "logit_normal"
        logit_normal_loc: float = 0.0,  # Center around 0.5
        logit_normal_scale: float = 1.0,  # Lower = tighter around 0.5
        
        # Classifier-Free Guidance (CFG)
        cond_dropout_prob: float = 0.0,  # Probability of dropping condition during training
        guidance_scale: float = 1.0,  # Guidance scale for sampling (>1.0 amplifies condition)
        
        # Optimal Transport sampling
        use_ot_sampling: bool = False,  # Enable OT-based x0 sampling
        ot_method: str = "exact",  # OT method: "exact", "sinkhorn", etc.
        ot_reg: float = 0.05,  # Regularization for Sinkhorn-based OT
        
        # Inference
        num_inference_steps: int = 50,
        ode_solver: str = "dopri5",
        test_method: Optional[str] = None,  # Or "full" for full ODE test on the whole test set
        test_num_ensemble: Optional[int] = 10,
        test_threshold: Optional[float] = 0.13, # for detection metrics (derived from Bayesian Analysis of trajectories)
        
        # Visualization
        visualization_params: Optional[list[tuple[int, str]]] = None,
        viz_sample_noise: Optional[list[Optional[float]]] = None,

        enable_logging: bool = True,  # Enable logging by default
        
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store dimensions
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.t_in = t_in
        self.t_out = t_out
        self.loss_type = loss_type
        self.prediction_mode = prediction_mode
        
        # Time sampling
        self.sample_t_distrib = sample_t_distrib
        self.logit_normal_loc = logit_normal_loc
        self.logit_normal_scale = logit_normal_scale
        
        # CFG parameters
        self.cond_dropout_prob = cond_dropout_prob
        self.guidance_scale = guidance_scale
        
        # OT sampler (optional)
        self.use_ot_sampling = use_ot_sampling and (OTPlanSampler is not None)
        if self.use_ot_sampling:
            self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg)
        else:
            self.ot_sampler = None
        
        # Build velocity network (implemented by subclasses)
        self.velocity_net = self._build_velocity_net()

        self.t_end = 1.0 - 1e-3 # Avoid t=1.0 for numerical stability during sampling but very close
        # Logic enforcement
        if self.prediction_mode == "v":
            print("Mode 'v' selected: Forcing loss_type='v' and ignoring x-loss optimization.")
            self.loss_type = "v"
            self.t_end = 1.0 # Can go all the way to t=1.0 when predicting velocity directly

        # Conditional Flow Matcher
        self.fm = ConditionalFlowMatcher(sigma=sigma_min)
        
        # Metrics
        self.train_ssim = StructuralSimilarityIndexMeasure()
        self.val_ssim = StructuralSimilarityIndexMeasure()
        self.test_ssim = StructuralSimilarityIndexMeasure()
        
        # Visualization
        self.visualization_params = visualization_params or [(50, "dopri5"), (100, "rk4")]
        
        # Logging control
        self.enable_logging = enable_logging
        
        # NOTE: AOD targets are NOT normalized in WeatherCuboidDataset
        # Only input features (MET, static, time) are normalized
        # Therefore, predictions are already in physical space (no denormalization needed)
        # See weather_cuboid.py line 283-287: only X is normalized, not y (AOD target)
        self.aod_is_normalized = False
        
        # Storage for per-frame metrics (used in full test mode)
        self.test_frame_metrics = []
    
    def _build_velocity_net(self):
        """Build the velocity network. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_velocity_net()")
    
    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Set normalization statistics for denormalization during evaluation.
        
        Args:
            mean: Normalization mean (C,) or (1, 1, C, 1, 1)
            std: Normalization std (C,) or (1, 1, C, 1, 1)
        """
        self.register_buffer('norm_mean', mean)
        self.register_buffer('norm_std', std)
        print(f"✅ Normalization stats registered: mean shape={mean.shape}, std shape={std.shape}")
    
    def _prepare_model_inputs(self, x_t: torch.Tensor, t: torch.Tensor, x_cond_raw: torch.Tensor):
        """
        Prepare inputs for the model forward pass.
        
        Must be implemented by subclasses to handle their specific input format.
        
        Args:
            x_t: Noisy latent (B, T_out, C_latent, H, W)
            t: Time steps (B,)
            x_cond_raw: Raw condition data (B, T_in, C_in, H, W)
            
        Returns:
            Dictionary with model inputs
        """
        raise NotImplementedError("Subclasses must implement _prepare_model_inputs()")
    
    def forward(self, batch):
        """
        Forward pass for inference and model analysis.
        
        Args:
            batch: Dictionary with keys:
                - 'cond': Condition tensor (B, T_in, C_in, H, W)
                - 'x_past': Optional past latents (B, T_in, C_out, H, W)
                - 'x_future': Optional target for shape inference
                
        Returns:
            Predicted output (B, T_out, C_out, H, W)
        """
        # Extract inputs
        x_cond_raw = batch['cond']
        x_past = batch.get('x_past', None)
        
        # Infer target shape from x_future if available, else use defaults
        if 'x_future' in batch:
            target_shape = batch['x_future'].shape
        else:
            # Default shape based on model config
            B = x_cond_raw.shape[0]
            target_shape = (B, self.hparams.t_out, self.hparams.latent_channels, 
                          x_cond_raw.shape[-2], x_cond_raw.shape[-1])
        
        # Sample from the model
        output = self.sample(
            x_cond_raw=x_cond_raw,
            x_past=x_past,
            target_shape=target_shape,
            num_steps=self.hparams.num_inference_steps,
            solver=self.hparams.ode_solver,
            return_trajectory=False
        )
        
        return output
    
    # ============================================================================
    # Core Step Method (Shared between train/val)
    # ============================================================================
    
    def step(self, batch: dict, batch_idx: int, stage: str = "train") -> tuple[torch.Tensor, dict]:
        """
        Shared step logic for training and validation.
        
        Implementation of "Back to Basics: Let Denoising Generative Models Denoise"
        (Li & He, 2026) - https://arxiv.org/abs/2511.13720
        
        Key insight from the paper:
        - Prediction space: Model predicts x (clean image) directly, NOT epsilon or v
          → Predicting epsilon fails (FID > 96) because it's off-manifold
          → Predicting x works (FID ~8-10) because it stays on the visual manifold
        
        - Loss space: Two options with the same x-prediction model:
          → x-loss: MSE(x_pred, x_target) - FID 10.14
          → v-loss: MSE(v_derived, v_target) - FID 8.62 ✓ BEST
        
        The v-loss is computed by:
          1. Model outputs x_pred (the denoised image)
          2. Derive velocity: v_pred = (x_pred - x_t) / (1 - t)
          3. Target velocity (from linear flow): v_target = x_1 - x_0
          4. Loss: MSE(v_pred, v_target)
        
        Why v-loss is better:
        - It provides implicit time-dependent weighting: weight ∝ 1/(1-t)²
        - Emphasizes difficult timesteps (high noise) more than easy ones
        - More stable training for long-horizon generation
        
        Flow:
        1. Get condition and target from batch
        2. Sample time t ~ U(0,1)
        3. Add noise: x_t = t*x_1 + (1-t)*x_0  (linear interpolation)
        4. Predict clean image: x_pred = model(x_t, t, conditions)
        5. Derive velocity: v_pred = (x_pred - x_t) / (1 - t)
        6. Compute both losses and return selected one
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            stage: "train" or "val"
            
        Returns:
            loss: Loss tensor to optimize
            metrics: Dictionary of metrics to log
        
        References:
            Li & He (2026). Back to Basics: Let Denoising Generative Models Denoise.
            arXiv:2511.13720
        """
        # Get data (support both old and new format)
        if "x_future" in batch:
            # New format: past+future strategy
            x_future = batch["x_future"]  # (B, T_out, C_out, H, W) - target
            x_cond_raw = batch["cond"]  # (B, T_in, C_in, H, W) - spatial condition
            x_past = batch.get("x_past", None)  # (B, T_in, C_out, H, W) - past latents or None
        elif "x_target" in batch:
            # Old format: backward compatibility
            x_future = batch["x_target"]
            x_cond_raw = batch["x_cond"]
            x_past = batch.get("x_past", None)
        else:
            raise KeyError(f"Batch must contain 'x_future'/'x_target' keys. Got: {batch.keys()}")
        
        B = x_future.shape[0]
        device = x_future.device
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(x_future)
        
        # Optimal Transport Sampling: Permute x_0 to minimize transport cost to x_future
        # CRITICAL: Only permute x_0 (noise), NOT x_future or x_cond_raw (conditions)
        # This preserves the alignment (x_future[i], x_cond_raw[i]) while improving trajectories
        if stage == "train" and self.use_ot_sampling:
            x_0_opt, _ = self.ot_sampler.sample_plan(x_0, x_future)
            x_0 = x_0_opt
        
        # === Classifier-Free Guidance: Randomly drop condition during training ===
        # Dropout condition with probability cond_dropout_prob for CFG
        if stage == "train" and self.cond_dropout_prob > 0:
            # Create mask: True = drop condition (replace with zeros)
            drop_mask = torch.rand(B, device=device) < self.cond_dropout_prob
            # Apply mask: zero out condition where drop_mask is True
            # Shape: (B, T_in, C_in, H, W) -> mask needs to be (B, 1, 1, 1, 1)
            x_cond_raw = x_cond_raw * (~drop_mask).view(B, 1, 1, 1, 1).float()
        
        # === Flow Matching: Sample location and flow (all-in-one) ===
        # Uses ConditionalFlowMatcher.sample_location_and_conditional_flow
        # This samples t, computes x_t, and computes the target velocity v_target
        # in one optimized step (Li & He, 2026):
        #   x_t = (1-t)*x_0 + t*x_1 + sigma * epsilon
        #   v_target = x_1 - x_0 (for linear flow)
        
        if self.sample_t_distrib == "logit_normal":
            # Sample t from logit-normal distribution (David Bertoin advice)
            # More emphasis on middle timesteps for better training stability
            t = torch.randn(B, device=device)
            t = t * self.logit_normal_scale + self.logit_normal_loc
            t = torch.sigmoid(t)
            t = torch.clamp(t, min=1e-3, max=1.0 - 0.05) # The magic number de Li & He (Sec 4.3)
            t, x_t, v_target = self.fm.sample_location_and_conditional_flow(x_0, x_future, t=t)
        else:
            # Default: uniform sampling t ~ U(0,1)
            t, x_t, v_target = self.fm.sample_location_and_conditional_flow(x_0, x_future, t=None)
        
        # === Model Prediction ===
        # Prepare model inputs (subclass-specific, now includes x_past)
        model_inputs = self._prepare_model_inputs(x_t, t, x_cond_raw, x_past=x_past)
        
        if self.prediction_mode == "x":
            # KEY INSIGHT (Li & He, 2026):
            # The model predicts x directly (the clean image), NOT epsilon or velocity.
            # This keeps predictions on the visual manifold and works much better.
            x_pred = self.velocity_net.predict_x_from_xt(**model_inputs)
            
            # === Velocity Derivation ===
            # From the predicted x, we derive the velocity field (Li & He, 2026, Eq. 6):
            #   v = dx/dt = (x - x_t) / (1 - t)
            # This is the instantaneous direction to move from x_t toward x.
            # Note: v_target was already computed by sample_location_and_conditional_flow above
            v_pred = self.velocity_net.compute_v_from_x_pred(x_pred, x_t, t)
            
            # === Loss Computation ===
            # Compute both losses for comparison (Li & He, 2026, Section 3):
            
            # v-loss: MSE in velocity space (BEST - FID 8.62)
            # Provides implicit time-dependent weighting ∝ 1/(1-t)²
            # Emphasizes difficult (noisy) timesteps more
            v_loss = nn.functional.mse_loss(v_pred, v_target)
            
            # x-loss: MSE in pixel space (GOOD - FID 10.14)
            # Direct comparison of predicted vs target clean image
            x_loss = nn.functional.mse_loss(x_pred, x_future)

        elif self.prediction_mode == "v":

            # Alternative: Model predicts velocity directly
            v_pred = self.velocity_net.predict_x_from_xt(**model_inputs)
            
            t_view = t.view(B, 1, 1, 1, 1)
            x_pred = x_t + v_pred * (1 - t_view)

            # === Loss Computation ===
            v_loss = nn.functional.mse_loss(v_pred, v_target)
            x_loss = nn.functional.mse_loss(x_pred, x_future)
        else:
            raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}")
        
        # === Select Loss ===
        # Based on hyperparameter, choose which loss to optimize
        # Paper recommendation: v-loss for best performance (Li & He, 2026)
        if self.loss_type == "v":
            loss = v_loss
        elif self.loss_type == "x":
            loss = x_loss
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Must be 'v' or 'x'.")
        
        # Compute metrics
        with torch.no_grad():
            # Norms for monitoring (flatten spatial/temporal dims then compute norm)
            v_pred_norm = v_pred.flatten(1).norm(dim=1).mean()
            v_target_norm = v_target.flatten(1).norm(dim=1).mean()
            x_pred_norm = x_pred.flatten(1).norm(dim=1).mean()
            x_target_norm = x_future.flatten(1).norm(dim=1).mean()
            x_t_norm = x_t.flatten(1).norm(dim=1).mean()
            t_mean = t.mean()
            
            # SSIM (only on first 3 channels, averaged over time)
            x_pred_vis = torch.clamp(x_pred[:, :, :3], -1, 1)
            x_target_vis = torch.clamp(x_future[:, :, :3], -1, 1)
            x_pred_avg = x_pred_vis.mean(dim=1)  # (B, 3, H, W)
            x_target_avg = x_target_vis.mean(dim=1)
            
            if stage == "train":
                ssim = self.train_ssim(x_pred_avg, x_target_avg)
            elif stage == "val":
                ssim = self.val_ssim(x_pred_avg, x_target_avg)
            else:
                ssim = self.test_ssim(x_pred_avg, x_target_avg)
        
        # Prepare metrics dictionary
        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/v_loss": v_loss,
            f"{stage}/x_loss": x_loss,
            f"{stage}/ssim": ssim,
            f"{stage}/v_pred_norm": v_pred_norm,
            f"{stage}/v_target_norm": v_target_norm,
            f"{stage}/x_pred_norm": x_pred_norm,
            f"{stage}/x_target_norm": x_target_norm,
            f"{stage}/x_t_norm": x_t_norm,
            f"{stage}/t_mean": t_mean,
        }
        
        return loss, metrics

    # ===========================================================================
    # Full step method (ODE)
    # ==========================================================================

    def _step_with_sampling(self, batch: dict, stage: str = "test", num_ensemble=10, threshold=0.13) -> torch.Tensor:
        """
        Full test step with ODE sampling on whole batch with ensemble predictions.

        Args:
            batch: Batch dictionary
            stage: "test"
            num_ensemble: Number of ensemble samples to average (overridden by hparams)
            threshold: Threshold for detection metrics (overridden by hparams)
        Returns:
            Loss value (sample MSE)
        """

        # Get data (support both old and new format)
        if "x_future" in batch:
            # New format: past+future strategy
            x_future = batch["x_future"]  # (B, T_out, C_out, H, W) - target
            x_cond_raw = batch["cond"]  # (B, T_in, C_in, H, W) - spatial condition
            x_past = batch.get("x_past", None)  # (B, T_in, C_out, H, W) - past latents or None
        elif "x_target" in batch:
            # Old format: backward compatibility
            x_future = batch["x_target"]
            x_cond_raw = batch["x_cond"]
            x_past = batch.get("x_past", None)
        else:
            raise KeyError(f"Batch must contain 'x_future'/'x_target' keys. Got: {batch.keys()}")

        B, T, C, H, W = x_future.shape
        target_shape = x_future.shape
        
        # Use hyperparameters if available
        num_ensemble = num_ensemble if num_ensemble is not None else 10
        threshold = threshold if threshold is not None else 0.13
        
        # Generate ensemble predictions
        ensemble_preds = []
        
        with torch.no_grad():
            for ens_idx in range(num_ensemble):
                # Sample trajectory (JvM or CFM depending on config)
                pred = self.sample(
                    x_cond_raw,
                    x_past=x_past,
                    target_shape=target_shape,
                    num_steps=self.hparams.num_inference_steps,
                    solver=self.hparams.ode_solver,
                )
                ensemble_preds.append(pred.detach().cpu())
            
            # Move target to CPU for metric computation
            x_future_cpu = x_future.detach().cpu()
            
            # Stack ensemble -> (B, N_ens, T, C, H, W)
            ensemble_preds = torch.stack(ensemble_preds, dim=1)
            
            # ===== Physical Space (No Denormalization Needed) =====
            # NOTE: AOD targets are NOT normalized in the datamodule
            # (see weather_cuboid.py line 283-287: only inputs are normalized, not AOD)
            # Therefore predictions are already in physical space
            # If you need denormalization for other setups, use set_normalization_stats()
            
            if self.aod_is_normalized and hasattr(self, 'norm_mean') and hasattr(self, 'norm_std'):
                # Optional: denormalize if AOD was normalized (not the case by default)
                mean = self.norm_mean.cpu()
                std = self.norm_std.cpu()
                
                if mean.dim() == 1:
                    mean = mean.view(1, 1, 1, -1, 1, 1)
                    std = std.view(1, 1, 1, -1, 1, 1)
                
                preds_phys = ensemble_preds * std + mean
                target_phys = x_future_cpu * std.squeeze(1) + mean.squeeze(1)
            else:
                # Default: predictions are already in physical space (AOD not normalized)
                preds_phys = ensemble_preds
                target_phys = x_future_cpu
            
            # ===== Compute Metrics =====
            
            # Mean prediction across ensemble (for deterministic-like metrics)
            pred_mean = preds_phys.mean(dim=1)  # (B, T, C, H, W)
            
            # 1. MSE (Mean Squared Error)
            mse = nn.functional.mse_loss(pred_mean, target_phys).item()
            
            # 2. MAE (Mean Absolute Error)
            mae = nn.functional.l1_loss(pred_mean, target_phys).item()
            
            # 3. SSIM (Structural Similarity) - per frame + store per-frame metrics
            ssim_per_frame = []
            for t_idx in range(T):
                # Clamp to valid range for SSIM (work with first 3 channels or all)
                # SSIM expects values in reasonable range
                n_channels = min(C, 3)
                pred_frame = torch.clamp(pred_mean[:, t_idx, :n_channels], -1, 1)
                target_frame = torch.clamp(target_phys[:, t_idx, :n_channels], -1, 1)
                ssim_t = self.test_ssim(pred_frame, target_frame)
                ssim_per_frame.append(ssim_t)
                
                # Per-frame MSE and MAE for this timestep
                frame_mse = nn.functional.mse_loss(pred_mean[:, t_idx], target_phys[:, t_idx]).item()
                frame_mae = nn.functional.l1_loss(pred_mean[:, t_idx], target_phys[:, t_idx]).item()
                
                # Per-frame detection metrics (POD, FAR, CSI)
                pred_binary_frame = (pred_mean[:, t_idx] > threshold).float()
                target_binary_frame = (target_phys[:, t_idx] > threshold).float()
                
                hits_frame = (pred_binary_frame * target_binary_frame).sum()
                misses_frame = ((1 - pred_binary_frame) * target_binary_frame).sum()
                false_alarms_frame = (pred_binary_frame * (1 - target_binary_frame)).sum()
                
                epsilon = 1e-6
                pod_frame = (hits_frame / (hits_frame + misses_frame + epsilon)).item()
                far_frame = (false_alarms_frame / (hits_frame + false_alarms_frame + epsilon)).item()
                csi_frame = (hits_frame / (hits_frame + misses_frame + false_alarms_frame + epsilon)).item()
                
                # Per-frame CRPS (if ensemble available)
                if num_ensemble > 1:
                    # Term 1: Mean absolute error for this frame
                    term1_frame = torch.abs(preds_phys[:, :, t_idx] - target_phys[:, t_idx].unsqueeze(1)).mean()
                    
                    # Term 2: Mean pairwise distance between ensemble members for this frame
                    term2_frame = 0.0
                    count_frame = 0
                    for i in range(num_ensemble):
                        for j in range(i + 1, num_ensemble):
                            term2_frame += torch.abs(preds_phys[:, i, t_idx] - preds_phys[:, j, t_idx]).mean()
                            count_frame += 1
                    term2_frame = term2_frame / count_frame if count_frame > 0 else 0.0
                    crps_frame = (term1_frame - 0.5 * term2_frame).item()
                else:
                    crps_frame = frame_mae
                
                # Store per-frame metrics for CSV export
                if self.enable_logging:
                    self.test_frame_metrics.append({
                        'frame_idx': t_idx,
                        'ssim': ssim_t.item(),
                        'mse': frame_mse,
                        'mae': frame_mae,
                        'crps': crps_frame,
                        'pod': pod_frame,
                        'far': far_frame,
                        'csi': csi_frame,
                    })
            
            # Compute aggregate SSIM metrics
            ssim_mean = torch.stack(ssim_per_frame).mean()
            ssim_start = ssim_per_frame[0]
            ssim_mid = ssim_per_frame[T // 2]
            ssim_end = ssim_per_frame[-1]
            
            # 4. CRPS (Continuous Ranked Probability Score) - Energy Score for ensemble
            if num_ensemble > 1:
                # Term 1: Mean absolute error of each member vs target
                term1 = torch.abs(preds_phys - target_phys.unsqueeze(1)).mean()
                
                # Term 2: Mean pairwise distance between ensemble members (dispersion)
                term2 = 0.0
                count = 0
                for i in range(num_ensemble):
                    for j in range(i + 1, num_ensemble):
                        term2 += torch.abs(preds_phys[:, i] - preds_phys[:, j]).mean()
                        count += 1
                term2 = term2 / count if count > 0 else 0.0
                
                crps = (term1 - 0.5 * term2).item()
            else:
                # For single ensemble member, CRPS = MAE
                crps = mae
            
            # 5. Detection Metrics (CSI, POD, FAR)
            # Binarize based on threshold
            pred_binary = (pred_mean > threshold).float()
            target_binary = (target_phys > threshold).float()
            
            # Compute contingency table elements
            hits = (pred_binary * target_binary).sum()
            misses = ((1 - pred_binary) * target_binary).sum()
            false_alarms = (pred_binary * (1 - target_binary)).sum()
            
            epsilon = 1e-6
            pod = (hits / (hits + misses + epsilon)).item()  # Probability of Detection
            far = (false_alarms / (hits + false_alarms + epsilon)).item()  # False Alarm Ratio
            csi = (hits / (hits + misses + false_alarms + epsilon)).item()  # Critical Success Index
            
            # Log all metrics
            if self.enable_logging:
                self.log(f"{stage}/sample_mse", mse, sync_dist=True, prog_bar=True)
                self.log(f"{stage}/sample_mae", mae, sync_dist=True)
                self.log(f"{stage}/sample_ssim", ssim_mean, sync_dist=True, prog_bar=True)
                self.log(f"{stage}/sample_ssim_start", ssim_start, sync_dist=True)
                self.log(f"{stage}/sample_ssim_mid", ssim_mid, sync_dist=True)
                self.log(f"{stage}/sample_ssim_end", ssim_end, sync_dist=True)
                self.log(f"{stage}/sample_crps", crps, sync_dist=True)
                self.log(f"{stage}/sample_pod", pod, sync_dist=True)
                self.log(f"{stage}/sample_far", far, sync_dist=True)
                self.log(f"{stage}/sample_csi", csi, sync_dist=True)
        
        return mse  # Return MSE as loss
    
    # ============================================================================
    # Training Step
    # ============================================================================
    
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step using shared step logic."""
        loss, metrics = self.step(batch, batch_idx, stage="train")
        
        # Log metrics
        if self.enable_logging:
            self.log("train/loss", metrics["train/loss"], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/v_loss", metrics["train/v_loss"], on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/x_loss", metrics["train/x_loss"], on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/ssim", metrics["train/ssim"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train/v_pred_norm", metrics["train/v_pred_norm"], on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/x_t_norm", metrics["train/x_t_norm"], on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    # ============================================================================
    # Validation & Test Steps
    # ============================================================================
    
    def _shared_validation_test_step(self, batch: dict, batch_idx: int, stage: str = "val") -> torch.Tensor:
        """
        Shared validation/test step using training step logic (NOT sampling).
        
        Sampling is done only once per epoch in on_validation_epoch_end for efficiency.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            stage: "val" or "test"
            
        Returns:
            Loss value
        """
        # Store first batch for epoch-end sampling
        if batch_idx == 0:
            self._eval_sample_batch = {k: v.detach().clone() for k, v in batch.items()}
        
        # Compute loss using same logic as training (efficient)
        loss, metrics = self.step(batch, batch_idx, stage=stage)
        
        # Log metrics
        if self.enable_logging:
            self.log(f"{stage}/loss", metrics[f"{stage}/loss"], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/v_loss", metrics[f"{stage}/v_loss"], on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/x_loss", metrics[f"{stage}/x_loss"], on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/ssim", metrics[f"{stage}/ssim"], on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._shared_validation_test_step(batch, batch_idx, stage="val")
    
    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Test step with visualization."""
        if self.hparams.test_method == "full":
            # Full ODE test step with ensemble sampling
            loss = self._step_with_sampling(batch, stage="test", num_ensemble=self.hparams.test_num_ensemble, threshold=self.hparams.test_threshold)
        else:
            loss = self._shared_validation_test_step(batch, batch_idx, stage="test")
        
            # Visualization on first batch
            if batch_idx == 0 and self.enable_logging:
                viz_dir = Path(self.logger.log_dir) / "visualizations"
                viz_dir.mkdir(exist_ok=True, parents=True)
                self._visualize_trajectory_evolution(batch, viz_dir)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Compute epoch-level metrics for validation with ODE sampling."""
        # Compute accumulated SSIM
        if self.enable_logging:
            val_ssim = self.val_ssim.compute()
            self.log("val/ssim_epoch", val_ssim, sync_dist=True)
            self.val_ssim.reset()
        
        # Sample-based metrics (expensive, done once per epoch)
        if not self.enable_logging or not hasattr(self, '_eval_sample_batch'):
            return
        
        batch = self._eval_sample_batch
        
        # Handle different batch formats (robust mapping)
        if "x_cond" in batch:
            x_cond_raw = batch["x_cond"]
            x_target = batch["x_target"]
            x_past = None
        elif "cond" in batch:
            x_cond_raw = batch["cond"]
            x_target = batch["x_future"]
            x_past = batch.get("x_past", None)
        else:
            # Fallback for old tuple format
            if isinstance(batch, (tuple, list)):
                x_target, x_cond_raw = batch
                x_past = None
            else:
                return
        
        try:
            with torch.no_grad():
                # Sample only 4 examples for efficiency
                x_cond_sample = x_cond_raw[:4]
                x_past_sample = x_past[:4] if x_past is not None else None
                target = x_target[:4]
                target_shape = target.shape
                
                # Full ODE sampling (expensive, done once per epoch)
                x_samples = self.sample(
                    x_cond_sample,
                    x_past=x_past_sample,
                    target_shape=target_shape,
                    num_steps=self.hparams.num_inference_steps,
                    solver=self.hparams.ode_solver,
                )
                
                # Compute sample MSE
                sample_mse = nn.functional.mse_loss(x_samples, target).item()
                self.log("val/sample_mse", sample_mse, sync_dist=True)
                
                # Compute SSIM per frame
                ssim_per_frame = []
                for t in range(target.shape[1]):
                    # Clamp to valid range and take first channel (grayscale)
                    pred_frame = torch.clamp(x_samples[:, t, :1], -1, 1)
                    target_frame = torch.clamp(target[:, t, :1], -1, 1)
                    ssim_t = self.val_ssim(pred_frame, target_frame)
                    ssim_per_frame.append(ssim_t)
                
                # Average SSIM
                ssim_mean = torch.stack(ssim_per_frame).mean().item()
                self.log("val/sample_ssim", ssim_mean, sync_dist=True, prog_bar=True)
                
                # Log start/end SSIM
                self.log("val/sample_ssim_start", ssim_per_frame[0], sync_dist=True)
                self.log("val/sample_ssim_end", ssim_per_frame[-1], sync_dist=True)
        
        finally:
            # Clean up to avoid memory leaks
            if hasattr(self, '_eval_sample_batch'):
                del self._eval_sample_batch
    
    def on_test_epoch_end(self):
        """Compute epoch-level metrics for test with ODE sampling."""

        if self.hparams.test_method == "full":
            # Save per-frame metrics to CSV
            if self.enable_logging and len(self.test_frame_metrics) > 0:
                import pandas as pd
                from pathlib import Path
                
                # Create DataFrame from accumulated metrics
                df = pd.DataFrame(self.test_frame_metrics)
                
                # Group by frame_idx and compute statistics across all batches
                df_stats = df.groupby('frame_idx').agg({
                    'ssim': ['mean', 'std', 'min', 'max'],
                    'mse': ['mean', 'std', 'min', 'max'],
                    'mae': ['mean', 'std', 'min', 'max'],
                    'crps': ['mean', 'std', 'min', 'max'],
                    'pod': ['mean', 'std', 'min', 'max'],
                    'far': ['mean', 'std', 'min', 'max'],
                    'csi': ['mean', 'std', 'min', 'max'],
                }).reset_index()
                
                # Flatten column names
                df_stats.columns = ['_'.join(col).strip('_') for col in df_stats.columns.values]
                
                # Save to CSV
                output_dir = Path(self.logger.log_dir) if self.logger.log_dir is not None else Path('.')
                output_path = output_dir / 'test_per_frame_metrics.csv'
                df_stats.to_csv(output_path, index=False)
                print(f"\n✅ Per-frame metrics saved to: {output_path}")
                print(f"   Metrics: SSIM, MSE, MAE, CRPS, POD, FAR, CSI")
                print(f"   Total frames: {len(df_stats)}, Total batches: {len(df)}")
                
                # Also save raw data (all batches)
                output_path_raw = output_dir / 'test_per_frame_metrics_raw.csv'
                df.to_csv(output_path_raw, index=False)
                print(f"✅ Raw per-frame metrics saved to: {output_path_raw}")
                
                # Clear metrics for next test run
                self.test_frame_metrics = []
            
            return  # Already done in test step

        # Compute accumulated SSIM
        if self.enable_logging:
            test_ssim = self.test_ssim.compute()
            self.log("test/ssim_epoch", test_ssim, sync_dist=True)
            self.test_ssim.reset()
        
        # Sample-based metrics
        if not self.enable_logging or not hasattr(self, '_eval_sample_batch'):
            return
        
        batch = self._eval_sample_batch        # Handle different batch formats (robust mapping)
        if "x_cond" in batch:
            x_cond_raw = batch["x_cond"]
            x_target = batch["x_target"]
            x_past = None
        elif "cond" in batch:
            x_cond_raw = batch["cond"]
            x_target = batch["x_future"]
            x_past = batch.get("x_past", None)
        else:
            # Fallback for old tuple format
            if isinstance(batch, (tuple, list)):
                x_target, x_cond_raw = batch
                x_past = None
            else:
                return
        
        try:
            with torch.no_grad():
                # Sample only 4 examples for efficiency
                x_cond_sample = x_cond_raw[:4]
                x_past_sample = x_past[:4] if x_past is not None else None
                target = x_target[:4]
                target_shape = target.shape
                
                # Full ODE sampling (expensive, done once per epoch)
                x_samples = self.sample(
                    x_cond_sample,
                    x_past=x_past_sample,
                    target_shape=target_shape,
                    num_steps=self.hparams.num_inference_steps,
                    solver=self.hparams.ode_solver,
                )
                
                # Compute sample MSE
                sample_mse = nn.functional.mse_loss(x_samples, target).item()
                self.log("test/sample_mse", sample_mse, sync_dist=True)
                
                # Compute SSIM per frame
                ssim_per_frame = []
                for t in range(target.shape[1]):
                    pred_frame = torch.clamp(x_samples[:, t, :1], -1, 1)
                    target_frame = torch.clamp(target[:, t, :1], -1, 1)
                    ssim_t = self.test_ssim(pred_frame, target_frame)
                    ssim_per_frame.append(ssim_t)
                
                # Average SSIM
                ssim_mean = torch.stack(ssim_per_frame).mean().item()
                self.log("test/sample_ssim", ssim_mean, sync_dist=True, prog_bar=True)
                
                # Log start/end SSIM
                self.log("test/sample_ssim_start", ssim_per_frame[0], sync_dist=True)
                self.log("test/sample_ssim_end", ssim_per_frame[-1], sync_dist=True)
        
        finally:
            if hasattr(self, '_eval_sample_batch'):
                del self._eval_sample_batch
    
    # ============================================================================
    # Sampling (ODE)
    # ============================================================================
    
    def sample(
        self,
        x_cond_raw: torch.Tensor,
        x_past: Optional[torch.Tensor] = None,
        target_shape: tuple = None,
        num_steps: int = None,
        solver: str = None,
        return_trajectory: bool = False,
        x_target: torch.Tensor = None,
        noise_t: float = None,
        guidance_scale: float = None,  # Override guidance scale for this sample
    ) -> torch.Tensor:
        """
        Sample from the model using ODE integration.
        
        Args:
            x_cond_raw: Condition data (B, T_in, C_in, H, W)
            x_past: Clean past latent (B, T_in, C_latent, H, W) or None
            target_shape: Target shape (B, T_out, C_latent, H, W). If None, inferred from hparams
            num_steps: Number of ODE steps (default: from hparams)
            solver: ODE solver ('dopri5', 'rk4', 'midpoint', 'euler')
            return_trajectory: If True, return full trajectory
            x_target: Ground truth target for noisy sampling (optional)
            noise_t: Noise level t ∈ [0,1] for noisy sampling (optional)
                    If None, use clean sampling (z0 ~ N(0,1))
                    If float, use noisy target: z0 = x_t = t*x_target + (1-t)*epsilon
            
        Returns:
            Sampled latent (B, T_out, C_latent, H, W)
            or trajectory if return_trajectory=True
        """
        if num_steps is None:
            num_steps = self.hparams.num_inference_steps
        if solver is None:
            solver = self.hparams.ode_solver
        
        B = x_cond_raw.shape[0]
        device = x_cond_raw.device
        
        # Get spatial dimensions
        if target_shape is not None:
            _, T_out, C_latent, H, W = target_shape
        else:
            # Fallback: use condition dimensions (may be wrong for unconditional)
            _, _, _, H, W = x_cond_raw.shape
            T_out = self.t_out
            C_latent = self.latent_channels
        
        # Initial state: clean or noisy sampling
        if noise_t is not None and x_target is not None:
            # Noisy sampling: start from x_t = t*x_target + (1-t)*epsilon
            epsilon = torch.randn(B, T_out, C_latent, H, W, device=device, dtype=torch.float32)
            z0 = noise_t * x_target.to(device=device, dtype=torch.float32) + (1 - noise_t) * epsilon
            # Adjust time span to start from noise_t instead of 1e-3
            t_start = noise_t
        else:
            # Clean sampling: start from pure noise
            z0 = torch.randn(B, T_out, C_latent, H, W, device=device, dtype=torch.float32)
            t_start = 1e-3
        
        # Keep x_cond_raw as-is and convert to FP32 for ODE stability
        x_cond_fp32 = x_cond_raw.to(device=device, dtype=torch.float32)
        
        # Keep x_past on correct device if provided
        x_past_fp32 = x_past.to(device=device, dtype=torch.float32) if x_past is not None else None
        
        # Determine guidance scale (use method argument or instance default)
        cfg_scale = guidance_scale if guidance_scale is not None else self.guidance_scale
        
        # Define ODE function with optional Classifier-Free Guidance
        def ode_func(t, x):
            # t is a scalar, x is (B, T_out, C_latent, H, W)
            t_scalar = t.item() if isinstance(t, torch.Tensor) else float(t)
            t_batch = torch.full((B,), t_scalar, device=device, dtype=x.dtype)
            
            # Prepare model inputs (conditional) - pass x_past if available
            if x_past_fp32 is not None:
                model_inputs = self._prepare_model_inputs(x, t_batch, x_cond_fp32, x_past=x_past_fp32)
            else:
                model_inputs = self._prepare_model_inputs(x, t_batch, x_cond_fp32)
            
            # Predict velocity (conditional)
            if self.prediction_mode == "x":
                x_pred_cond = self.velocity_net.predict_x_from_xt(**model_inputs)
                v_cond = self.velocity_net.compute_v_from_x_pred(x_pred_cond, x, t_batch)
            elif self.prediction_mode == "v":
                v_cond = self.velocity_net.predict_x_from_xt(**model_inputs)
            else:
                raise ValueError(f"Unknown prediction_mode: {self.prediction_mode}")
            
            # If CFG is enabled (guidance_scale != 1.0), compute unconditional prediction
            if cfg_scale != 1.0:
                # Create unconditional input: zero out the condition
                x_cond_uncond = torch.zeros_like(x_cond_fp32)
                
                # Prepare unconditional model inputs
                if x_past_fp32 is not None:
                    model_inputs_uncond = self._prepare_model_inputs(x, t_batch, x_cond_uncond, x_past=x_past_fp32)
                else:
                    model_inputs_uncond = self._prepare_model_inputs(x, t_batch, x_cond_uncond)
                
                # Predict unconditional
                x_pred_uncond = self.velocity_net.predict_x_from_xt(**model_inputs_uncond)
                v_uncond = self.velocity_net.compute_v_from_x_pred(x_pred_uncond, x, t_batch)
                
                # Apply guidance: v_final = v_uncond + w * (v_cond - v_uncond)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
                return v
            else:
                return v_cond
        
        # Time grid [t_start, 1] - Adjust based on sampling mode
        t_span = torch.linspace(t_start, self.t_end, num_steps, device=device, dtype=torch.float32)
        
        # Integration (Force FP32 for stability)
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.autocast(device_type=autocast_device, enabled=False):
            with torch.no_grad():
                if solver == "dopri5":
                    trajectory = odeint(ode_func, z0, t_span, method="dopri5", rtol=1e-5, atol=1e-5)
                elif solver == "rk4":
                    trajectory = odeint(ode_func, z0, t_span, method="rk4")
                elif solver == "midpoint":
                    trajectory = odeint(ode_func, z0, t_span, method="midpoint")
                else:  # euler
                    trajectory = odeint(ode_func, z0, t_span, method="euler")
        
        # J'ai testé, inutile d'ajouter la dernière prédiction...
        # if self.prediction_mode == "x":
        #     model_inputs = self._prepare_model_inputs(trajectory[-1], torch.full((B,), 1.0, device=device, dtype=trajectory[-1].dtype), x_cond_fp32, x_past=x_past_fp32)
        #     torch.cat([trajectory, self.velocity_net.predict_x_from_xt(**model_inputs).unsqueeze(0)], dim=0)
        
        if return_trajectory:
            return trajectory  # (num_steps, B, T_out, C_latent, H, W)
        # Return last state (t=1)
        return trajectory[-1]
    
    # ============================================================================
    # Optimizer Configuration
    # ============================================================================
    
    def configure_optimizers(self):
        """Configure Adam optimizer with cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    # ============================================================================
    # Visualization
    # ============================================================================
    
    def _visualize_trajectory_evolution(self, batch: dict, viz_dir: Path) -> None:
        """Visualize trajectory evolution using unified visualization function."""
        from models.utils import visualize_trajectory_evolution
        
        # Handle batch format (robust mapping)
        if isinstance(batch, (tuple, list)):
            x_target, x_cond_raw = batch
            batch = {"x_target": x_target, "x_cond": x_cond_raw}
            x_past = None
        elif "cond" in batch:
            # New format
            x_cond_raw = batch["cond"]
            z1_gt = batch["x_future"]
            x_past = batch.get("x_past", None)
        elif "x_cond" in batch:
            # Old format
            x_cond_raw = batch["x_cond"]
            z1_gt = batch["x_target"]
            x_past = None
        else:
            print("Unknown batch format, skipping visualization.")
            return
        
        # Select first sample in batch
        if "cond" not in batch:  # Need to extract from already parsed vars
            x_cond_raw = batch.get("x_cond", batch.get("cond"))[:1]
            z1_gt = batch.get("x_target", batch.get("x_future"))[:1]
            x_past = batch.get("x_past", None)
            if x_past is not None:
                x_past = x_past[:1]
        else:
            x_cond_raw = x_cond_raw[:1]
            z1_gt = z1_gt[:1]
            if x_past is not None:
                x_past = x_past[:1]
        
        # Get solver configs
        solver_configs = getattr(self.hparams, 'visualization_params', None)
        if solver_configs is None:
            print("No visualization_params provided, skipping visualization.")
            return
        
        # Get noise levels for visualization
        viz_sample_noise = getattr(self.hparams, 'viz_sample_noise', None)
        
        # Extract target shape
        _, T_out, C_out, H_out, W_out = z1_gt.shape
        target_shape = (1, T_out, C_out, H_out, W_out)
        
        # Create sample function wrapper
        def sample_fn(num_steps, solver, return_trajectory, x_target=None, noise_t=None):
            return self.sample(
                x_cond_raw=x_cond_raw,
                x_past=x_past,
                target_shape=target_shape,
                num_steps=num_steps,
                solver=solver,
                return_trajectory=return_trajectory,
                x_target=x_target,
                noise_t=noise_t,
            )
        
        # Call unified visualization
        visualize_trajectory_evolution(
            sample_fn=sample_fn,
            batch=batch,
            viz_dir=viz_dir,
            solver_configs=solver_configs,
            z1_gt=z1_gt,
            viz_sample_noise=viz_sample_noise,
        )


# ==============================================================================
# JvM Lightning Module (Pure Mamba)
# ==============================================================================

class JvMLightningModule(JvMBaseLightningModule):
    """
    Lightning Module for JvM (Pure Mamba) architecture.
    
    No external context encoder - uses learned context tokens.
    
    Args:
        # Model architecture
        model_dim: Model dimension
        time_dim: Time dimension
        num_blocks: Number of JvM blocks
        patch_size: Patch size (T, H, W)
        bottleneck_dim: Bottleneck dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution size
        expand: Mamba expansion factor
        dropout: Dropout rate
        context_tokens: Number of learned context tokens
        
        # Other args inherited from base class
    """
    
    def __init__(
        self,
        # Data dimensions
        in_channels: int = 27,
        latent_channels: int = 4,
        t_in: int = 8,
        t_out: int = 24,
        
        # Model architecture
        model_dim: int = 256,
        time_dim: int = 512,
        num_blocks: int = 12,
        patch_size: list = [2, 16, 16],
        bottleneck_dim: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,

        # Training
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        sigma_min: float = 0.0,
        loss_type: str = "v",
        prediction_mode: str = "x",

        
        # Inference
        num_inference_steps: int = 50,
        ode_solver: str = "dopri5",
        test_method: Optional[str] = None,  # Or "full" for full ODE test on the whole test set
        test_num_ensemble: Optional[int] = 10,
        test_threshold: Optional[float] = 0.13, # for detection metrics (derived from Bayesian Analysis of trajectories)

        
        # Visualization
        visualization_params: Optional[list[tuple[int, str]]] = None,
        viz_sample_noise: Optional[list[Optional[float]]] = None,
        
        **kwargs,
    ):
        # Store model-specific params before calling super().__init__
        self.model_dim = model_dim
        self.time_dim = time_dim
        self.num_blocks = num_blocks
        self.patch_size = tuple(patch_size)
        self.bottleneck_dim = bottleneck_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.num_classes = in_channels  # Use in_channels as number of classes (10 for MNIST)
        
        super().__init__(
            in_channels=in_channels,
            latent_channels=latent_channels,
            t_in=t_in,
            t_out=t_out,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            sigma_min=sigma_min,
            loss_type=loss_type,
            prediction_mode=prediction_mode,
            num_inference_steps=num_inference_steps,
            ode_solver=ode_solver,
            visualization_params=visualization_params,
            test_method=test_method,
            test_num_ensemble=test_num_ensemble,
            test_threshold=test_threshold,
            viz_sample_noise=viz_sample_noise,
            **kwargs,
        )
    
    def _build_velocity_net(self):
        """Build JvM model."""
        return JvMModel(
            in_channels=self.latent_channels,
            num_classes=self.num_classes,
            model_dim=self.model_dim,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            patch_size=self.patch_size,
            bottleneck_dim=self.bottleneck_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout        )
    
    def _prepare_model_inputs(self, x_t: torch.Tensor, t: torch.Tensor, x_cond_raw: torch.Tensor):
        """
        Prepare inputs for JvM (uses class embedding internally).
        
        Args:
            x_t: Noisy latent (B, T_out, C_latent, H, W)
            t: Time steps (B,)
            x_cond_raw: Class labels (B,) as long tensors from dataset
            
        Returns:
            Dict with x_t, t, and class_labels
        """
        # Dataset now returns labels directly as (B,) long tensors
        return {"x_t": x_t, "t": t, "class_labels": x_cond_raw}


# ==============================================================================
# JvTM Lightning Module (Contextual Transformer-Mamba)
# ==============================================================================

class JvTMLightningModule(JvMBaseLightningModule):
    """
    Lightning Module for JvTM (Contextual Transformer-Mamba) architecture.
    
    Uses ViTEncoder3D to encode raw conditions into context tokens.
    
    Args:
        # Context encoder
        patch_size_encoder: Patch size for ViT encoder (T, H, W)
        embed_dim: Encoder embedding dimension
        depth: Encoder depth
        num_heads_encoder: Number of heads in encoder
        mlp_ratio: MLP ratio in encoder
        
        # Model architecture
        model_dim: Model dimension
        context_dim: Context dimension (should match embed_dim)
        time_dim: Time dimension
        num_blocks: Number of JvTM blocks
        num_heads: Number of heads in model
        patch_size: Patch size for velocity net (T, H, W)
        bottleneck_dim: Bottleneck dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution size
        expand: Mamba expansion factor
        dropout: Dropout rate
        max_t, max_h, max_w: Max dimensions for RoPE3D
        
        # Other args inherited from base class
    """
    
    def __init__(
        self,
        # Data dimensions
        in_channels: int = 27,
        latent_channels: int = 4,
        t_in: int = 8,
        t_out: int = 24,
        
        # Context encoder
        patch_size_encoder: list = [2, 4, 4],
        embed_dim: int = 256,
        depth: int = 6,
        num_heads_encoder: int = 8,
        mlp_ratio: float = 4.0,
        
        # Model architecture
        model_dim: int = 256,
        context_dim: int = 256,
        time_dim: int = 512,
        num_blocks: int = 12,
        num_heads: int = 8,
        patch_size: list = [2, 16, 16],
        bottleneck_dim: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        max_t: int = 100,
        max_h: int = 64,
        max_w: int = 152,
        
        # Training
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        sigma_min: float = 0.0,
        loss_type: str = "v",
        prediction_mode: str = "x",
        
        # Inference
        num_inference_steps: int = 50,
        ode_solver: str = "dopri5",
        test_method: Optional[str] = None,  # Or "full" for full ODE test on the whole test set
        test_num_ensemble: Optional[int] = 10,
        test_threshold: Optional[float] = 0.13, # for detection metrics (derived from Bayesian Analysis of trajectories)

        
        # Visualization
        visualization_params: Optional[list[tuple[int, str]]] = None,
        viz_sample_noise: Optional[list[Optional[float]]] = None,
        
        **kwargs,
    ):
        # Store encoder-specific params
        self.patch_size_encoder = tuple(patch_size_encoder)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads_encoder = num_heads_encoder
        self.mlp_ratio = mlp_ratio
        
        # Store model-specific params
        self.model_dim = model_dim
        self.context_dim = context_dim
        self.time_dim = time_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.patch_size = tuple(patch_size)
        self.bottleneck_dim = bottleneck_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        
        super().__init__(
            in_channels=in_channels,
            latent_channels=latent_channels,
            t_in=t_in,
            t_out=t_out,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            sigma_min=sigma_min,
            loss_type=loss_type,
            prediction_mode=prediction_mode,
            num_inference_steps=num_inference_steps,
            ode_solver=ode_solver,
            visualization_params=visualization_params,
            viz_sample_noise=viz_sample_noise,
            test_method=test_method,
            test_num_ensemble=test_num_ensemble,
            test_threshold=test_threshold,
            **kwargs,
        )
        
        # Build context encoder
        self.context_encoder = ViTEncoder3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads_encoder,
            patch_size=self.patch_size_encoder,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
    
    def _build_velocity_net(self):
        """Build JvTM model."""
        return JvTMModel(
            in_channels=self.latent_channels,
            model_dim=self.model_dim,
            context_dim=self.context_dim,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            bottleneck_dim=self.bottleneck_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout,
            max_t=self.max_t,
            max_h=self.max_h,
            max_w=self.max_w,
        )
    
    def _prepare_model_inputs(self, x_t: torch.Tensor, t: torch.Tensor, x_cond_raw: torch.Tensor):
        """
        Prepare inputs for JvTM (encode raw conditions with ViT).
        
        Args:
            x_t: Noisy latent (B, T_out, C_latent, H, W)
            t: Time steps (B,)
            x_cond_raw: Raw conditions (B, T_in, C_in, H, W)
            
        Returns:
            Dict with x_t, context, and t
        """
        # Encode conditions to context tokens
        context = self.context_encoder(x_cond_raw)  # (B, N, context_dim)
        
        # Classifier-Free Guidance: Randomly drop context during training
        if self.training and self.cond_dropout_prob > 0:
            B = context.shape[0]
            drop_mask = torch.rand(B, device=context.device) < self.cond_dropout_prob
            # Zero out context where drop_mask is True
            context = context * (~drop_mask).view(B, 1, 1).float()
        
        return {"x_t": x_t, "context": context, "t": t}


# ==============================================================================
# SPADEJvM Lightning Module (SPADE-conditioned Mamba)
# ==============================================================================

class SPADEJvMLightningModule(JvMBaseLightningModule):
    """
    Lightning Module for SPADEJvM (Spatial Adaptive Just video Mamba) architecture.
    
    Uses SPADE (Spatial Adaptive Normalization) instead of Cross-Attention for strong
    spatial conditioning. Includes ViT encoder to process complex conditions.
    
    Philosophy:
    - "Just Image" prediction (predicts clean x, not epsilon or velocity)
    - ViT encoder compresses conditions into tokens (handles complex multi-variable data)
    - Tokens projected back to spatial format for SPADE
    - SPADE for multiplicative spatial conditioning (cannot be ignored)
    
    Args:
        # Context encoder
        use_encoder: Whether to use ViT encoder (recommended for complex conditions)
        patch_size_encoder: Patch size for ViT encoder (T, H, W)
        embed_dim: Encoder embedding dimension
        depth: Encoder depth
        num_heads_encoder: Number of heads in encoder
        mlp_ratio: MLP ratio in encoder
        
        # Model architecture
        model_dim: Model dimension
        cond_channels: Spatial condition channels (after projection from encoder)
        time_dim: Time dimension
        num_blocks: Number of SPADEJvM blocks
        patch_size: Patch size (T, H, W)
        bottleneck_dim: Bottleneck dimension
        d_state: Mamba state dimension
        d_conv: Mamba convolution size
        expand: Mamba expansion factor
        dropout: Dropout rate
        
        # Other args inherited from base class, notably:
        - prediction_mode: Should be "x" for SPADEJvM, "v" for CFM with the same model.
        - loss_type: Should be "v" for velocity loss, "x" for direct x loss with x prediction.
        - sample_t_distrib: Sampling distribution for t ("uniform" or "logit-normal")
        - logit_normal_loc, logit_normal_scale: Parameters for logit-normal sampling
    """
    
    def __init__(
        self,
        # Data dimensions
        in_channels: int = 1,  # Moving MNIST: 1 channel, AOD: 27 variables
        latent_channels: int = 1,
        t_in: int = 10,
        t_out: int = 10,
        
        
        # Model architecture
        model_dim: int = 256,
        cond_channels: int = 128,  # Channels for SPADE (projected from encoder)
        time_dim: int = 512,
        num_blocks: int = 12,
        patch_size: list = [2, 16, 16],
        bottleneck_dim: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,

        # Args pour le Context Encoder
        use_encoder: bool = True,
        # cond_channels: int = 64,          # Output du ContextNet (input du SPADE)
        cond_encoder_hidden_dim: int = 128, # <--- Le fameux paramètre interne
        cond_encoder_depth: int = 3,      # Nombre de layers ResNet
        cond_dropout: float = 0.0,
        
        # Training
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        sigma_min: float = 0.0,
        loss_type: str = "v",
        prediction_mode: str = "x",
        sample_t_distrib: str = "uniform",
        logit_normal_loc: float = 0.0,
        logit_normal_scale: float = 1.0,
        
        # Classifier-Free Guidance
        cfg_dropout_prob: float = 0.0,
        guidance_scale: float = 1.0,
        
        # Optimal Transport
        use_ot_sampling: bool = False,
        ot_method: str = "exact",
        ot_reg: float = 0.05,
        
        # Inference
        num_inference_steps: int = 50,
        ode_solver: str = "dopri5",
        test_method: Optional[str] = None,  # Or "full" for full ODE test on the whole test set
        test_num_ensemble: Optional[int] = 10,
        test_threshold: Optional[float] = 0.13, # for detection metrics (derived from Bayesian Analysis of trajectories)
        
        # Visualization
        visualization_params: Optional[list[tuple[int, str]]] = None,
        viz_sample_noise: Optional[list[Optional[float]]] = None,
        
        **kwargs,
    ):
        # Store encoder-specific params
        self.use_encoder = use_encoder
        
        # Store model-specific params
        self.model_dim = model_dim
        self.cond_channels = cond_channels
        self.time_dim = time_dim
        self.num_blocks = num_blocks
        self.patch_size = tuple(patch_size)
        self.bottleneck_dim = bottleneck_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dropout = dropout
        
        super().__init__(
            in_channels=in_channels,
            latent_channels=latent_channels,
            t_in=t_in,
            t_out=t_out,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            sigma_min=sigma_min,
            loss_type=loss_type,
            prediction_mode=prediction_mode,
            sample_t_distrib=sample_t_distrib,
            logit_normal_loc=logit_normal_loc,
            logit_normal_scale=logit_normal_scale,
            cond_dropout_prob=cfg_dropout_prob,
            guidance_scale=guidance_scale,
            use_ot_sampling=use_ot_sampling,
            ot_method=ot_method,
            ot_reg=ot_reg,
            num_inference_steps=num_inference_steps,
            ode_solver=ode_solver,
            visualization_params=visualization_params,
            viz_sample_noise=viz_sample_noise,
            test_method=test_method,
            test_num_ensemble=test_num_ensemble,
            test_threshold=test_threshold,
            **kwargs,
        )
        
        # Build ContextCNN encoder if enabled
        if self.use_encoder:
            self.context_encoder = ContextNet(
                in_channels=self.hparams.in_channels, # ex: 27
                t_in=self.hparams.t_in,               # ex: 8
                out_channels=cond_channels,           # ex: 64
                hidden_dim=cond_encoder_hidden_dim,   # <--- On le passe ici
                num_layers=cond_encoder_depth,
                dropout=cond_dropout
            )
        else:
            self.context_encoder = None
    
    def _build_velocity_net(self):
        """Build SPADEJvM model with past+future support."""
        from models.jvm.jvm_spade import SPADEJvMModel
        
        return SPADEJvMModel(
            in_channels=self.latent_channels,
            model_dim=self.model_dim,
            cond_channels=self.cond_channels,
            time_dim=self.time_dim,
            num_blocks=self.num_blocks,
            patch_size=self.patch_size,
            bottleneck_dim=self.bottleneck_dim,
            t_in=self.t_in,
            t_out=self.t_out,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout,
        )
    
    def _prepare_model_inputs(self, x_t: torch.Tensor, t: torch.Tensor, x_cond_raw: torch.Tensor, x_past: Optional[torch.Tensor] = None):
        """
        Prepare inputs for SPADEJvM with past+future strategy.
        
        Args:
            x_t: Noisy future latent (B, T_out, C_latent, H, W)
            t: Time steps (B,)
            x_cond_raw: Raw conditions (B, T_in, C_in, H, W)
            x_past: Clean past latent (B, T_in, C_latent, H, W) or None
            
        Returns:
            Dict with x_t, cond_spatial, t, and x_past
        """
        B = x_t.shape[0]
        
        if self.use_encoder:
            # Use ContextCNN: (B, T_in, C_in, H, W) → (B, cond_channels, H, W)
            cond_spatial = self.context_encoder(x_cond_raw)  # (B, cond_channels, H, W)
        else:
            # Fallback: direct flatten (for simple cases like Moving MNIST)
            T_in, C_in, H, W = x_cond_raw.shape[1:]
            cond_spatial = rearrange(x_cond_raw, 'b t c h w -> b (t c) h w')
        
        # Classifier-Free Guidance: Randomly drop condition during training
        if self.training and self.cond_dropout_prob > 0:
            drop_mask = torch.rand(B, device=cond_spatial.device) < self.cond_dropout_prob
            cond_spatial = cond_spatial * (~drop_mask).view(B, 1, 1, 1).float()
        
        return {"x_t": x_t, "cond_spatial": cond_spatial, "t": t, "x_past": x_past}
