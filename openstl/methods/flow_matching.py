# Copyright (c) CAIRI AI Lab. All rights reserved
# Flow Matching Method for OpenSTL
# Ported from MambaFlow/JvM implementation

"""
FlowMatching Method for Spatiotemporal Prediction.

Implements the "Just Image" (JvM) philosophy from:
"Back to Basics: Let Denoising Generative Models Denoise" (Li & He, 2026)

Key principle: Models predict clean images (x) directly, not noise (epsilon) or velocity (v).
This keeps predictions on the natural image manifold and significantly improves generation quality.

Supports:
- prediction_mode: "x" (JvM), "v" (CFM), "epsilon" (Diffusion)
- loss_type: "v" (velocity loss, best), "x" (pixel loss)
- OT sampling for optimal transport trajectories
- Multiple ODE solvers for inference
- Time sampling distributions (uniform, logit_normal)
- Classifier-Free Guidance (CFG)
"""

import numpy as np
import torch
import torch.nn as nn
import os.path as osp
from typing import Literal, Optional

from openstl.methods.base_method import Base_method

# Constants for Flow Matching (from Li & He, 2026, Section 4.3)
TIME_MIN_EPSILON = 1e-3  # Minimum time to avoid numerical instability
TIME_MAX_OFFSET = 0.05   # Offset from t=1 for stable training with x-prediction
from openstl.models import SPADEJvM_Model
from openstl.utils import print_log, check_dir
from openstl.core import metric

# Optional dependencies with graceful fallback
try:
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
    TORCHCFM_AVAILABLE = True
except ImportError:
    TORCHCFM_AVAILABLE = False
    ConditionalFlowMatcher = None

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None

try:
    from torchcfm.optimal_transport import OTPlanSampler
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
    OTPlanSampler = None


class FlowMatching(Base_method):
    """
    Flow Matching Method for Spatiotemporal Video Prediction.
    
    Implements the "Just Image" approach with optional velocity/epsilon prediction modes.
    
    Args:
        prediction_mode: Target prediction type
            - "x": Predict clean image directly (JvM, best)
            - "v": Predict velocity (standard CFM)
            - "epsilon": Predict noise (diffusion-style)
        loss_type: Loss computation space
            - "v": Velocity loss (best, implicit time weighting)
            - "x": Pixel-space loss
        sigma_min: Minimum noise level for CFM interpolation
        ode_solver: ODE solver for inference ("dopri5", "rk4", "euler", "midpoint")
        num_inference_steps: Number of ODE integration steps
        sample_t_distrib: Time sampling distribution ("uniform", "logit_normal")
        logit_normal_loc: Location parameter for logit-normal distribution
        logit_normal_scale: Scale parameter for logit-normal distribution
        use_ot_sampling: Enable Optimal Transport sampling
        ot_method: OT method ("exact", "sinkhorn")
        ot_reg: OT regularization for Sinkhorn
        cond_dropout_prob: Condition dropout probability for CFG
        guidance_scale: Guidance scale for CFG sampling (>1.0 amplifies condition)
    """
    
    def __init__(
        self,
        # Flow Matching parameters
        prediction_mode: Literal["x", "v", "epsilon"] = "x",
        loss_type: Literal["v", "x"] = "v",
        sigma_min: float = 0.0,
        ode_solver: str = "dopri5",
        num_inference_steps: int = 50,
        
        # Time sampling
        sample_t_distrib: str = "uniform",
        logit_normal_loc: float = 0.0,
        logit_normal_scale: float = 1.0,
        
        # Optimal Transport
        use_ot_sampling: bool = False,
        ot_method: str = "exact",
        ot_reg: float = 0.05,
        
        # Classifier-Free Guidance
        cond_dropout_prob: float = 0.0,
        guidance_scale: float = 1.0,
        
        **args,
    ):
        # Validate dependencies
        if not TORCHCFM_AVAILABLE:
            raise ImportError(
                "torchcfm is required for FlowMatching. "
                "Install with: pip install torchcfm"
            )
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError(
                "torchdiffeq is required for FlowMatching. "
                "Install with: pip install torchdiffeq"
            )
        
        # Store flow matching parameters before super().__init__
        self._prediction_mode = prediction_mode
        self._loss_type = loss_type
        self._sigma_min = sigma_min
        self._ode_solver = ode_solver
        self._num_inference_steps = num_inference_steps
        self._sample_t_distrib = sample_t_distrib
        self._logit_normal_loc = logit_normal_loc
        self._logit_normal_scale = logit_normal_scale
        self._use_ot_sampling = use_ot_sampling and OT_AVAILABLE
        self._ot_method = ot_method
        self._ot_reg = ot_reg
        self._cond_dropout_prob = cond_dropout_prob
        self._guidance_scale = guidance_scale
        
        # Enforce loss_type consistency for v prediction
        if prediction_mode == "v":
            self._loss_type = "v"
        
        # Initialize base
        super().__init__(**args)
        
        # Store in hparams
        self.save_hyperparameters(
            'prediction_mode', 'loss_type', 'sigma_min', 'ode_solver',
            'num_inference_steps', 'sample_t_distrib', 'logit_normal_loc',
            'logit_normal_scale', 'use_ot_sampling', 'ot_method', 'ot_reg',
            'cond_dropout_prob', 'guidance_scale'
        )
        
        # Flow Matching components
        self.fm = ConditionalFlowMatcher(sigma=sigma_min)
        
        # OT Sampler (optional)
        if self._use_ot_sampling:
            self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg)
        else:
            self.ot_sampler = None
        
        # Time endpoint for sampling
        self.t_end = 1.0 - 1e-3 if prediction_mode == "x" else 1.0
        
        # Test outputs storage
        self.test_outputs = []
    
    def _build_model(self, **args):
        """Build SPADEJvM model."""
        # Get model parameters from args
        pre_seq_length = args.get('pre_seq_length', 10)
        aft_seq_length = args.get('aft_seq_length', 10)
        in_shape = args.get('in_shape', (pre_seq_length, 1, 64, 64))
        
        # Model configuration
        model_config = {
            'in_shape': in_shape,
            'in_channels': args.get('in_channels', in_shape[1]),
            'model_dim': args.get('model_dim', args.get('hid_S', 256)),
            'cond_channels': args.get('cond_channels', 10),
            'time_dim': args.get('time_dim', 512),
            'num_blocks': args.get('num_blocks', args.get('N_T', 12)),
            'patch_size': args.get('patch_size', (2, 8, 8)),
            'bottleneck_dim': args.get('bottleneck_dim', 128),
            'pre_seq_length': pre_seq_length,
            'aft_seq_length': aft_seq_length,
            'block_type': args.get('block_type', 'mamba'),
            'use_spade': args.get('use_spade', True),
            'use_rope': args.get('use_rope', True),
            'd_state': args.get('d_state', 64),
            'd_conv': args.get('d_conv', 4),
            'expand': args.get('expand', 2),
            'num_heads': args.get('num_heads', 8),
            'dropout': args.get('drop', 0.0),
            'gradient_checkpointing': args.get('gradient_checkpointing', True),
        }
        
        return SPADEJvM_Model(**model_config)
    
    def forward(self, batch_x, batch_y=None, **kwargs):
        """
        Forward pass for inference.
        
        Args:
            batch_x: Input/condition sequence (B, T_in, C, H, W)
            batch_y: Target sequence (B, T_out, C, H, W) for shape inference
            
        Returns:
            Predicted sequence (B, T_out, C, H, W)
        """
        return self.sample(batch_x, target_shape=batch_y.shape if batch_y is not None else None)
    
    def training_step(self, batch, batch_idx):
        """
        Training step with Flow Matching.
        
        Implements "Back to Basics: Let Denoising Generative Models Denoise":
        - Model predicts x (clean image) directly
        - Loss can be computed in velocity space (v-loss) or pixel space (x-loss)
        - v-loss provides implicit time-dependent weighting
        """
        batch_x, batch_y = batch  # (B, T_in, C, H, W), (B, T_out, C, H, W)
        B = batch_y.shape[0]
        device = batch_y.device
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(batch_y)
        
        # Optimal Transport: permute x_0 to minimize transport cost
        if self._use_ot_sampling and self.ot_sampler is not None:
            x_0, _ = self.ot_sampler.sample_plan(x_0, batch_y)
        
        # Classifier-Free Guidance: randomly drop condition
        cond = batch_x.clone()
        if self._cond_dropout_prob > 0:
            drop_mask = torch.rand(B, device=device) < self._cond_dropout_prob
            cond = cond * (~drop_mask).view(B, 1, 1, 1, 1).float()
        
        # Sample time and interpolate
        if self._sample_t_distrib == "logit_normal":
            t = torch.randn(B, device=device)
            t = t * self._logit_normal_scale + self._logit_normal_loc
            t = torch.sigmoid(t)
            t = torch.clamp(t, min=TIME_MIN_EPSILON, max=1.0 - TIME_MAX_OFFSET)
            t, x_t, v_target = self.fm.sample_location_and_conditional_flow(x_0, batch_y, t=t)
        else:
            t, x_t, v_target = self.fm.sample_location_and_conditional_flow(x_0, batch_y, t=None)
        
        # Model prediction
        if self._prediction_mode == "x":
            # Predict clean image directly (JvM)
            x_pred = self.model(x_t, cond, t)
            # Derive velocity from x prediction
            v_pred = self.model.compute_v_from_x_pred(x_pred, x_t, t)
        elif self._prediction_mode == "v":
            # Predict velocity directly (CFM)
            v_pred = self.model(x_t, cond, t)
            # Derive x from v prediction
            t_view = t.view(B, 1, 1, 1, 1)
            x_pred = x_t + v_pred * (1 - t_view)
        else:  # epsilon
            # Predict noise (diffusion-style)
            eps_pred = self.model(x_t, cond, t)
            # Convert to x and v
            t_view = t.view(B, 1, 1, 1, 1)
            x_pred = (x_t - (1 - t_view) * eps_pred) / t_view.clamp(min=1e-5)
            v_pred = self.model.compute_v_from_x_pred(x_pred, x_t, t)
        
        # Compute losses
        v_loss = nn.functional.mse_loss(v_pred, v_target)
        x_loss = nn.functional.mse_loss(x_pred, batch_y)
        
        # Select loss based on config
        if self._loss_type == "v":
            loss = v_loss
        else:
            loss = x_loss
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_v_loss', v_loss, on_step=False, on_epoch=True)
        self.log('train_x_loss', x_loss, on_step=False, on_epoch=True)
        self.log('train_t_mean', t.mean(), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step using training logic."""
        batch_x, batch_y = batch
        B = batch_y.shape[0]
        device = batch_y.device
        
        # Sample and compute loss
        x_0 = torch.randn_like(batch_y)
        t, x_t, v_target = self.fm.sample_location_and_conditional_flow(x_0, batch_y, t=None)
        
        if self._prediction_mode == "x":
            x_pred = self.model(x_t, batch_x, t)
            v_pred = self.model.compute_v_from_x_pred(x_pred, x_t, t)
        elif self._prediction_mode == "v":
            v_pred = self.model(x_t, batch_x, t)
            t_view = t.view(B, 1, 1, 1, 1)
            x_pred = x_t + v_pred * (1 - t_view)
        else:
            eps_pred = self.model(x_t, batch_x, t)
            t_view = t.view(B, 1, 1, 1, 1)
            x_pred = (x_t - (1 - t_view) * eps_pred) / t_view.clamp(min=1e-5)
            v_pred = self.model.compute_v_from_x_pred(x_pred, x_t, t)
        
        v_loss = nn.functional.mse_loss(v_pred, v_target)
        x_loss = nn.functional.mse_loss(x_pred, batch_y)
        loss = v_loss if self._loss_type == "v" else x_loss
        
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step with ODE sampling."""
        batch_x, batch_y = batch
        
        # Sample using ODE
        pred_y = self.sample(batch_x, target_shape=batch_y.shape)
        
        outputs = {
            'inputs': batch_x.cpu().numpy(),
            'preds': pred_y.cpu().numpy(),
            'trues': batch_y.cpu().numpy()
        }
        self.test_outputs.append(outputs)
        return outputs
    
    def sample(
        self,
        cond: torch.Tensor,
        target_shape: Optional[tuple] = None,
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Sample from the model using ODE integration.
        
        Args:
            cond: Condition sequence (B, T_in, C, H, W)
            target_shape: Target output shape (B, T_out, C, H, W)
            num_steps: Number of ODE steps
            solver: ODE solver ('dopri5', 'rk4', 'euler', 'midpoint')
            guidance_scale: Override guidance scale
            return_trajectory: Return full ODE trajectory
            
        Returns:
            Sampled sequence (B, T_out, C, H, W) or trajectory
        """
        if num_steps is None:
            num_steps = self._num_inference_steps
        if solver is None:
            solver = self._ode_solver
        if guidance_scale is None:
            guidance_scale = self._guidance_scale
        
        B = cond.shape[0]
        device = cond.device
        
        # Determine output shape
        if target_shape is not None:
            _, T_out, C, H, W = target_shape
        else:
            T_out = self.hparams.get('aft_seq_length', self.hparams.get('pre_seq_length', cond.shape[1]))
            C, H, W = cond.shape[2], cond.shape[3], cond.shape[4]
        
        # Initial noise
        z0 = torch.randn(B, T_out, C, H, W, device=device, dtype=torch.float32)
        
        # Convert condition to FP32
        cond_fp32 = cond.to(dtype=torch.float32)
        
        # ODE function
        def ode_func(t_scalar, x):
            t_val = t_scalar.item() if isinstance(t_scalar, torch.Tensor) else float(t_scalar)
            t_batch = torch.full((B,), t_val, device=device, dtype=x.dtype)
            
            # Conditional prediction
            if self._prediction_mode == "x":
                x_pred = self.model(x, cond_fp32, t_batch)
                v_cond = self.model.compute_v_from_x_pred(x_pred, x, t_batch)
            elif self._prediction_mode == "v":
                v_cond = self.model(x, cond_fp32, t_batch)
            else:  # epsilon
                eps_pred = self.model(x, cond_fp32, t_batch)
                t_view = t_batch.view(B, 1, 1, 1, 1)
                x_pred = (x - (1 - t_view) * eps_pred) / t_view.clamp(min=1e-5)
                v_cond = self.model.compute_v_from_x_pred(x_pred, x, t_batch)
            
            # Classifier-Free Guidance
            if guidance_scale != 1.0:
                cond_uncond = torch.zeros_like(cond_fp32)
                
                if self._prediction_mode == "x":
                    x_pred_uncond = self.model(x, cond_uncond, t_batch)
                    v_uncond = self.model.compute_v_from_x_pred(x_pred_uncond, x, t_batch)
                elif self._prediction_mode == "v":
                    v_uncond = self.model(x, cond_uncond, t_batch)
                else:
                    eps_pred_uncond = self.model(x, cond_uncond, t_batch)
                    t_view = t_batch.view(B, 1, 1, 1, 1)
                    x_pred_uncond = (x - (1 - t_view) * eps_pred_uncond) / t_view.clamp(min=1e-5)
                    v_uncond = self.model.compute_v_from_x_pred(x_pred_uncond, x, t_batch)
                
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
                return v
            
            return v_cond
        
        # Time grid
        t_span = torch.linspace(1e-3, self.t_end, num_steps, device=device, dtype=torch.float32)
        
        # ODE integration
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
        
        if return_trajectory:
            return trajectory
        return trajectory[-1]
    
    def on_test_epoch_end(self):
        """Compute metrics at end of test epoch."""
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
        eval_res, eval_log = metric(
            results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std,
            metrics=self.metric_list,
            channel_names=self.channel_names,
            spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None)
        )
        
        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        
        # Clear test outputs
        self.test_outputs = []
        return results_all
