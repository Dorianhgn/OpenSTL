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
import torch.nn.functional as F
import os.path as osp
from typing import Literal, Optional

from openstl.methods.base_method import Base_method

# Constants for Flow Matching (from Li & He, 2026, Section 4.3)
TIME_MIN_EPSILON = 1e-3  # Minimum time to avoid numerical instability
TIME_MAX_OFFSET = 0.05   # Offset from t=1 for stable training with x-prediction
from openstl.models import SPADEJvM_Model, build_context_encoder
from openstl.utils import print_log, check_dir
from openstl.core import metric, per_frame_metric

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
        test_num_ensemble: int = 5,
        metric_threshold: Optional[float] = 128.0,
        
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

        # Optional label conditioning for MMNIST-style benchmarks
        use_label_conditioner: bool = False,
        label_conditioner_type: str = 'LabelConditioner',
        label_conditioner_params: Optional[dict] = None,
        num_classes: int = 10,
        
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
        self._use_label_conditioner = use_label_conditioner
        self._label_conditioner_type = label_conditioner_type
        self._label_conditioner_params = label_conditioner_params or {}
        self._num_classes = num_classes
        self._use_adaln_label_conditioning = (
            self._use_label_conditioner and str(self._label_conditioner_type).lower() == 'adaln'
        )
        self._use_spatial_label_conditioner = self._use_label_conditioner and not self._use_adaln_label_conditioning
        self._test_num_ensemble = test_num_ensemble
        self._metric_threshold = metric_threshold
        
        # Enforce loss_type consistency for v prediction
        if prediction_mode == "v":
            self._loss_type = "v"
        
        # Initialize base
        args["metric_threshold"] = metric_threshold
        args["test_num_ensemble"] = test_num_ensemble
        super().__init__(**args)
        
        # Store in hparams
        self.save_hyperparameters(
            'prediction_mode', 'loss_type', 'sigma_min', 'ode_solver',
            'num_inference_steps', 'sample_t_distrib', 'logit_normal_loc',
            'logit_normal_scale', 'use_ot_sampling', 'ot_method', 'ot_reg',
            'cond_dropout_prob', 'guidance_scale', 'use_label_conditioner',
            'label_conditioner_type', 'label_conditioner_params', 'num_classes',
            'test_num_ensemble', 'metric_threshold'
        )
        
        # Flow Matching components
        self.fm = ConditionalFlowMatcher(sigma=sigma_min)
        
        # OT Sampler (optional)
        if self._use_ot_sampling:
            self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg)
        else:
            self.ot_sampler = None
        
        # Context Encoder (optional, instantiated after parent init so we have hparams)
        self.context_encoder = None
        if args.get('use_encoder', False):
            encoder_type = args.get('context_encoder_type', 'ContextNet')
            encoder_params = args.get('context_encoder_params', {})
            
            # Get dataset info from hparams
            in_channels = args.get('in_channels', args.get('in_shape', (10, 1, 64, 64))[1])
            pre_seq_length = args.get('pre_seq_length', 10)
            cond_channels = args.get('cond_channels', 10)
            
            # Build context encoder
            self.context_encoder = build_context_encoder(
                encoder_type=encoder_type,
                in_channels=in_channels,
                t_in=pre_seq_length,
                out_channels=cond_channels,
                **encoder_params
            )
            print(f"Initialized {encoder_type} context encoder with params: {encoder_params}")

        # Label Conditioner (optional, used for label-injection benchmarks)
        self.label_conditioner = None
        if self._use_spatial_label_conditioner:
            label_out_channels = args.get('cond_channels', 10)
            label_params = dict(self._label_conditioner_params)
            label_params.setdefault('spatial_size', args.get('label_conditioner_spatial_size', 16))
            label_params.setdefault('base_size', args.get('label_conditioner_base_size', 4))
            self.label_conditioner = build_context_encoder(
                encoder_type=self._label_conditioner_type,
                num_classes=self._num_classes,
                out_channels=label_out_channels,
                **label_params,
            )
            print(
                f"Initialized {self._label_conditioner_type} label conditioner "
                f"for {self._num_classes} classes with params: {label_params}"
            )

        # AdaLN-style global label embedding (optional)
        self.label_embedding = None
        self.label_mlp = None
        if self._use_adaln_label_conditioning:
            label_params = dict(self._label_conditioner_params)
            label_dim = int(label_params.get('label_dim', 128))
            time_dim = args.get('time_dim', 512)
            self.label_embedding = nn.Embedding(self._num_classes, label_dim)
            self.label_mlp = nn.Sequential(
                nn.Linear(label_dim, label_dim),
                nn.SiLU(),
                nn.Linear(label_dim, time_dim),
            )
            print(
                f"Initialized AdaLN label conditioner for {self._num_classes} classes "
                f"with label_dim={label_dim}, time_dim={time_dim}"
            )
        
        # Time endpoint for sampling
        self.t_end = 1.0 - 1e-3 if prediction_mode == "x" else 1.0
        
        # Test outputs storage
        self.test_outputs = []

    def _split_batch(self, batch):
        """Handle either (x, y) or (x, y, labels) batches without breaking SimVP."""
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0], batch[1], None
            if len(batch) == 3:
                return batch[0], batch[1], batch[2]
        raise ValueError(
            f"Unsupported batch format: type={type(batch)}, len={len(batch) if hasattr(batch, '__len__') else 'NA'}"
        )
    
    def _build_model(self, **args):
        """Build SPADEJvM model."""
        # Get model parameters from args
        pre_seq_length = args.get('pre_seq_length', 10)
        aft_seq_length = args.get('aft_seq_length', 10)
        in_shape = args.get('in_shape', (pre_seq_length, 1, 64, 64))
        
        use_spade = args.get('use_spade', True) or self._use_spatial_label_conditioner
        if self._use_adaln_label_conditioning:
            # AdaLN label conditioning is injected via time embedding, no spatial SPADE map.
            use_spade = False

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
            'use_spade': use_spade,
            'use_rope': args.get('use_rope', True),
            'd_state': args.get('d_state', 64),
            'd_conv': args.get('d_conv', 4),
            'expand': args.get('expand', 2),
            'num_heads': args.get('num_heads', 8),
            'dropout': args.get('drop', 0.0),
            'gradient_checkpointing': args.get('gradient_checkpointing', True),
        }
        
        return SPADEJvM_Model(**model_config)

    def _labels_to_one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert integer or multi-hot labels to float one-hot/multi-hot vectors."""
        if labels.ndim == 1:
            labels = F.one_hot(labels.long(), num_classes=self._num_classes)
        elif labels.ndim != 2:
            raise ValueError(f"Unsupported labels shape for AdaLN: {tuple(labels.shape)}")
        if labels.shape[-1] != self._num_classes:
            raise ValueError(
                f"Expected labels last dim={self._num_classes}, got {labels.shape[-1]}"
            )
        return labels.float()

    def _get_label_emb(self, batch_labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Build global label embedding (B, time_dim) for AdaLN conditioning."""
        if not self._use_adaln_label_conditioning or self.label_embedding is None or self.label_mlp is None:
            return None
        if batch_labels is None:
            return None

        labels = batch_labels.to(device=self.label_embedding.weight.device)
        labels = self._labels_to_one_hot(labels)
        label_feat = labels @ self.label_embedding.weight
        return self.label_mlp(label_feat)
    
    def forward(self, batch_x, batch_y=None, **kwargs):
        """
        Forward pass for inference.
        
        Args:
            batch_x: Input/condition sequence (B, T_in, C, H, W)
            batch_y: Target sequence (B, T_out, C, H, W) for shape inference
            
        Returns:
            Predicted sequence (B, T_out, C, H, W)
        """
        return self.sample(
            batch_x,
            target_shape=batch_y.shape if batch_y is not None else None,
            labels=kwargs.get('labels', None),
        )
    
    def _prepare_model_inputs(self, x_t, t, batch_x, batch_labels=None, apply_cond_dropout=False):
        """
        Prepare inputs for the SPADEJvM model.
        
        Logic:
        - x_past: Always batch_x (clean past frames)
        - cond_spatial: 
            * If use_encoder=True: batch_x → ContextNet → cond_spatial
            * If use_encoder=False and use_spade=True: batch_x flattened → cond_spatial
            * If use_spade=False: cond_spatial=None (concatenation mode)
        
        Args:
            x_t: Noisy future frames (B, T_out, C, H, W)
            t: Time steps (B,)
            batch_x: Past frames (B, T_in, C, H, W)
            batch_labels: Optional sequence labels (B, num_classes)
            apply_cond_dropout: Whether to apply CFG dropout
            
        Returns:
            Dict with keys: x_t, t, x_past, cond_spatial, label_emb
        """
        B = x_t.shape[0]
        device = x_t.device
        
        x_past = batch_x
        cond_spatial = None
        label_emb = self._get_label_emb(batch_labels)
        
        # Prepare SPADE condition if use_spade=True
        use_spade = self.hparams.get('use_spade', True) or self._use_spatial_label_conditioner
        if self._use_adaln_label_conditioning:
            use_spade = False
        if use_spade:
            if batch_labels is not None and self.label_conditioner is not None:
                cond_spatial = self.label_conditioner(batch_labels)
            elif self.context_encoder is not None:
                # Use ContextNet: (B, T, C, H, W) → (B, cond_channels, H, W)
                cond_spatial = self.context_encoder(batch_x)
            else:
                # Fallback: flatten temporal dimension into channels
                # (B, T, C, H, W) → (B, T*C, H, W)
                from einops import rearrange
                cond_spatial = rearrange(batch_x, 'b t c h w -> b (t c) h w')
        
        # Classifier-Free Guidance: randomly drop conditioning
        if apply_cond_dropout and self._cond_dropout_prob > 0:
            drop_mask = torch.rand(B, device=device) < self._cond_dropout_prob
            # Drop both x_past and cond_spatial
            x_past = x_past * (~drop_mask).view(B, 1, 1, 1, 1).float()
            if cond_spatial is not None:
                cond_spatial = cond_spatial * (~drop_mask).view(B, 1, 1, 1).float()
            if label_emb is not None:
                label_emb = label_emb * (~drop_mask).view(B, 1).float()
        
        return {
            "x_t": x_t,
            "t": t,
            "x_past": x_past,
            "cond_spatial": cond_spatial,
            "label_emb": label_emb,
        }
    
    def _call_model(self, x_t, t, x_past=None, cond_spatial=None, label_emb=None):
        """Call model with the correct signature (x_t, t, cond_spatial, x_past)."""
        return self.model(x_t, t, cond_spatial=cond_spatial, x_past=x_past, label_emb=label_emb)
    
    def _call_model_from_inputs(self, model_inputs):
        """Call model using prepared inputs dict from _prepare_model_inputs."""
        return self._call_model(
            model_inputs["x_t"], model_inputs["t"],
            x_past=model_inputs["x_past"],
            cond_spatial=model_inputs["cond_spatial"],
            label_emb=model_inputs.get("label_emb", None),
        )
    
    def training_step(self, batch, batch_idx):
        """
        Training step with Flow Matching.
        
        Implements "Back to Basics: Let Denoising Generative Models Denoise":
        - Model predicts x (clean image) directly
        - Loss can be computed in velocity space (v-loss) or pixel space (x-loss)
        - v-loss provides implicit time-dependent weighting
        
        In OpenSTL, batch_x = past frames, batch_y = future frames.
        Past frames are always passed as x_past (concatenated in time dim).
        SPADE spatial condition is derived from x_past inside the model.
        """
        batch_x, batch_y, batch_labels = self._split_batch(batch)  # (B, T_in, C, H, W), (B, T_out, C, H, W), labels optional
        B = batch_y.shape[0]
        device = batch_y.device
        
        # Sample noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(batch_y)
        
        # Optimal Transport: permute x_0 to minimize transport cost
        if self._use_ot_sampling and self.ot_sampler is not None:
            x_0, _ = self.ot_sampler.sample_plan(x_0, batch_y)
        
        # Sample time and interpolate
        if self._sample_t_distrib == "logit_normal":
            t = torch.randn(B, device=device)
            t = t * self._logit_normal_scale + self._logit_normal_loc
            t = torch.sigmoid(t)
            t = torch.clamp(t, min=TIME_MIN_EPSILON, max=1.0 - TIME_MAX_OFFSET)
            t, x_t, v_target = self.fm.sample_location_and_conditional_flow(x_0, batch_y, t=t)
        else:
            t, x_t, v_target = self.fm.sample_location_and_conditional_flow(x_0, batch_y, t=None)
        
        # Prepare model inputs (centralizes x_past / cond logic + CFG dropout)
        model_inputs = self._prepare_model_inputs(
            x_t, t, batch_x, batch_labels=batch_labels, apply_cond_dropout=True
        )
        
        # Model prediction
        if self._prediction_mode == "x":
            # Predict clean image directly (JvM)
            x_pred = self._call_model_from_inputs(model_inputs)
            # Derive velocity from x prediction
            v_pred = self.model.compute_v_from_x_pred(x_pred, x_t, t)
        elif self._prediction_mode == "v":
            # Predict velocity directly (CFM)
            v_pred = self._call_model_from_inputs(model_inputs)
            # Derive x from v prediction
            t_view = t.view(B, 1, 1, 1, 1)
            x_pred = x_t + v_pred * (1 - t_view)
        else:  # epsilon
            # Predict noise (diffusion-style)
            eps_pred = self._call_model_from_inputs(model_inputs)
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
        ssim = self._compute_ssim(x_pred, batch_y, stage='train')
        
        # Log metrics
        metrics = {
            'loss': loss,
            'v_loss': v_loss,
            'x_loss': x_loss,
            'ssim': ssim,
        }
        self._log_step_metrics('train', metrics, on_step=True, on_epoch=True, prog_bar_keys={'loss', 'ssim'})
        self.log('train/t_mean', t.mean(), on_step=False, on_epoch=True)
        
        return metrics
    
    def validation_step(self, batch, batch_idx):
        """Validation step using training logic."""
        batch_x, batch_y, batch_labels = self._split_batch(batch)
        B = batch_y.shape[0]
        
        if batch_idx == 0:
            self._eval_sample_batch = (
                batch_x[:4].detach().clone(),
                batch_y[:4].detach().clone(),
                batch_labels[:4].detach().clone() if batch_labels is not None else None,
            )
            
        # Sample and compute loss
        x_0 = torch.randn_like(batch_y)
        t, x_t, v_target = self.fm.sample_location_and_conditional_flow(x_0, batch_y, t=None)
        
        # Prepare model inputs (no CFG dropout during validation)
        model_inputs = self._prepare_model_inputs(
            x_t, t, batch_x, batch_labels=batch_labels, apply_cond_dropout=False
        )
        
        if self._prediction_mode == "x":
            x_pred = self._call_model_from_inputs(model_inputs)
            v_pred = self.model.compute_v_from_x_pred(x_pred, x_t, t)
        elif self._prediction_mode == "v":
            v_pred = self._call_model_from_inputs(model_inputs)
            t_view = t.view(B, 1, 1, 1, 1)
            x_pred = x_t + v_pred * (1 - t_view)
        else:
            eps_pred = self._call_model_from_inputs(model_inputs)
            t_view = t.view(B, 1, 1, 1, 1)
            x_pred = (x_t - (1 - t_view) * eps_pred) / t_view.clamp(min=1e-5)
            v_pred = self.model.compute_v_from_x_pred(x_pred, x_t, t)
        
        v_loss = nn.functional.mse_loss(v_pred, v_target)
        x_loss = nn.functional.mse_loss(x_pred, batch_y)
        loss = v_loss if self._loss_type == "v" else x_loss
        ssim = self._compute_ssim(x_pred, batch_y, stage='val')
        
        metrics = {
            'loss': loss,
            'v_loss': v_loss,
            'x_loss': x_loss,
            'ssim': ssim,
        }
        self._log_step_metrics('val', metrics, on_step=True, on_epoch=True, prog_bar_keys={'loss', 'ssim'}, sync_dist=True)
        return metrics
    
    def on_validation_epoch_end(self):
        """Compute epoch-level metrics for validation with ODE sampling."""
        if not hasattr(self, '_eval_sample_batch'):
            return
            
        batch_x, batch_y, batch_labels = self._eval_sample_batch
        
        try:
            with torch.no_grad():
                # Force n_ens=1 for validation fast sampling
                x_samples = self.sample(batch_x, target_shape=batch_y.shape, labels=batch_labels)
                
                # Compute Sample MSE & MAE directly
                sample_mse = nn.functional.mse_loss(x_samples, batch_y).item()
                sample_mae = nn.functional.l1_loss(x_samples, batch_y).item()
                self.log("val/sample_mse", sample_mse, sync_dist=True)
                self.log("val/sample_mae", sample_mae, sync_dist=True)
                self.log("val/sample_crps", sample_mae, sync_dist=True)  # Equal to MAE for n_ens=1
                
                preds = x_samples.cpu().numpy()
                trues = batch_y.cpu().numpy()
                
                # Full metrics
                threshold = self._get_metric_threshold()
                print("METRICS BEFORE CALC:", self.metric_list, threshold)
                eval_res, _ = metric(
                    preds, trues,
                    self.hparams.test_mean, self.hparams.test_std,
                    metrics=self.metric_list,
                    channel_names=self.channel_names,
                    spatial_norm=self.spatial_norm,
                    threshold=threshold
                )
                
                pf_res = per_frame_metric(
                    preds, trues,
                    mean=self.hparams.test_mean, std=self.hparams.test_std,
                    metrics=self.metric_list, spatial_norm=self.spatial_norm,
                    threshold=threshold
                )
                
                if 'ssim' in pf_res:
                    ssim_pf = pf_res['ssim']
                    T = len(ssim_pf)
                    ssim_mean = np.mean(ssim_pf)
                    self.log("val/sample_ssim", float(ssim_mean), sync_dist=True, prog_bar=True)
                    self.log("val/sample_ssim_start", float(ssim_pf[0]), sync_dist=True)
                    self.log("val/sample_ssim_mid", float(ssim_pf[T // 2]), sync_dist=True)
                    self.log("val/sample_ssim_end", float(ssim_pf[-1]), sync_dist=True)
                
                # Other deterministic metrics
                if 'pod' in eval_res: self.log("val/sample_pod", eval_res['pod'], sync_dist=True)
                if 'far' in eval_res: self.log("val/sample_far", eval_res['far'], sync_dist=True)
                if 'csi' in eval_res: self.log("val/sample_csi", eval_res['csi'], sync_dist=True)

        finally:
            if hasattr(self, '_eval_sample_batch'):
                del self._eval_sample_batch

    def test_step(self, batch, batch_idx):
        """Test step with ODE sampling."""
        batch_x, batch_y, batch_labels = self._split_batch(batch)
        
        n_ens = int(self.hparams.get('test_num_ensemble', self._test_num_ensemble))
        
        ensemble_preds = []
        for _ in range(n_ens):
            pred_yi = self.sample(batch_x, target_shape=batch_y.shape, labels=batch_labels)
            ensemble_preds.append(pred_yi.unsqueeze(1))

        ensemble_preds = torch.cat(ensemble_preds, dim=1)
        pred_y = ensemble_preds.mean(dim=1)

        outputs = {
            'inputs': batch_x.cpu().numpy(),
            'preds': pred_y.cpu().numpy(),
            'trues': batch_y.cpu().numpy(),
            'ensemble_preds': ensemble_preds.cpu().numpy()
        }
        if batch_labels is not None:
            outputs['labels'] = batch_labels.cpu().numpy()

        if n_ens > 1:
            trues = batch_y.unsqueeze(1)
            term1 = torch.abs(ensemble_preds - trues).mean()
            pairwise_terms = []
            for i in range(n_ens):
                for j in range(i + 1, n_ens):
                    pairwise_terms.append(torch.abs(ensemble_preds[:, i] - ensemble_preds[:, j]).mean())
            term2 = torch.stack(pairwise_terms).mean() if pairwise_terms else torch.zeros((), device=batch_y.device)
            crps = term1 - 0.5 * term2
        else:
            crps = torch.abs(pred_y - batch_y).mean()

        self.test_outputs.append(outputs)
        return outputs
    
    def sample(
        self,
        x_past: torch.Tensor,
        target_shape: Optional[tuple] = None,
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        labels: Optional[torch.Tensor] = None,
        cond_spatial: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Sample from the model using ODE integration.
        
        Args:
            x_past: Past frames (B, T_in, C, H, W) - temporal context
            target_shape: Target output shape (B, T_out, C, H, W)
            num_steps: Number of ODE steps
            solver: ODE solver ('dopri5', 'rk4', 'euler', 'midpoint')
            guidance_scale: Override guidance scale
            labels: Optional sequence labels (B, num_classes) for label conditioning
            cond_spatial: External spatial condition (B, Cond_C, H, W), optional.
                         If None, will be computed from x_past if use_spade=True.
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
        
        B = x_past.shape[0]
        device = x_past.device
        
        # Determine output shape
        if target_shape is not None:
            _, T_out, C, H, W = target_shape
        else:
            T_out = self.hparams.get('aft_seq_length', self.hparams.get('pre_seq_length', x_past.shape[1]))
            C, H, W = x_past.shape[2], x_past.shape[3], x_past.shape[4]
        
        # Initial noise
        z0 = torch.randn(B, T_out, C, H, W, device=device, dtype=torch.float32)
        
        # Convert past to FP32
        x_past_fp32 = x_past.to(dtype=torch.float32)
        
        # Prepare SPADE condition if not provided
        if cond_spatial is None:
            use_spade = self.hparams.get('use_spade', True) or self._use_spatial_label_conditioner
            if self._use_adaln_label_conditioning:
                use_spade = False
            if use_spade:
                if labels is not None and self.label_conditioner is not None:
                    cond_spatial = self.label_conditioner(labels)
                elif self.context_encoder is not None:
                    cond_spatial = self.context_encoder(x_past_fp32)
                else:
                    # Fallback: flatten temporal dimension
                    from einops import rearrange
                    cond_spatial = rearrange(x_past_fp32, 'b t c h w -> b (t c) h w')
        
        # Convert to FP32
        cond_spatial_fp32 = cond_spatial.to(dtype=torch.float32) if cond_spatial is not None else None
        label_emb = self._get_label_emb(labels)
        label_emb_fp32 = label_emb.to(dtype=torch.float32) if label_emb is not None else None
        
        # ODE function
        def ode_func(t_scalar, x):
            t_val = t_scalar.item() if isinstance(t_scalar, torch.Tensor) else float(t_scalar)
            t_batch = torch.full((B,), t_val, device=device, dtype=x.dtype)
            
            # Conditional prediction
            if self._prediction_mode == "x":
                x_pred = self._call_model(
                    x,
                    t_batch,
                    x_past=x_past_fp32,
                    cond_spatial=cond_spatial_fp32,
                    label_emb=label_emb_fp32,
                )
                v_cond = self.model.compute_v_from_x_pred(x_pred, x, t_batch)
            elif self._prediction_mode == "v":
                v_cond = self._call_model(
                    x,
                    t_batch,
                    x_past=x_past_fp32,
                    cond_spatial=cond_spatial_fp32,
                    label_emb=label_emb_fp32,
                )
            else:  # epsilon
                eps_pred = self._call_model(
                    x,
                    t_batch,
                    x_past=x_past_fp32,
                    cond_spatial=cond_spatial_fp32,
                    label_emb=label_emb_fp32,
                )
                t_view = t_batch.view(B, 1, 1, 1, 1)
                x_pred = (x - (1 - t_view) * eps_pred) / t_view.clamp(min=1e-5)
                v_cond = self.model.compute_v_from_x_pred(x_pred, x, t_batch)
            
            # Classifier-Free Guidance
            if guidance_scale != 1.0:
                # Unconditional: zero out past frames and condition
                x_past_uncond = torch.zeros_like(x_past_fp32)
                
                if self._prediction_mode == "x":
                    x_pred_uncond = self._call_model(x, t_batch, x_past=x_past_uncond, cond_spatial=None)
                    v_uncond = self.model.compute_v_from_x_pred(x_pred_uncond, x, t_batch)
                elif self._prediction_mode == "v":
                    v_uncond = self._call_model(x, t_batch, x_past=x_past_uncond, cond_spatial=None)
                else:
                    eps_pred_uncond = self._call_model(x, t_batch, x_past=x_past_uncond, cond_spatial=None)
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
        
        threshold = self._get_metric_threshold()

        # Global metrics
        eval_res, eval_log = metric(
            results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std,
            metrics=self.metric_list,
            channel_names=self.channel_names,
            spatial_norm=self.spatial_norm,
            threshold=threshold
        )
        
        # Per-frame metrics
        pf_res = per_frame_metric(
            results_all['preds'], results_all['trues'],
            mean=self.hparams.test_mean, std=self.hparams.test_std,
            metrics=self.metric_list, spatial_norm=self.spatial_norm,
            threshold=threshold)

        # Build comprehensive metrics dict
        results_all['metrics'] = eval_res
        results_all['per_frame_metrics'] = pf_res

        # Calculate CRPS for flow matching if ensemble was computed
        if 'ensemble_preds' in results_all:
            eps_preds = results_all['ensemble_preds']  # (Total_B, N_ens, T, C, H, W)
            trues = results_all['trues']  # (Total_B, T, C, H, W)
            
            # Term 1: MAE of each ensemble vs target
            diff1 = np.abs(eps_preds - np.expand_dims(trues, axis=1))
            # Average over batch (0), ensemble (1), time (2), and sum over channels, h, w (3, 4, 5)
            term1 = diff1.mean(axis=(0, 1, 2)).sum()
            
            # Term 2: Mean pairwise distance between ensembles
            term2 = 0.0
            count = 0
            N_ens = eps_preds.shape[1]
            for i in range(N_ens):
                for j in range(i + 1, N_ens):
                    diff2 = np.abs(eps_preds[:, i] - eps_preds[:, j])
                    term2 += diff2.mean(axis=(0, 1)).sum()
                    count += 1
            if count > 0:
                term2 /= count
                
            crps = float(term1 - 0.5 * term2)
        else:
            # Equivalent to MAE when n_ens = 1
            crps = float(eval_res.get('mae', 0.0))
            
        eval_res['crps'] = crps
        eval_log += f", crps:{crps:.4f}"
        self.log('test/crps', crps, on_step=False, on_epoch=True, sync_dist=True)

        # Add summary SSIM at start/mid/end if available
        if 'ssim' in pf_res:
            ssim_pf = pf_res['ssim']
            T = len(ssim_pf)
            eval_res['ssim_start'] = float(ssim_pf[0])
            eval_res['ssim_mid'] = float(ssim_pf[T // 2])
            eval_res['ssim_end'] = float(ssim_pf[-1])
            eval_log += f", ssim_start:{ssim_pf[0]}, ssim_mid:{ssim_pf[T // 2]}, ssim_end:{ssim_pf[-1]}"

        self.log_dict({f'test/{k}': float(v) for k, v in eval_res.items()}, sync_dist=True)

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            # Save raw arrays
            for np_data in ['inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])

            # Save global metrics dict
            np.save(osp.join(folder_path, 'metrics.npy'), eval_res)

            # Save per-frame metrics dict
            np.save(osp.join(folder_path, 'per_frame_metrics.npy'), pf_res)

            print_log(f"Saved test results to {folder_path}")
        
        # Clear test outputs
        self.test_outputs = []
        return results_all
