# MambaFlow Ablation Configuration for OpenSTL
# Template config for running ablation studies on Flow Matching methods
#
# This config demonstrates how to switch between different ablation settings:
# - Prediction mode: x (JvM) vs v (CFM) vs epsilon (Diffusion)
# - Loss type: v (velocity loss) vs x (pixel loss)
# - Backbone: mamba vs attention vs conv
# - Conditioning: use_spade (SPADE) vs concatenation
# - OT Sampling: enable/disable optimal transport
#
# Usage:
#   python tools/train.py -c configs/custom/mambaflow_ablation.py
#
# To run different ablations, modify the parameters below.

# ==============================================================================
# Method Configuration
# ==============================================================================
method = 'FlowMatching'

# ==============================================================================
# Flow Matching Parameters (Ablation: Objective)
# ==============================================================================

# Prediction mode (what the model predicts)
# - 'x': JvM (Just Image) - predicts clean image directly (BEST)
# - 'v': CFM (Continuous Flow Matching) - predicts velocity
# - 'epsilon': Diffusion-style - predicts noise
prediction_mode = 'x'  # Ablation: Change to 'v' for CFM or 'epsilon' for diffusion

# Loss type (how to compute the loss)
# - 'v': Velocity loss - provides implicit time weighting (BEST for mode='x')
# - 'x': Pixel loss - direct MSE in pixel space
# Note: When prediction_mode='v', loss_type is forced to 'v'
loss_type = 'v'  # Ablation: Change to 'x' for pixel-space loss

# ODE solver for inference
# - 'dopri5': Adaptive solver (accurate but slower)
# - 'rk4': Fixed-step 4th order Runge-Kutta
# - 'euler': Simple Euler (fast but less accurate)
# - 'midpoint': Midpoint method
ode_solver = 'dopri5'

# Number of ODE integration steps for inference
num_inference_steps = 50

# CFM noise level (sigma_min in interpolation)
sigma_min = 0.0

# ==============================================================================
# Time Sampling (Ablation: Training Distribution)
# ==============================================================================

# Time sampling distribution
# - 'uniform': Standard uniform sampling
# - 'logit_normal': Logit-normal (emphasizes middle timesteps)
sample_t_distrib = 'uniform'  # Ablation: Change to 'logit_normal'

# Logit-normal parameters (only used if sample_t_distrib='logit_normal')
logit_normal_loc = 0.0
logit_normal_scale = 1.0

# ==============================================================================
# Optimal Transport (Ablation: OT vs Standard)
# ==============================================================================

# Whether to use OT sampling for better trajectories
use_ot_sampling = False  # Ablation: Set to True for OT sampling

# OT method: 'exact' or 'sinkhorn'
ot_method = 'exact'
ot_reg = 0.05  # Regularization for Sinkhorn

# ==============================================================================
# Classifier-Free Guidance (Ablation: Guidance)
# ==============================================================================

# Probability of dropping condition during training (for CFG)
cond_dropout_prob = 0.0  # Ablation: Set to 0.1-0.2 for CFG training

# Guidance scale for sampling (>1.0 amplifies condition)
guidance_scale = 1.0  # Ablation: Increase to 2.0-7.0 for stronger guidance

# ==============================================================================
# Model Architecture (Ablation: Backbone)
# ==============================================================================

# Backbone type for SPADEJvM blocks
# - 'mamba': STVMamba (STSS + STDSConv) - efficient for long sequences
# - 'attention': Self-attention - standard transformer approach
# - 'conv': Convolutional - lightweight baseline
block_type = 'mamba'  # Ablation: Change to 'attention' or 'conv'

# Model dimensions
model_dim = 256  # Hidden dimension
hid_S = 256      # Alias for SimVP compatibility
time_dim = 512   # Time embedding dimension

# Number of blocks
num_blocks = 12
N_T = 12  # Alias for SimVP compatibility

# Patch embedding
patch_size = (2, 8, 8)  # (T, H, W)
bottleneck_dim = 128

# Mamba-specific parameters (only used if block_type='mamba')
d_state = 64
d_conv = 4
expand = 2

# Attention-specific parameters (only used if block_type='attention')
num_heads = 8

# ==============================================================================
# Conditioning (Ablation: SPADE vs Concatenation)
# ==============================================================================

# Whether to use SPADE conditioning
# - True: SPADE (Spatial Adaptive Normalization) - multiplicative conditioning
# - False: Concatenation - additive conditioning
use_spade = True  # Ablation: Set to False for concatenation

# Number of SPADE condition channels (used if use_spade=True)
cond_channels = 10

# ==============================================================================
# Context Encoder (Optional)
# ==============================================================================

# Whether to use a context encoder (for complex multi-variable conditions)
# - False: Direct conditioning (for simple datasets like Moving MNIST)
# - True: Use context encoder (for complex datasets like weather, AOD)
use_encoder = False  # Set to True for complex datasets

# Context encoder type (only used if use_encoder=True)
context_encoder_type = 'ContextNet'

# Context encoder parameters (only used if use_encoder=True)
context_encoder_params = {
    'hidden_dim': 128,      # Internal network width
    'num_layers': 3,        # Number of residual blocks
    'dropout': 0.0,         # Dropout rate
}

# ==============================================================================
# Positional Encoding (Ablation: RoPE)
# ==============================================================================

# Whether to use RoPE (Rotary Positional Encoding)
# Note: This is a placeholder - actual RoPE implementation may vary
use_rope = True  # Ablation: Set to False to disable

# ==============================================================================
# Training Configuration
# ==============================================================================

# Learning rate
lr = 1e-3

# Batch size
batch_size = 16

# Number of epochs
epoch = 200

# Dropout
drop = 0.0
drop_path = 0.0

# Learning rate scheduler
sched = 'onecycle'

# Gradient checkpointing (saves memory, slightly slower)
gradient_checkpointing = True

# ==============================================================================
# Data Configuration (Override as needed)
# ==============================================================================

# These should be set based on your dataset
# pre_seq_length = 10  # Number of input frames
# aft_seq_length = 10  # Number of output frames (prediction horizon)
# in_shape = (pre_seq_length, 1, 64, 64)  # (T, C, H, W)

# ==============================================================================
# Quick Ablation Presets
# ==============================================================================
# Uncomment one of these blocks to quickly switch ablation settings:

# --- Preset: Standard JvM (Default) ---
# prediction_mode = 'x'
# loss_type = 'v'
# block_type = 'mamba'
# use_spade = True

# --- Preset: Standard CFM ---
# prediction_mode = 'v'
# loss_type = 'v'
# block_type = 'mamba'
# use_spade = True

# --- Preset: Diffusion-style ---
# prediction_mode = 'epsilon'
# loss_type = 'x'
# block_type = 'mamba'
# use_spade = True

# --- Preset: Attention Backbone ---
# prediction_mode = 'x'
# loss_type = 'v'
# block_type = 'attention'
# use_spade = True

# --- Preset: Conv Backbone ---
# prediction_mode = 'x'
# loss_type = 'v'
# block_type = 'conv'
# use_spade = True

# --- Preset: No SPADE (Concatenation) ---
# prediction_mode = 'x'
# loss_type = 'v'
# block_type = 'mamba'
# use_spade = False

# --- Preset: With OT Sampling ---
# prediction_mode = 'x'
# loss_type = 'v'
# use_ot_sampling = True
# ot_method = 'exact'

# --- Preset: With CFG Training ---
# cond_dropout_prob = 0.1
# guidance_scale = 2.0
