method = 'FlowMatching'
prediction_mode = 'x'  # Ablation: Change to 'v' for CFM or 'epsilon' for diffusion
loss_type = 'v'  # Ablation: Change to 'x' for pixel-space loss
ode_solver = 'dopri5'
num_inference_steps = 50
sigma_min = 0.0
sample_t_distrib = 'logit_normal'  # Ablation: Change to 'uniform'
logit_normal_loc = -0.5
logit_normal_scale = 1.0

use_ot_sampling = True
ot_method = 'exact'
ot_reg = 0.05  # Regularization for Sinkhorn

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

# Conditionning
use_spade = False # we have mmnist so no conditionning, but we keep the option for future experiments
use_encoder = False  # Set to True for complex datasets


use_rope = False # TODO: implement RoPE positional in SPADEJvM

# training
lr = 2.0e-4
batch_size = 16
drop_path = 0
sched = 'cosine'
lr_min = 1e-6

fp16 = True
opt_lower = 'adamw'
weight_decay = 1.0e-5
