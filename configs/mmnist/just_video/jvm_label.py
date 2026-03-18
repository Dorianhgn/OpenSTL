method = 'FlowMatching'

# training parameters
prediction_mode = 'x'  # JvM (Just Image) - predicts clean image directly
loss_type = 'v'  # Velocity loss - provides implicit time weighting
ode_solver = 'dopri5'
num_inference_steps = 50
sigma_min = 0.0

# time sampling
sample_t_distrib = 'logit_normal'
logit_normal_loc = -0.5
logit_normal_scale = 1.0

use_ot_sampling = True
ot_method = 'exact'
ot_reg = 0.05

# Classifier-Free Guidance (CFG)
cfg_dropout_prob = 0.1  # 10% chance to drop condition during training
guidance_scale = 1.0  # No guidance during training (will be overridden at test time)

# model parameters
block_type = 'mamba'  # Ablation: Change to 'mamba', 'attention' or 'conv'

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

# Attention-specific 
num_heads = 8

# Conditioning: label injection benchmark
use_spade = True
use_encoder = False
use_label_conditioner = True
label_conditioner_type = 'LabelConditioner'
label_conditioner_params = {
    'hidden_dim': 32,
    'dropout': 0.0,
}
num_classes = 10
return_labels = True

# Positional Encoding
use_rope = False # no RoPE for JvM

# ==============================================================================
# Training Configuration
# ==============================================================================
lr = 2e-4
batch_size = 64
drop_path = 0.02
sched = 'cosine'
warmup_epoch = 0
clip_grad = 1.0
clip_mode = 'norm'

# Gradient checkpointing (saves memory, slightly slower)
gradient_checkpointing = True

epoch=100

fp16 = True
opt = 'adamw'
weight_decay = 1.0e-4

# ==============================================================================
# Fast dev smoke test
# ==============================================================================
fast_dev_run = 1
limit_test_batches = 1

# ==============================================================================
# Testing Configuration
# ==============================================================================
test_num_ensemble=5
metric_threshold=0.8
metrics = ['crps', 'mse', 'mae', 'ssim', 'psnr', 'lpips', 'pod', 'far', 'csi']
