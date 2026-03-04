method = 'SimVP'

# ── Spatial encoder / decoder ─────────────────────────────────────────────────
spatio_kernel_enc = 3
spatio_kernel_dec = 3
N_S = 4          # encoder depth  →  bottleneck H = 64/4 = 16, W = 16

# ── Temporal bottleneck ───────────────────────────────────────────────────────
model_type = 'stvmamba'

# SimVP parameters still required by the Encoder/Decoder
hid_S = 64       # spatial hidden dim (encoder output channels)
hid_T = 512      # unused for stvmamba, kept for API compat
N_T   = 4        # unused for stvmamba, kept for API compat

# STVMamba multi-resolution bottleneck
# ── Rule of 8 check ──────────────────────────────────────────────────────────
# STVMambaModule(dim=hid_c1, expand=2) → inner_dim = 128
# STSS(d_model=128) → Mamba2(d_model=128, expand=2) → nheads=128*2/64=4 ✗
# STVMambaModule(dim=hid_c1=128, expand=2) → inner_dim=256
# STSS(d_model=256) → Mamba2(d_model=256, expand=2) → nheads=256*2/64=8 ✓
hid_c1 = 128     # tier-1 bottleneck channels
hid_c2 = 128     # tier-2 bottleneck channels (Branch A2, after spatial down)

n_t1 = 4         # number of Branch-A1 STVMamba blocks (full-res)
n_t2 = 1         # number of Branch-A2 STVMamba blocks (half-res)
d_state = 64     # Mamba SSM state dimension
layer_scale_init = 1e-4

# ── Training ─────────────────────────────────────────────────────────────────
lr         = 1e-3
batch_size = 16
drop_path  = 0.02
sched      = 'onecycle'
