# Ablation Study & Paper Plan: Just Video Mamba (JvM)

**Title:** Just Video Mamba: Transformer-Free Highly Contextual Video Generation Model for Spatio-Temporal Forecasting
**Target Venue:** Journal of Machine Learning Research (JMLR) or similar high-impact venue.

## 1. Experimental Setup & Dataset Selection

To prove the model's versatility across varying context levels:

### Low/Normal Context (Spatio-Temporal Video Prediction)
1. **Moving MNIST**: Sanity check, fast iteration, baseline comparisons.
2. **KittiCaltech**: Real-world driving video (tests spatial dynamics and object persistence).
3. **Weather Prediction (ERA5 - t2m or uv10)**: Real-world physics, continuous flows (good for evaluating generative continuity).

### Highly Dimensional Context (Complex Conditioned Forecasting)
4. **DUAOD Dust Plume Dataset (Custom)**: Predicting DUAOD550nm conditioned on past DUAOD + meteorological tensors (wind, humidity, etc.). This highlights your model's unique capability (SPADE + ContextNet integration).

## 2. Core Comparisons & Ablations

### A. The Backbone Architecture (Generative Setting)
Goal: Prove Mamba is highly competitive or superior to Transformers for this task without quadratic complexity.
*   **JvM (STVMamba)** with RoPE3D
*   **JvM (STVMamba)** without RoPE3D (tests Mamba's innate sequence routing vs explicit positional cues)
*   **JvM (Transformer/Attention)** with RoPE3D (The standard baseline)
*   **JvM (Conv)** (Lightweight baseline)

### B. Prediction Mode & Convergence (SSIM over time)
Goal: Show why "Just Image" (`x` prediction) is structurally better for spatiotemporal coherence.
*   **Mode `x` (JvM)** vs **Mode `v` (Continuous Flow Matching)**.
*   *Metric Focus:* Plot validation SSIM over training epochs. Generative structure (SSIM) often takes longer to emerge even if MSE/v-loss drops rapidly. 

### C. Generative vs. Deterministic
Goal: Demonstrate if/when the generative Flow Matching paradigm beats standard deterministic forecasting.
*   **Deterministic SimVP-STVMamba** (Same backbone, deterministic x-loss)
*   **Generative Flow Matching JvM-STVMamba** (Generative sampling)
*   **Deterministic SimVP-gSTA** (Existing Transformer baseline)

### D. The Importance of SPADE for High-Dimensional Context
Goal: Prove that multiplicative SPADE conditioning is superior to standard channel concatenation for complex physics/weather data.
*   **JvM on Dust Dataset + SPADE (with ContextNet Encoder)**
*   **JvM on Dust Dataset + Channel Concatenation** (Standard conditioning)

---

## 3. Current Status

✅ **Done / Available:**
*   **Flow Matching Framework:** Built and configured for `x`, `v`, `epsilon` modes with OT sampling (`openstl/methods/flow_matching.py`).
*   **SPADEJvM Model Layout:** Configured to swap backbones (`mamba`, `attention`, `conv`) and handles SPADE or concat conditioning (`spade_jvm_model.py`).
*   **Dust Dataset Codes:** Present in `temp/aod_prediction_project/datasets/`.
*   **Mamba Rule of 8 Specs:** Documented guidelines to prevent CUDA crashes.

## 4. TODO List for the Codebase

### 1. Data Integration
*   [ ] Adapt `jvm_weather_datamodule.py` and `weather_cuboid.py` from `temp/` into `openstl/datasets/` and register them in OpenSTL's data pipeline.

### 2. Implement Missing Modules
*   [ ] **SimVP + STVMamba Integration:** Create the STVMamba block wrapper for the SimVP method. Ensure the SimVP hidden dimensions and Mamba `headdim` comply with the **Rule of 8** (e.g., if SimVP uses `hid_S=256`, `expand=2`, `headdim=64` -> 8 heads, which is valid).
*   [ ] **RoPE3D Implementation:** The ablation config marks `use_rope = True` as a placeholder. Need to implement true rotary embeddings for the 3D grid in `spade_jvm_model.py` (for both Attention and Mamba backbones).
*   [ ] **Finish `openstl/modules/stvmamba_modules.py`:** Ensure the forward passes for `STSS`, `SwiGLU`, and `STVMambaModule` are complete and bug-free (some lines look truncated or WIP).

### 3. Setup Experiment Scripts
*   [ ] Create standard bash scripts (or slurm jobs) mapped to the variations in `configs/custom/jvm_ablation.py` to launch the runs systematically.
*   [x] Build automated SSIM metric plotting to visualize the `x` vs `v` mode convergence cleanly.