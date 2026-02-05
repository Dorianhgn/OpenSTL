# aod\_prediction\_project

This project is dedicated to forecasting Aerosol Optical Depth (AOD) using deep learning techniques. It is structured to be modular, scalable, and research-friendly, leveraging PyTorch and PyTorch Lightning for robust experimentation.

## Project phases

### Phase 1 - Explore

Playing with Unet on a small dataset, to understand preprocessing and useful variables.

### Phase 2 - Deterministic Models

This time, we use a larger map (**latitude** from 12 to 57, **longitude** from -37 to 76), covering all the Sahara, a big part of the atlantic, up to northern Europe, and all the way to Kazakstan (Qizilqum desert).

#### A. Applying existing SOTA architectures

- U-Net basline
- SimVP (gSTA) from [Tan, Cheng, et al. "SimVPv2: Towards simple yet powerful spatiotemporal predictive learning." IEEE Transactions on Multimedia (2025).](https://ieeexplore.ieee.org/abstract/document/10891381)
- STVMamba from [Zou, Maoyang, et al. "STVMamba: precipitation nowcasting with spatiotemporal prediction model." Scientific Reports 15.1 (2025): 22568.](https://www.nature.com/articles/s41598-025-05902-4)

#### B. Exploring architectures

-> Explore CrossAttention with STVMamba.

#### C. Data Pipeline

The `WeatherDataModule` handles loading and preprocessing of the Phase 2 dataset. It supports two dataset types:
- **`flat`**: For U-Net models - flattens temporal sequences into channels
- **`cuboid`**: For spatiotemporal models (SimVP, STVMamba) - preserves temporal dimension

Key parameters:
```yaml
data:
  cams_dir: data/phase2/processed/1_cams_chunks        # CAMS AOD data
  era5_dir: data/phase2/processed/2_met_chunks         # ERA5 meteorological data
  dataset_type: cuboid                                 # flat or cuboid
  lagged_aod: 8                                        # Input timesteps for AOD
  lagged_met: 8                                        # Input timesteps for meteorology
  forecast_horizon: 24                                 # Forecast 24 timesteps (72h)
  train_years: 17.0                                    # Training data duration
  val_years: 2.0                                       # Validation data duration
```

The datamodule automatically handles normalization, chronological train/val/test splits, and memory-efficient data loading via memory-mapping.

### Phase 3 & 4 - Conditional Flow Matching

#### A. Encoding the dataset
Training [AutoencoderKL](https://huggingface.co/docs/diffusers/v0.35.1/api/models/autoencoderkl) to encode all AOD frames.
Freeze the weights of the best STVMamba model from *phase 2*, to extract the part responsible for encoding the meterological variables.

#### B. Training a model on Conditional

Train the best *phase 2* model to predict the vector field of the CFM workflow.

### Phase 5 - JustVideoMamba (JvM)

Building on the Conditional Flow Matching framework from Phase 3 & 4, we develop **JustVideoMamba** (JvM) - a spatiotemporal video generation model that combines Mamba's selective state spaces with advanced spatial conditioning.

#### Key Features:
- **SPADEJvM**: Uses SPADE (Spatial Adaptive Normalization) for strong spatial conditioning with meteorological variables
- **Direct Prediction**: Predicts clean target frames (x) directly, avoiding noise/velocity space
- **Context Encoding**: ContextNet (ResNet-style) encodes complex multi-variable conditions (27 meteorological channels)
- **Past+Future Strategy**: Concatenates clean past frames with noisy future frames for temporal coherence
- **Optimal Transport Sampling**: Optional OT-CFM for improved sample quality
- **Classifier-Free Guidance**: Supports conditional dropout during training for controllable generation

#### Architecture:
- Mamba-based spatiotemporal processing (STSS + STDSConv)
- SPADE normalization for multiplicative spatial conditioning
- Gated residuals for controlled information flow
- RoPE3D for geometric position encoding

Configurations available in `config_phase5/`, with support for both raw weather data and latent space training.

-----

## 🧱 Project Structure

```
aod_prediction_project/
├── data/                         # Raw and preprocessed data
│   ├── phase_1/
│   │   ├── raw/                  # .grib, .nc, .zarr files, etc.
│   │   └── processed/            # Stacked .npy files
│   ├── phase_2/
│   │   ├── raw/
│   │   └── processed/
│   ├── phase_3/
│   │   └── latent/               # Latent encoded Dataset for CFM
│   └── phase_4/
│
├── datasets/                     # All your PyTorch Datasets and DataLoaders
│   ├── data_phase_1_prep.ipynb
│   ├── data_phase_2_prep.ipynb   # Data preparation notebooks with conda-forge env and dask
│   ├── weather_flat.py
│   ├── weather_cuboid.py
│   ├── weather_datamodule.py     # lightning datamodule
│   ├── phase3_latent_dataset.py
│   ├── phase3_datamodule.py
│   ├── phase3_dataset.py
│   ├── phase4_mnist.py
│   ├── phase4_moving_mnist.py
│   ├── phase4_otfm_datamodule.py
│   ├── phase4_otfm_dataset.py
│   └── jvm_weather_datamodule.py # Phase 5 datamodule for JvM 
│
├── models/                       # One folder per architecture
│   ├── unet/
│   │   ├── lightning_module.py   # PyTorch LightningModule
│   │   ├── model.py              # Pure model architecture (nn.Module)
│   │   ├── config/               # YAML files for the Lightning CLI
│   │   │   ├── classic.yaml
│   │   │   ├── sparse_lag.yaml
│   │   │   └── exp_lag.yaml
│   │   └── ...
│   ├── simvp/
│   │   └── ...
│   ├── stvmamba/
│   │   └── ...
│   ├── cfm/                      # Conditional Flow Matching models
│   │   └── ...
│   └── jvm/                      # Phase 5: JustVideoMamba
│       ├── jvm_base.py
│       ├── jvm_pure.py
│       ├── jvm_spade.py
│       ├── lightning_module_jvm.py
│       ├── config_jvm/
│       └── config_jvtm/
│
├── experiments/                       # Training results and artifacts
│   ├── phase1/
│   ├── phase2/
│   │   ├── simvp/
│   │   │   ├── lag_8_4a.1
│   │   │   │   ├── logs/              # TensorBoard logs
│   │   │   │   ├── checkpoints/       # Automatically saved model checkpoints
│   │   │   │   └── config.yaml        # Saved Lightning CLI config for reproducibility
│   │   ├── stvmamba/
│   │   ├── unet/
│   │   └── final_presentation_plots/  # Plots that compare different models and checkpoints
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   │   ├── jvm/
│   │   └── jvtm/
│   └── duaod_final/                   # Final experiments for thesis
│
├── scripts/                     # Utility functions
│   ├── evaluate_model.py        # Evaluation for phase 2
│   ├── animate_predictions.py   # MP4 generators, plots, etc.
│   ├── model_summary.py         # Model summary generator for phase 2
│   └── ...                      # AutoencoderKL trainer, evalutation, phase3 dataset encoding...
│
├── notebooks/                   # Analysis and visualization notebooks
│   ├── evaluation_unet.ipynb
│   ├── shap_analysis.ipynb
│   ├── forecast_comparison.ipynb
│   └── visualization.ipynb
│
├── tests/                         # Unit tests (pytest)
│   ├── test_data_loading.py
│   ├── smoke_test_stvmamba.py
│   └── lightning_one_step_stvmamba.py
|
├── jobs/                          # Scripts for launching jobs (e.g., on a cluster)
│   ├── train_unet_sparse_lag.sh
│   ├── multinode_stvmamba.sh
│   ├── ...
│   └── templates/
│       └── slurm_template.sh      # (optional) reusable SLURM template
│
├── logs/                          # SLURM .out and .err log files
│
├── main.py                        # Project entry point - Lightning CLI
├── README.md                      # Project explanation
├── requirements.txt               # Project dependencies
├── .gitignore
└── .env                           # (optional) environment variables
```

-----

## 🧭 Directory Roles

| Directory          | Main Role                                                      |
| ------------------ | -------------------------------------------------------------- |
| `data/`            | Stores raw and preprocessed datasets.                          |
| `datasets/`        | Contains modular PyTorch `Dataset` and `DataLoader` classes.   |
| `models/`          | Defines architectures, `LightningModule`s, CLI configs, and checkpoints. |
| `experiments/`     | Stores complete outputs of runs (logs, checkpoints, configs).  |
| `scripts/`         | Scripts with CLI. To run with `python -m`                      |
| `notebooks/`       | Used for post-training analysis and visualization.             |
| `tests/`           | Contains fast unit tests to validate individual components.    |
| `jobs/`            | Includes scripts for launching training/evaluation jobs (e.g., with SLURM). |
| `logs/`            | Contains output logs from job scheduler (e.g., SLURM).         |
| `main.py`          | Central entry point for running the project via Lightning CLI. |
| `requirements.txt` | Lists all dependencies for easy environment setup.             |

-----

## 🔌 Usage Example (Phase 1 & 2)

### Train a model using the Lightning CLI:

```bash
python main.py fit --config models/unet/config/sparse_lag.yaml
```

### Visualize training logs:

```bash
tensorboard --logdir=experiments/phase2/
```

### Resume from a checkpoint:

You can easily resume a previously trained configuration with:

```bash
python main.py fit --ckpt_path experiments/unet_sparse_lag_3-6-9/checkpoints/last.ckpt
```

## 🔌 Usage Example (Phase 3, 4 & 5)

### Phase 3 & 4 - CFM with latent encoding:
```bash
python main_phase3.py fit --config config_phase5/cfm_spade_raw_weather.yaml
```

### Phase 5 - JustVideoMamba:
```bash
python main_phase3.py fit --config config_phase5/jvm_spade_raw_weather.yaml
```

-----

## 🎯 Key Advantages

✅ **Modular**: Allows for rapid iteration on architectures and data configurations. <br/>
✅ **Readable**: The structure ensures you'll understand your work even 6 months later. <br/>
✅ **Extensible**: Easily add new models, data phases, or outputs with no structural changes. <br/>
✅ **Research-Friendly**: The project can be cleanly published, shared, and reproduced. <br/>

---

## Proposed Evaluation Metrics

#### 1. Pixel-wise Error Metrics 📏

These are direct improvements on MSE that are often more interpretable.

* **RMSE (Root Mean Squared Error)**: This is simply the square root of your MSE ($\sqrt{MSE}$). Its main advantage is that the error is in the **same units as your AOD data**, making it much easier to interpret. For example, an RMSE of 0.05 means the typical error is about 0.05 AOD units.
* **MAE (Mean Absolute Error)**: This metric calculates the average of the absolute differences between prediction and ground truth. Unlike MSE, it's **not as sensitive to large, outlier errors**. It gives a very clear picture of the average error magnitude.

---
#### 2. Perceptual Quality Metrics 🖼️

These metrics measure how visually similar the prediction is to the ground truth, which often aligns better with human perception than pixel-wise errors.

* **SSIM (Structural Similarity Index Measure)**: This is one of the most important metrics you can add. It measures the similarity in terms of structure, luminance, and contrast. A score of **1.0 means a perfect structural match**. [cite_start]This metric was used in the STVMamba paper to evaluate performance [cite: 335] and is excellent for showing that your model preserves the shapes and textures of the AOD plumes.

---
#### 3. Skill Scores (Threshold-based) 🎯

These are standard in meteorology and are perfect for your presentation. They measure the model's ability to correctly forecast "events," which you define by setting a threshold (e.g., an AOD value of 0.3 or higher signifies a significant dust event).

* **CSI (Critical Success Index)**: Also called the Threat Score, this is the most common skill score. It measures the fraction of correct "yes" forecasts out of all the times an event was forecasted or observed. It answers the question: **"Of all the important dust plumes (real or predicted), what fraction did we get right?"** A higher CSI is better. [cite_start]This was also a key metric in the STVMamba paper[cite: 335].
* **POD (Probability of Detection)**: Also called the Hit Rate. It answers: **"What fraction of the *actual* dust plumes did the model successfully predict?"**
* **FAR (False Alarm Ratio)**: It answers: **"What fraction of the plumes the model *predicted* were actually false alarms?"**