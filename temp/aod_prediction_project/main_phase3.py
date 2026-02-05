import os
import torch
from lightning.pytorch.cli import LightningCLI

# Set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set matmul precision
torch.set_float32_matmul_precision('high')

save_config_callback = int(os.environ.get('SAVE_CONFIG_CALLBACK', 1))

def main():
    """
    Main entry point for Phase 3 (CFM) experiments.
    Uses a standard LightningCLI without the specific WeatherDataModule logic
    present in the main project entry point.
    """
    if save_config_callback:
        cli = LightningCLI(
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_kwargs={"overwrite": True},
        )
    else:
        cli = LightningCLI(
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_kwargs={"overwrite": False},
            save_config_callback=None,
        )

if __name__ == '__main__':
    main()
