"""
JvM Weather DataModule - Wraps WeatherDataModule for SPADEJvM format.

Transforms WeatherCuboidDataset output from (X, y) to {"x_past", "x_future", "cond"}
format expected by SPADEJvMLightningModule.
"""
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from typing import Union, List, Optional

from datasets.weather_datamodule import WeatherDataModule


class JvMWeatherDatasetWrapper(Dataset):
    """
    Wrapper around WeatherCuboidDataset to transform output format.
    
    WeatherCuboidDataset returns (X, y) where:
        - X: (T_in, C_total, H, W) - concatenated past AOD + MET + Time + Static
        - y: (T_out, C_aod, H, W) - future AOD to predict
    
    SPADEJvM expects {"x_past", "x_future", "cond"} where:
        - x_past: (T_in, C_aod, H, W) - past AOD only (clean, for input)
        - x_future: (T_out, C_aod, H, W) - future AOD (target)
        - cond: (T_in, C_raw, H, W) - raw weather data (MET + optional time/static)
    
    Args:
        dataset: WeatherCuboidDataset instance
        c_aod: Number of AOD channels (default 6)
    """
    
    def __init__(self, dataset: Dataset, c_aod: int = 6):
        self.dataset = dataset
        self.c_aod = c_aod
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> dict:
        X, y = self.dataset[idx]
        
        # X shape: (T_in, C_total, H, W)
        # First c_aod channels are past AOD, rest are MET/time/static
        x_past = X[:, :self.c_aod, :, :]      # (T_in, C_aod, H, W)
        cond = X[:, self.c_aod:, :, :]        # (T_in, C_raw, H, W)
        x_future = y                          # (T_out, C_aod, H, W)
        
        return {
            "x_past": x_past,
            "x_future": x_future,
            "cond": cond,
        }


class JvMWeatherDataModule(WeatherDataModule):
    """
    DataModule for SPADEJvM training with raw weather data.
    
    Inherits from WeatherDataModule and wraps datasets to provide
    the {"x_past", "x_future", "cond"} format expected by SPADEJvMLightningModule.
    
    This is designed for training on raw data (not precomputed latents).
    For latent-space training, use Phase4OTFMDataModule instead.
    
    Args:
        c_aod: Number of AOD channels (default 6)
        All other args inherited from WeatherDataModule
    
    Example:
        >>> dm = JvMWeatherDataModule(
        ...     cams_dir="data/phase2/processed/1_cams_chunks",
        ...     era5_dir="data/phase2/processed/2_met_chunks",
        ...     dataset_type="cuboid",
        ...     lagged_aod=8,
        ...     lagged_met=8,
        ...     forecast_horizon=24,
        ...     train_years=[2003, 2004, ..., 2016],
        ...     val_years=[2017, 2018],
        ...     test_years=[2019],
        ...     batch_size=4,
        ...     num_workers=4,
        ...     pin_memory=True,
        ... )
        >>> dm.setup()
        >>> batch = next(iter(dm.train_dataloader()))
        >>> # batch contains: x_past, x_future, cond
    """
    
    def __init__(
        self,
        # Path args
        cams_dir: str,
        era5_dir: str,
        dataset_type: str = "cuboid",
        lagged_aod: Union[list, int] = 8,
        lagged_met: Union[list, int] = 8,
        forecast_horizon: Union[list, int] = 24,
        # Splitting args
        train_years: Union[float, List[int]] = 17,
        val_years: Union[float, List[int]] = 2,
        # Dataloader args
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        static_path: Optional[str] = None,
        time_path: Optional[str] = None,
        n_s: Optional[int] = None,
        test_years: Optional[List[int]] = None,
        # JvM-specific
        c_aod: int = 6,
    ):
        super().__init__(
            cams_dir=cams_dir,
            era5_dir=era5_dir,
            dataset_type=dataset_type,
            lagged_aod=lagged_aod,
            lagged_met=lagged_met,
            forecast_horizon=forecast_horizon,
            train_years=train_years,
            val_years=val_years,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            static_path=static_path,
            time_path=time_path,
            n_s=n_s,
            test_years=test_years,
        )
        self.c_aod = c_aod
    
    @classmethod
    def from_config(cls, data_cfg: dict):
        """
        Construct JvMWeatherDataModule from config dictionary.
        
        Args:
            data_cfg: Dictionary containing data configuration
        
        Returns:
            JvMWeatherDataModule instance
        """
        return cls(
            cams_dir=data_cfg["cams_dir"],
            era5_dir=data_cfg["era5_dir"],
            dataset_type=data_cfg.get("dataset_type", "cuboid"),
            lagged_aod=data_cfg["lagged_aod"],
            lagged_met=data_cfg["lagged_met"],
            forecast_horizon=data_cfg["forecast_horizon"],
            train_years=data_cfg.get("train_years", 17),
            val_years=data_cfg.get("val_years", 2),
            batch_size=data_cfg.get("batch_size", 4),
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=data_cfg.get("pin_memory", False),
            static_path=data_cfg.get("static_path", None),
            time_path=data_cfg.get("time_path", None),
            n_s=data_cfg.get("n_s", None),
            test_years=data_cfg.get("test_years", None),
            c_aod=data_cfg.get("c_aod", 6),
        )
    
    def setup(self, stage: str = None):
        """Setup datasets and wrap them for JvM format."""
        # Call parent setup to create train/val/test datasets
        super().setup(stage)
        
        # Wrap datasets to transform output format
        if hasattr(self, 'train_dataset'):
            self.train_dataset = JvMWeatherDatasetWrapper(self.train_dataset, self.c_aod)
        if hasattr(self, 'val_dataset'):
            self.val_dataset = JvMWeatherDatasetWrapper(self.val_dataset, self.c_aod)
        if hasattr(self, 'test_dataset'):
            self.test_dataset = JvMWeatherDatasetWrapper(self.test_dataset, self.c_aod)
        
        print(f"\n✅ JvMWeatherDataModule setup complete")
        print(f"   Format: {{'x_past': (T_in, {self.c_aod}, H, W), 'x_future': (T_out, {self.c_aod}, H, W), 'cond': (T_in, C_raw, H, W)}}")
