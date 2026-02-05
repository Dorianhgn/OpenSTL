# file: datasets/weather_cuboid.py
import os
import json
import numpy as np
import torch
import hashlib
from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import math

class WeatherCuboidDataset(Dataset):
    """
    PyTorch Dataset to load, sequence, and serve weather and AOD data
    in a cuboid format (T, C, H, W) for spatiotemporal models like SimVP.

    This version uses NumPy's memory-mapping for efficient data handling.
    """
    def __init__(self, data_paths: dict, lag_config: dict, forecast_horizon: int,
                 load_data: bool = True, mean=None, std=None, n_s: Optional[int] = None):
        super().__init__()

        # --- Arguments are identical to WeatherFlatDataset ---
        self.lag_config = lag_config
        self.forecast_horizon = forecast_horizon
        self.data_paths = data_paths
        self.mean = mean
        self.std = std
        self.n_s = n_s 
        
        # --- Max lag calculation is identical ---
        aod_config = lag_config['aod']
        met_config = lag_config['met']
        
        # NOTE: For cuboid, we expect dense lags (int) to define the input sequence length.
        # This logic can be simplified if you only plan to use integer lags for SimVP.
        if not isinstance(aod_config, int) or not isinstance(met_config, int):
             raise ValueError("WeatherCuboidDataset expects integer lag configurations for dense input sequences.")
        if aod_config != met_config:
            print(f"Warning: AOD lag ({aod_config}) and MET lag ({met_config}) are different. "
                  f"Using the larger one ({max(aod_config, met_config)}) as input sequence length.")

        self.in_seq_len = max(aod_config, met_config)
        self.max_lag = self.in_seq_len # Max lag is simply the input sequence length

        if load_data:
            self._load_all_data()
        else:
            self.aod_data = self.met_data = self.static_features = self.time_features = None
            self.num_timesteps = self.height = self.width = self.num_samples = None

        # Check if the padding isn't too big, if n_s is provided
        self._check_padding_sanity()

        # --- Feature name loading is identical ---
        data_root = os.path.dirname(data_paths['cams_dir'])
        static_path = data_paths.get('static_path')
        time_path = data_paths.get('time_path')

        self.aod_names = list(json.load(open(os.path.join(data_root, "1_cams_variable_names.json"))))
        self.met_names = list(json.load(open(os.path.join(data_root, "2_era5_variable_names.json"))))
        self.static_names = list(json.load(open(os.path.join(data_root, "4_static_feature_names.json")))) if static_path else []
        # Time features are often handled differently in spatiotemporal models (e.g., as separate inputs)
        # For now, we will concatenate them as channels, similar to gSTA.
        self.time_names = list(json.load(open(os.path.join(data_root, "3_time_feature_names.json")))) if time_path else []
        self._generate_feature_names()

    def _check_padding_sanity(self):
        """
        Checks if the padding required by the model's n_s is excessive and warns the user.
        """
        if self.n_s is not None and self.n_s > 0 and self.height is not None:
            divisor = 2**self.n_s
            warning_threshold = 0.25  # Warn if padding is > 25% of original dimension

            original_h, original_w = self.height, self.width
            target_h = math.ceil(original_h / divisor) * divisor
            target_w = math.ceil(original_w / divisor) * divisor

            pad_h = target_h - original_h
            pad_w = target_w - original_w

            if pad_h > 0 and (pad_h / original_h > warning_threshold):
                print("\n" + "="*80)
                print(f"⚠️ PADDING WARNING (Height):")
                print(f"  - Your original image height is {original_h}.")
                print(f"  - With n_s={self.n_s}, it will be padded to {target_h} (an increase of {pad_h / original_h:.1%}).")
                print(f"  - This may distort data integrity. Consider reducing n_s in your model configuration.")
                print("="*80 + "\n")

            if pad_w > 0 and (pad_w / original_w > warning_threshold):
                print("\n" + "="*80)
                print(f"⚠️ PADDING WARNING (Width):")
                print(f"  - Your original image width is {original_w}.")
                print(f"  - With n_s={self.n_s}, it will be padded to {target_w} (an increase of {pad_w / original_w:.1%}).")
                print(f"  - This may distort data integrity. Consider reducing n_s in your model configuration.")
                print("="*80 + "\n")

    def _load_data(self, directory: str) -> np.ndarray:
        """
        Loads data from a directory of .npy files into a memory-mapped NumPy array.
        This method is safe for both distributed (multi-GPU) training AND parallel HPO workers.
        Uses file locking to prevent simultaneous cache creation.
        """
        import fcntl  # For file locking
        import time
        
        # --- 1. Get Rank and File List ---
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])
        if not files:
            raise ValueError(f"No .npy files found in {directory}")

        # --- 2. Create a Robust Fingerprint of the Data ---
        hasher = hashlib.md5()
        hasher.update(os.path.abspath(directory).encode())
        for f_path in files:
            hasher.update(os.path.basename(f_path).encode())
            hasher.update(str(os.path.getmtime(f_path)).encode())
        
        data_hash = hasher.hexdigest()
        mmap_filename = f"/tmp/mmap_stack_{data_hash}.mmap"
        meta_filename = f"/tmp/mmap_meta_{data_hash}.json"
        lock_filename = f"/tmp/mmap_stack_{data_hash}.lock"

        # --- 3. Use File Lock to Coordinate Cache Creation ---
        # Acquire lock (will wait if another process is creating cache)
        lock_file = open(lock_filename, 'w')
        try:
            # Try to acquire lock (wait up to 60 seconds)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            # Check if cache already exists (another process may have created it while we waited)
            if os.path.exists(mmap_filename) and os.path.exists(meta_filename):
                try:
                    with open(meta_filename, 'r') as f:
                        meta = json.load(f)
                    # Get expected shape/dtype to validate
                    sample_squeezed = np.load(files[0]).squeeze()
                    expected_shape = (len(files),) + sample_squeezed.shape
                    expected_dtype = str(sample_squeezed.dtype)
                    
                    if meta.get('shape') == list(expected_shape) and meta.get('dtype') == expected_dtype:
                        print(f"--> Found existing valid mmap cache for {os.path.basename(directory)}. Skipping creation.")
                    else:
                        print("--> Cache metadata mismatch. Rebuilding...")
                        self._create_mmap_cache(files, mmap_filename, meta_filename)
                except (json.JSONDecodeError, IOError):
                    print("--> Corrupt metadata file. Rebuilding cache...")
                    self._create_mmap_cache(files, mmap_filename, meta_filename)
            else:
                # No cache exists, create it
                print(f"--> Creating new mmap cache for {os.path.basename(directory)}...")
                self._create_mmap_cache(files, mmap_filename, meta_filename)
        
        finally:
            # Release lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

        # --- 4. All Processes: Load the Memory-Mapped File ---
        with open(meta_filename, 'r') as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        dtype = np.dtype(meta['dtype'])
        
        mmap_array = np.memmap(mmap_filename, dtype=dtype, mode='r', shape=shape)
        return mmap_array
    
    def _create_mmap_cache(self, files: list, mmap_filename: str, meta_filename: str):
        """Helper method to create memory-mapped cache."""
        sample_squeezed = np.load(files[0]).squeeze()
        expected_shape = (len(files),) + sample_squeezed.shape
        expected_dtype = str(sample_squeezed.dtype)
        
        mmap_array = np.memmap(mmap_filename, dtype=sample_squeezed.dtype, mode='w+', shape=expected_shape)
        for i, f_path in enumerate(files):
            chunk_squeezed = np.load(f_path).squeeze()
            mmap_array[i] = chunk_squeezed
        mmap_array.flush()
        
        with open(meta_filename, 'w') as f:
            json.dump({'shape': list(expected_shape), 'dtype': expected_dtype}, f)

    def _load_all_data(self):
        """Load all data arrays and set up dataset dimensions."""
        self.aod_data = self._load_data(self.data_paths['cams_dir']).transpose(1, 0, 2, 3) # -> (T, C_aod, H, W)
        self.met_data = self._load_data(self.data_paths['era5_dir']).transpose(1, 0, 2, 3) # -> (T, C_met, H, W)
        
        static_path = self.data_paths.get('static_path')
        time_path = self.data_paths.get('time_path')
        self.static_features = np.load(static_path) if static_path else None # -> (C_static, H, W)
        self.time_features = np.load(time_path) if time_path else None # -> (T, C_time)

        assert self.aod_data.shape[0] == self.met_data.shape[0], "Time dimension mismatch!"
        if self.time_features is not None:
            assert self.aod_data.shape[0] == self.time_features.shape[0], "Time dimension mismatch with Time Features!"

        self.num_timesteps = self.aod_data.shape[0]
        self.height = self.aod_data.shape[2]
        self.width = self.aod_data.shape[3]
        self.num_samples = self.num_timesteps - self.in_seq_len - self.forecast_horizon + 1


    # --- This method is simplified for the cuboid format ---
    def _generate_feature_names(self):
        """Generates the ordered list of channel names for the input cuboid."""
        self.feature_names = []
        self.feature_names.extend(self.aod_names)
        self.feature_names.extend(self.met_names)
        self.feature_names.extend(self.time_names)
        self.feature_names.extend(self.static_names)

    def __len__(self) -> int:
        if self.num_samples is None:
            # You might want a lazy-loading mechanism here if needed
            self._load_all_data()
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves a sample (X, y) from the dataset.
        X is a cuboid of shape (in_seq_len, C, H, W)
        y is a cuboid of shape (forecast_horizon, C_aod, H, W)
        """
        # 1. Determine time indices
        start_idx = idx
        end_idx = start_idx + self.in_seq_len
        y_start = end_idx
        y_end = y_start + self.forecast_horizon

        # 2. Slice the data to get sequences
        past_aod = self.aod_data[start_idx:end_idx] # (T_in, C_aod, H, W)
        past_met = self.met_data[start_idx:end_idx] # (T_in, C_met, H, W)
        y = self.aod_data[y_start:y_end]            # (T_out, C_aod, H, W)

        # 3. Handle time and static features
        # For gSTA, we expand them and concatenate on the channel dimension.
        features_to_concat = [past_aod, past_met]

        if self.time_features is not None:
            time_seq = self.time_features[start_idx:end_idx] # (T_in, C_time)
            # Expand dims to (T_in, C_time, 1, 1) and broadcast to (T_in, C_time, H, W)
            time_seq_4d = np.broadcast_to(
                time_seq[:, :, np.newaxis, np.newaxis],
                (self.in_seq_len, len(self.time_names), self.height, self.width)
            )
            features_to_concat.append(time_seq_4d)
        
        if self.static_features is not None:
            # Expand dims for static features to (1, C_static, H, W) and broadcast to (T_in, C_static, H, W)
            static_4d = np.broadcast_to(
                self.static_features[np.newaxis, :, :, :],
                (self.in_seq_len, len(self.static_names), self.height, self.width)
            )
            features_to_concat.append(static_4d)

        # 4. Concatenate all features along the channel axis (axis=1)
        X = np.concatenate(features_to_concat, axis=1) # -> (T_in, C_total, H, W)

        # --- NEW: Automatic Padding Logic ---
        if self.n_s is not None and self.n_s > 0:
            divisor = 2**self.n_s

            # Get current height and width from X
            _, _, h, w = X.shape
            
            # Calculate the target height and width (next nearest multiple of divisor)
            target_h = math.ceil(h / divisor) * divisor
            target_w = math.ceil(w / divisor) * divisor
            
            # Calculate padding amounts
            pad_h = target_h - h
            pad_w = target_w - w
            
            # Apply "bottom-right" padding to the spatial dimensions (H, W) of both X and y
            if pad_h > 0 or pad_w > 0:
                pad_width_4d = ((0, 0), (0, 0), (0, pad_h), (0, pad_w))
                X = np.pad(X, pad_width_4d, mode='constant', constant_values=0)
                y = np.pad(y, pad_width_4d, mode='constant', constant_values=0)
        # --- END NEW ---

        # 5. Apply normalization
        if self.mean is not None and self.std is not None:
            # DataModule provides mean/std shaped for broadcasting (1, C, 1, 1)
            mean_np = self.mean.numpy() if hasattr(self.mean, 'numpy') else self.mean
            std_np = self.std.numpy() if hasattr(self.std, 'numpy') else self.std
            X = (X - mean_np) / (std_np + 1e-8)

        # Final check on output shapes
        # X shape: (in_seq_len, C_total, H, W)
        # y shape: (forecast_horizon, C_aod, H, W)
        return X.astype(np.float32), y.astype(np.float32)

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_padded_dims(self) -> Tuple[int, int]:
        """
        Calcule et retourne les dimensions spatiales (H, W) après padding.
        Cette logique est nécessaire pour que le modèle connaisse sa taille d'entrée exacte.
        """
        if self.height is None or self.width is None:
            # S'assure que les données sont chargées si ce n'était pas le cas
            if self.aod_data is None:
                self._load_all_data()

        h, w = self.height, self.width

        if self.n_s is not None and self.n_s > 0:
            divisor = 2**self.n_s
            target_h = math.ceil(h / divisor) * divisor
            target_w = math.ceil(w / divisor) * divisor
            return target_h, target_w
        
        # S'il n'y a pas de padding, retourne les dimensions originales
        return h, w