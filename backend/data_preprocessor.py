"""
Data preprocessing and feature engineering for battery cycle life prediction
"""

import os
import numpy as np
from scipy import interpolate
from scipy.ndimage import uniform_filter1d
from typing import List, Dict, Tuple, Optional
from config import Config
from utils import setup_logging

logger = setup_logging()


class DataPreprocessor:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
    def extract_discharge_data(self, battery_data: np.ndarray) -> List[Dict[str, List[np.ndarray]]]:
        """
        Extract measurements corresponding to discharge portion of cycle
        
        Args:
            battery_data: Raw battery data from .mat file
            
        Returns:
            List of processed discharge data for each battery
        """
        logger.info("Extracting discharge data...")
        discharge_data = []

        batteries = self._iter_struct(battery_data)

        for i, battery in enumerate(batteries):
            battery_discharge = {'Vd': [], 'Td': [], 'QdClipped': []}

            try:
                cycles = self._unwrap_field(battery, 'cycles')

                for cycle in self._iter_struct(cycles):
                    # Extract voltage, temperature, and discharge capacity
                    V = self._extract_array(self._unwrap_field(cycle, 'V'))
                    T = self._extract_array(self._unwrap_field(cycle, 'T'))
                    Qd = self._extract_array(self._unwrap_field(cycle, 'Qd'))
                    
                    # Find indices for discharge portion (3.6V to 2.0V)
                    discharge_data_cycle = self._extract_discharge_portion(V, T, Qd)
                    
                    if discharge_data_cycle is not None:
                        V_discharge, T_discharge, Qd_discharge = discharge_data_cycle
                        
                        battery_discharge['Vd'].append(V_discharge)
                        battery_discharge['Td'].append(T_discharge)
                        battery_discharge['QdClipped'].append(Qd_discharge)
                
                discharge_data.append(battery_discharge)

                if (i + 1) % 5 == 0:
                    logger.info("Processed %d/%d batteries", i + 1, len(batteries))

            except Exception as e:
                logger.warning("Error processing battery %d: %s", i, e)
                discharge_data.append({'Vd': [], 'Td': [], 'QdClipped': []})

        logger.info("Discharge data extraction completed.")
        return discharge_data

    def _iter_struct(self, arr) -> List:
        """
        Iterate a MATLAB struct array regardless of scipy's 2-D promotion.
        Accepts shapes (), (N,), (N,1), (1,N), and (1,M) container wrappers.
        """
        a = np.asarray(arr)
        # Unwrap single-element object containers (e.g. loadmat cell wrappers).
        while a.dtype == object and a.size == 1:
            inner = a.item()
            if isinstance(inner, np.ndarray):
                a = inner
            else:
                break
        if a.ndim == 0:
            return [a.item()] if a.dtype == object else [a]
        return list(a.ravel())

    def _unwrap_field(self, record, field: str):
        """Pull a field from a struct record (void or 0-D struct ndarray)."""
        if isinstance(record, np.ndarray) and record.dtype.names and field in record.dtype.names:
            value = record[field]
        else:
            value = record[field]
        # scipy wraps scalar/nested values in a 1x1 object array — peel one layer.
        if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
            return value.item()
        return value

    def _extract_array(self, data) -> np.ndarray:
        """Extract numpy array from MATLAB data structure"""
        # Peel object wrappers repeatedly until we hit numeric data.
        while isinstance(data, np.ndarray) and data.dtype == object and data.size == 1:
            data = data.item()
        if hasattr(data, 'flatten'):
            return np.asarray(data).flatten()
        return np.asarray(data).flatten()
    
    def _extract_discharge_portion(self, V: np.ndarray, T: np.ndarray, Qd: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Extract discharge portion of the cycle
        
        Args:
            V: Voltage array
            T: Temperature array
            Qd: Discharge capacity array
            
        Returns:
            Tuple of (V_discharge, T_discharge, Qd_discharge) or None if invalid
        """
        try:
            # Find indices for discharge portion
            start_indices = np.where(V >= self.config.VOLTAGE_RANGE[0])[0]  # >= 3.6V
            end_indices = np.where(V <= self.config.VOLTAGE_RANGE[1])[0]    # <= 2.0V
            
            if len(start_indices) > 0 and len(end_indices) > 0:
                start_idx = start_indices[-1]  # Last index >= 3.6V
                end_idx = end_indices[0]       # First index <= 2.0V
                
                if start_idx < end_idx:
                    # Extract discharge portion and apply smoothing
                    V_discharge = uniform_filter1d(V[start_idx:end_idx+1], size=3)
                    T_discharge = uniform_filter1d(T[start_idx:end_idx+1], size=3)
                    Qd_discharge = uniform_filter1d(Qd[start_idx:end_idx+1], size=3)
                    
                    return V_discharge, T_discharge, Qd_discharge
            
            return None
            
        except Exception as e:
            logger.warning("Error extracting discharge portion: %s", e)
            return None
    
    def linear_interpolation(self, discharge_data: List[Dict[str, List[np.ndarray]]]) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        Interpolate data on voltage range and reshape to 30x30 matrices
        
        Args:
            discharge_data: List of battery discharge data
            
        Returns:
            Tuple of (V_interp, T_interp, Qd_interp) interpolated data
        """
        logger.info("Performing linear interpolation...")
        V_interp, T_interp, Qd_interp = [], [], []
        
        # Create voltage range for interpolation
        volt_range = np.linspace(
            self.config.VOLTAGE_RANGE[0], 
            self.config.VOLTAGE_RANGE[1], 
            self.config.INTERPOLATION_POINTS
        )
        
        for i, battery in enumerate(discharge_data):
            battery_V, battery_T, battery_Qd = [], [], []
            
            for j in range(len(battery['Vd'])):
                try:
                    volt = battery['Vd'][j]
                    temp = battery['Td'][j]
                    qd = battery['QdClipped'][j]
                    
                    # Interpolate data
                    interp_data = self._interpolate_cycle_data(volt, temp, qd, volt_range)
                    
                    if interp_data is not None:
                        V_reshaped, T_reshaped, Qd_reshaped = interp_data
                        
                        battery_V.append(V_reshaped)
                        battery_T.append(T_reshaped)
                        battery_Qd.append(Qd_reshaped)
                        
                except Exception as e:
                    logger.warning("Error interpolating battery %d, cycle %d: %s", i, j, e)
                    continue
            
            V_interp.append(battery_V)
            T_interp.append(battery_T)
            Qd_interp.append(battery_Qd)
            
            if (i + 1) % 5 == 0:
                logger.info("Interpolated %d/%d batteries", i + 1, len(discharge_data))

        logger.info("Linear interpolation completed.")
        return V_interp, T_interp, Qd_interp
    
    def _interpolate_cycle_data(self, volt: np.ndarray, temp: np.ndarray, qd: np.ndarray, volt_range: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Interpolate single cycle data
        
        Args:
            volt: Voltage data
            temp: Temperature data
            qd: Discharge capacity data
            volt_range: Target voltage range
            
        Returns:
            Tuple of reshaped interpolated data or None if failed
        """
        try:
            # Remove duplicates and sort for interpolation
            unique_indices = np.unique(volt, return_index=True)[1]
            volt_unique = volt[unique_indices]
            temp_unique = temp[unique_indices]
            qd_unique = qd[unique_indices]
            
            # Sort by voltage (descending for discharge)
            sort_indices = np.argsort(volt_unique)[::-1]
            volt_sorted = volt_unique[sort_indices]
            temp_sorted = temp_unique[sort_indices]
            qd_sorted = qd_unique[sort_indices]
            
            # Check if we have enough points for interpolation
            if len(volt_sorted) < 2:
                return None
            
            # Create interpolation functions
            f_temp = interpolate.interp1d(
                volt_sorted, temp_sorted, 
                bounds_error=False, 
                fill_value='extrapolate',
                kind='linear'
            )
            f_qd = interpolate.interp1d(
                volt_sorted, qd_sorted, 
                bounds_error=False, 
                fill_value='extrapolate',
                kind='linear'
            )
            
            # Interpolate onto voltage range
            temp_interp = f_temp(volt_range)
            qd_interp = f_qd(volt_range)
            
            # Reshape to 30x30 matrices
            reshape_size = self.config.RESHAPE_SIZE
            V_reshaped = volt_range.reshape(reshape_size, reshape_size)
            T_reshaped = temp_interp.reshape(reshape_size, reshape_size)
            Qd_reshaped = qd_interp.reshape(reshape_size, reshape_size)
            
            return V_reshaped, T_reshaped, Qd_reshaped
            
        except Exception as e:
            logger.warning("Error in cycle interpolation: %s", e)
            return None
    
    def reshape_for_cnn(self, V_interp: List[List[np.ndarray]], T_interp: List[List[np.ndarray]], 
                       Qd_interp: List[List[np.ndarray]], battery_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape data for CNN input and create RUL labels
        
        Args:
            V_interp: Interpolated voltage data
            T_interp: Interpolated temperature data
            Qd_interp: Interpolated discharge capacity data
            battery_indices: Indices of batteries to include
            
        Returns:
            Tuple of (signal_data, rul_data)
        """
        logger.info("Reshaping data for CNN (batteries: %d)...", len(battery_indices))
        
        all_data = []
        all_rul = []
        
        for i in battery_indices:
            if i < len(V_interp) and len(V_interp[i]) > 0:
                try:
                    num_cycles = len(V_interp[i])
                    
                    # Create predictor array (num_cycles, 30, 30, 3)
                    predictor = np.zeros((num_cycles, self.config.RESHAPE_SIZE, self.config.RESHAPE_SIZE, self.config.NUM_CHANNELS))
                    
                    for j in range(num_cycles):
                        predictor[j, :, :, 0] = V_interp[i][j]   # Voltage
                        predictor[j, :, :, 1] = Qd_interp[i][j]  # Discharge capacity
                        predictor[j, :, :, 2] = T_interp[i][j]   # Temperature
                    
                    # Create RUL labels (normalized by max_battery_life)
                    cycles = np.arange(1, num_cycles + 1)
                    rul_battery = (num_cycles + 1 - cycles) / self.config.MAX_BATTERY_LIFE
                    
                    all_data.append(predictor)
                    all_rul.append(rul_battery)
                    
                except Exception as e:
                    logger.warning("Error processing battery %d in reshape: %s", i, e)
                    continue
        
        # Concatenate all data
        if all_data:
            signal_data = np.concatenate(all_data, axis=0)
            rul_data = np.concatenate(all_rul, axis=0)
            
            logger.info("Final data shape: %s", signal_data.shape)
            logger.info("Final RUL shape: %s", rul_data.shape)
        else:
            logger.warning("No valid data found during reshape.")
            signal_data = np.empty((0, self.config.RESHAPE_SIZE, self.config.RESHAPE_SIZE, self.config.NUM_CHANNELS))
            rul_data = np.empty((0,))
        
        return signal_data, rul_data
    
    def split_data_indices(self, num_batteries: int) -> Tuple[List[int], List[int], List[int]]:
        """
        Split battery indices for train/validation/test
        
        Args:
            num_batteries: Total number of batteries
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        test_indices = list(range(
            self.config.TEST_BATTERY_START,
            num_batteries,
            self.config.TEST_BATTERY_STEP,
        ))

        val_indices = list(range(
            self.config.VAL_BATTERY_START,
            num_batteries,
            self.config.VAL_BATTERY_STEP,
        ))

        test_set = set(test_indices)
        val_set = set(val_indices)
        overlap = test_set & val_set
        if overlap:
            raise ValueError(
                f"Val and test battery indices overlap: {sorted(overlap)}. "
                f"Check TEST_BATTERY_START/STEP vs VAL_BATTERY_START/STEP in Config."
            )

        excluded = test_set | val_set
        train_indices = [i for i in range(num_batteries) if i not in excluded]

        if not train_indices:
            raise ValueError("Train split is empty — all batteries went to val/test.")

        logger.info(
            "Data split - Train: %d, Val: %d, Test: %d",
            len(train_indices), len(val_indices), len(test_indices),
        )

        return train_indices, val_indices, test_indices
    
    def normalize_data(self, data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
        """
        Normalize input data
        
        Args:
            data: Input data to normalize
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if method == 'minmax':
            data_min = np.min(data, axis=(0, 1, 2), keepdims=True)
            data_max = np.max(data, axis=(0, 1, 2), keepdims=True)
            normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
            params = {'min': data_min, 'max': data_max, 'method': 'minmax'}
            
        elif method == 'zscore':
            data_mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
            data_std = np.std(data, axis=(0, 1, 2), keepdims=True)
            normalized_data = (data - data_mean) / (data_std + 1e-8)
            params = {'mean': data_mean, 'std': data_std, 'method': 'zscore'}
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized_data, params
    
    def denormalize_data(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """
        Denormalize data using stored parameters
        """
        if params['method'] == 'minmax':
            return data * (params['max'] - params['min']) + params['min']
        elif params['method'] == 'zscore':
            return data * params['std'] + params['mean']
        else:
            raise ValueError(f"Unknown normalization method: {params['method']}")

    def apply_normalization(self, data: np.ndarray, params: Dict) -> np.ndarray:
        """Apply pre-fitted normalization params to new data (val/test/inference)."""
        if params['method'] == 'minmax':
            return (data - params['min']) / (params['max'] - params['min'] + 1e-8)
        if params['method'] == 'zscore':
            return (data - params['mean']) / (params['std'] + 1e-8)
        raise ValueError(f"Unknown normalization method: {params['method']}")

    def save_norm_params(self, params: Dict, path: Optional[str] = None) -> str:
        """Persist normalization params so inference can re-apply them."""
        path = path or self.config.NORM_PARAMS_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {'method': np.array(params['method'])}
        for k, v in params.items():
            if k == 'method':
                continue
            payload[k] = np.asarray(v)
        np.savez(path, **payload)
        logger.info("Saved normalization params to %s", path)
        return path

    def load_norm_params(self, path: Optional[str] = None) -> Dict:
        """Load normalization params previously written by save_norm_params."""
        path = path or self.config.NORM_PARAMS_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"Normalization params not found at {path}")
        with np.load(path, allow_pickle=False) as data:
            params = {key: data[key] for key in data.files}
        params['method'] = str(params['method'])
        logger.info("Loaded normalization params from %s", path)
        return params
