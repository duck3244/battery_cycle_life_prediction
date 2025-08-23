"""
Data loading and downloading utilities for battery cycle life prediction
"""

import os
import requests
import zipfile
import numpy as np
from typing import Optional, List, Dict, Any
from config import Config

class DataLoader:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
    def download_data(self, url: str = None, force_download: bool = False) -> bool:
        """
        Download and extract battery data
        
        Args:
            url: Data URL (uses config default if None)
            force_download: Force download even if file exists
            
        Returns:
            bool: True if successful, False otherwise
        """
        url = url or self.config.DATA_URL
        zip_path = os.path.join(self.config.DATA_DIR, 'batteryDischargeData.zip')
        data_path = os.path.join(self.config.DATA_DIR, self.config.DATA_FILE)
        
        # Create data directory
        self.config.create_directories()
        
        # Check if data already exists
        if os.path.exists(data_path) and not force_download:
            print("Data file already exists. Use force_download=True to re-download.")
            return True
            
        try:
            print("Downloading battery data (this may take a while, ~1.2GB)...")
            
            # Download the zip file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
            
            print("\nExtracting data...")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.DATA_DIR)
            
            # Clean up zip file
            os.remove(zip_path)
            
            print("Data downloaded and extracted successfully!")
            return True
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    
    def load_battery_data(self, data_path: str = None) -> Optional[np.ndarray]:
        """
        Load battery data from .mat file
        
        Args:
            data_path: Path to data file (uses config default if None)
            
        Returns:
            Loaded battery data or None if failed
        """
        data_path = data_path or os.path.join(self.config.DATA_DIR, self.config.DATA_FILE)
        
        try:
            from scipy.io import loadmat
            print(f"Loading battery data from {data_path}...")
            
            mat_data = loadmat(data_path)
            battery_data = mat_data['batteryDischargeData']
            
            print(f"Loaded data for {len(battery_data[0])} batteries")
            return battery_data
            
        except ImportError:
            print("Error: scipy is required to load .mat files. Install it with: pip install scipy")
            return None
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            print("Please run download_data() first or check the file path.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_synthetic_data(self, num_batteries: int = 40) -> List[Dict[str, List[np.ndarray]]]:
        """
        Create synthetic battery data for testing purposes
        
        Args:
            num_batteries: Number of batteries to generate
            
        Returns:
            List of battery discharge data
        """
        print(f"Creating synthetic data for {num_batteries} batteries...")
        
        discharge_data = []
        np.random.seed(self.config.RANDOM_SEED)
        
        for i in range(num_batteries):
            battery = {'Vd': [], 'Td': [], 'QdClipped': []}
            num_cycles = np.random.randint(150, 500)  # Random number of cycles
            
            for j in range(num_cycles):
                # Generate synthetic discharge curves
                n_points = np.random.randint(800, 1200)
                
                # Voltage: decreases from 3.6V to 2.0V with some noise
                voltage = np.linspace(3.6, 2.0, n_points) + np.random.normal(0, 0.01, n_points)
                
                # Temperature: around 25°C with variation
                temperature = 25 + np.random.normal(0, 2, n_points) + 0.1 * (voltage - 2.8)
                
                # Capacity: related to voltage with some degradation over cycles
                base_capacity = 1.1 * (voltage - 2.0) / 1.6
                degradation = j * 0.001  # Capacity degrades over cycles
                capacity = base_capacity * (1 - degradation) + np.random.normal(0, 0.02, n_points)
                
                battery['Vd'].append(voltage)
                battery['Td'].append(temperature)
                battery['QdClipped'].append(capacity)
            
            discharge_data.append(battery)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_batteries} batteries")
        
        print("Synthetic data generation completed!")
        return discharge_data
    
    def get_battery_info(self, discharge_data: List[Dict[str, List[np.ndarray]]]) -> Dict[str, Any]:
        """
        Get information about the loaded battery data
        
        Args:
            discharge_data: Battery discharge data
            
        Returns:
            Dictionary with data information
        """
        num_batteries = len(discharge_data)
        total_cycles = sum(len(battery['Vd']) for battery in discharge_data)
        
        cycles_per_battery = [len(battery['Vd']) for battery in discharge_data]
        
        info = {
            'num_batteries': num_batteries,
            'total_cycles': total_cycles,
            'avg_cycles_per_battery': np.mean(cycles_per_battery),
            'min_cycles': np.min(cycles_per_battery),
            'max_cycles': np.max(cycles_per_battery),
            'cycles_per_battery': cycles_per_battery
        }
        
        return info
    
    def validate_data(self, discharge_data: List[Dict[str, List[np.ndarray]]]) -> bool:
        """
        Validate the loaded battery data
        
        Args:
            discharge_data: Battery discharge data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if not discharge_data:
            print("Error: No data provided")
            return False
        
        try:
            for i, battery in enumerate(discharge_data):
                if not all(key in battery for key in ['Vd', 'Td', 'QdClipped']):
                    print(f"Error: Battery {i} missing required keys")
                    return False
                
                num_cycles = len(battery['Vd'])
                if num_cycles == 0:
                    print(f"Warning: Battery {i} has no cycles")
                    continue
                
                # Check that all measurements have same number of cycles
                if not (len(battery['Td']) == num_cycles and len(battery['QdClipped']) == num_cycles):
                    print(f"Error: Battery {i} has inconsistent cycle counts")
                    return False
                
                # Check data types
                for j in range(min(3, num_cycles)):  # Check first few cycles
                    if not all(isinstance(battery[key][j], np.ndarray) for key in battery.keys()):
                        print(f"Error: Battery {i}, cycle {j} contains non-array data")
                        return False
            
            print("Data validation completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error during data validation: {e}")
            return False
