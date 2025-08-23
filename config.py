"""
Configuration settings for battery cycle life prediction
"""

import os

class Config:
    # Data settings
    DATA_URL = 'https://ssd.mathworks.com/supportfiles/predmaint/batterycyclelifeprediction/v2/batteryDischargeData.zip'
    DATA_DIR = 'data'
    DATA_FILE = 'batteryDischargeData.mat'
    
    # Model settings
    MAX_BATTERY_LIFE = 2000  # Used for output normalization
    VOLTAGE_RANGE = (3.6, 2.0)  # Discharge voltage range
    INTERPOLATION_POINTS = 900
    RESHAPE_SIZE = 30  # 30x30 matrix
    NUM_CHANNELS = 3  # Voltage, Temperature, Discharge Capacity
    
    # Training settings
    BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_PATIENCE = 10
    EARLY_STOPPING_PATIENCE = 10
    
    # Data split indices
    TEST_BATTERY_STEP = 8
    VAL_BATTERY_STEP = 8
    TEST_BATTERY_START = 1  # Every 8th battery starting from 1
    VAL_BATTERY_START = 0   # Every 8th battery starting from 0
    
    # Model architecture
    CONV_FILTERS = [8, 16, 32, 32, 32]
    CONV_KERNEL_SIZE = 3
    POOL_SIZE = 2
    
    # Random seeds
    RANDOM_SEED = 42
    
    # Paths
    MODEL_SAVE_PATH = os.path.join('models', 'battery_model.h5')
    RESULTS_DIR = 'results'
    
    @staticmethod
    def create_directories():
        """Create necessary directories"""
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
