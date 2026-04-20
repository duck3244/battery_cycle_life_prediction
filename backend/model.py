"""
CNN Model architecture and training for battery cycle life prediction
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D,
    Dense, Flatten, LayerNormalization, ReLU, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Dict, List, Tuple, Optional, Any
import json
from config import Config
from utils import set_global_seed, setup_logging, sanitize_for_json

logger = setup_logging()


class BatteryLifeModel:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.history = None
        self.is_trained = False

        set_global_seed(self.config.RANDOM_SEED)
    
    def create_model(self, input_shape: Tuple[int, int, int] = None) -> Model:
        """
        Create CNN model architecture
        
        Args:
            input_shape: Input shape (height, width, channels)
            
        Returns:
            Compiled Keras model
        """
        if input_shape is None:
            input_shape = (self.config.RESHAPE_SIZE, self.config.RESHAPE_SIZE, self.config.NUM_CHANNELS)
        
        logger.info("Creating model with input shape: %s", input_shape)
        
        # Input layer — normalization is applied in the preprocessing pipeline
        # (DataPreprocessor.normalize_data), so no Rescaling layer is needed here.
        inputs = Input(shape=input_shape)
        x = inputs

        # First convolutional block
        x = Conv2D(self.config.CONV_FILTERS[0], self.config.CONV_KERNEL_SIZE, padding='same')(x)
        x = LayerNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(self.config.POOL_SIZE, strides=2)(x)
        
        # Second convolutional block
        x = Conv2D(self.config.CONV_FILTERS[1], self.config.CONV_KERNEL_SIZE, padding='same')(x)
        x = LayerNormalization()(x)
        x = ReLU()(x)
        x = AveragePooling2D(self.config.POOL_SIZE, strides=2)(x)
        
        # Third convolutional block
        x = Conv2D(self.config.CONV_FILTERS[2], self.config.CONV_KERNEL_SIZE, padding='same')(x)
        x = LayerNormalization()(x)
        x = ReLU()(x)
        
        # Fourth convolutional block
        x = Conv2D(self.config.CONV_FILTERS[3], self.config.CONV_KERNEL_SIZE, padding='same')(x)
        x = LayerNormalization()(x)
        x = ReLU()(x)
        
        # Fifth convolutional block
        x = Conv2D(self.config.CONV_FILTERS[4], self.config.CONV_KERNEL_SIZE, padding='same')(x)
        x = LayerNormalization()(x)
        x = ReLU()(x)
        
        # Output layer
        x = Flatten()(x)
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='BatteryLifePredictionModel')
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='mae',  # Mean Absolute Error
            metrics=['mse', 'mae']
        )
        
        self.model = model
        logger.info("Model created. Total parameters: %s", f"{model.count_params():,}")
        
        return model
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not created yet"
        
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.model.summary()
        return f.getvalue()
    
    def create_callbacks(self, val_data: Tuple[np.ndarray, np.ndarray] = None) -> List:
        """
        Create training callbacks
        
        Args:
            val_data: Validation data tuple (X_val, y_val)
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss' if val_data is not None else 'loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if val_data is not None else 'loss',
            factor=0.5,
            patience=self.config.VALIDATION_PATIENCE,
            min_lr=self.config.LEARNING_RATE * 0.01,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        if not os.path.exists(os.path.dirname(self.config.MODEL_SAVE_PATH)):
            os.makedirs(os.path.dirname(self.config.MODEL_SAVE_PATH))
        
        checkpoint = ModelCheckpoint(
            self.config.MODEL_SAVE_PATH,
            monitor='val_loss' if val_data is not None else 'loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        return callbacks
    
    def train(self, train_data: np.ndarray, train_labels: np.ndarray, 
              val_data: np.ndarray = None, val_labels: np.ndarray = None,
              epochs: int = None, batch_size: int = None) -> Dict[str, List]:
        """
        Train the model
        
        Args:
            train_data: Training data
            train_labels: Training labels
            val_data: Validation data (optional)
            val_labels: Validation labels (optional)
            epochs: Number of epochs (uses config default if None)
            batch_size: Batch size (uses config default if None)
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            raise ValueError("Model not created yet! Call create_model() first.")
        
        epochs = epochs or self.config.EPOCHS
        batch_size = batch_size or self.config.BATCH_SIZE
        
        logger.info("Starting training. Train %s / Labels %s", train_data.shape, train_labels.shape)
        if val_data is not None:
            logger.info("Validation data %s / labels %s", val_data.shape, val_labels.shape)
        
        # Prepare validation data
        validation_data = None
        if val_data is not None and val_labels is not None:
            validation_data = (val_data, val_labels)
        
        # Create callbacks
        callbacks = self.create_callbacks(validation_data)
        
        # Train model
        self.history = self.model.fit(
            train_data, train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Training completed.")
        
        return self.history.history
    
    def predict(self, data: np.ndarray, rescale: bool = True) -> np.ndarray:
        """
        Make predictions.

        Args:
            data: Input data, already normalized the same way as the training set.
            rescale: If True, multiply network output (normalized RUL in [0,1])
                by MAX_BATTERY_LIFE to return predictions in raw cycle units.
                Set to False only when comparing against normalized targets.
        """
        if self.model is None:
            raise ValueError("Model not created yet!")

        if not self.is_trained:
            logger.warning("Model has not been trained yet!")

        logger.info("Making predictions for %d samples...", data.shape[0])
        predictions = self.model.predict(data, verbose=0).flatten()

        if rescale:
            predictions = predictions * self.config.MAX_BATTERY_LIFE

        return predictions
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            test_data: Test data
            test_labels: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not created yet!")
        
        logger.info("Evaluating model...")
        
        # Get model metrics
        model_metrics = self.model.evaluate(test_data, test_labels, verbose=0)
        
        # Make predictions
        predictions = self.predict(test_data, rescale=False)
        
        # Calculate additional metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(test_labels, predictions))
        mae = mean_absolute_error(test_labels, predictions)
        r2 = r2_score(test_labels, predictions)
        
        # Calculate MAPE (avoiding division by zero)
        mask = test_labels != 0
        if np.any(mask):
            mape = float(np.mean(
                np.abs((test_labels[mask] - predictions[mask]) / test_labels[mask])
            ) * 100)
        else:
            mape = float('nan')
        
        metrics = {
            'loss': model_metrics[0],
            'mse': model_metrics[1],
            'mae': model_metrics[2],
            'rmse': rmse,
            'r2_score': r2,
            'mape': mape
        }
        
        return metrics
    
    def save_model(self, filepath: str = None) -> bool:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model (uses config default if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save")
            return False
        
        filepath = filepath or self.config.MODEL_SAVE_PATH
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            self.model.save(filepath)
            
            # Save training history if available
            if self.history is not None:
                history_path = filepath.replace('.h5', '_history.json')
                with open(history_path, 'w') as f:
                    json.dump(sanitize_for_json(self.history.history), f, indent=2)

            logger.info("Model saved to %s", filepath)
            return True

        except Exception as e:
            logger.error("Error saving model: %s", e)
            return False
    
    def load_model(self, filepath: str = None) -> bool:
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model (uses config default if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        filepath = filepath or self.config.MODEL_SAVE_PATH
        
        try:
            if not os.path.exists(filepath):
                logger.error("Model file not found at %s", filepath)
                return False

            self.model = tf.keras.models.load_model(filepath)
            self.is_trained = True

            # Load training history if available
            history_path = filepath.replace('.h5', '_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history_dict = json.load(f)
                    for key, value in history_dict.items():
                        history_dict[key] = np.array(value)
                    self.history = type('History', (), {'history': history_dict})()

            logger.info("Model loaded from %s", filepath)
            return True

        except Exception as e:
            logger.error("Error loading model: %s", e)
            return False
    
    def get_layer_output(self, data: np.ndarray, layer_name: str) -> np.ndarray:
        """
        Get output from a specific layer
        
        Args:
            data: Input data
            layer_name: Name of the layer
            
        Returns:
            Layer output
        """
        if self.model is None:
            raise ValueError("Model not created yet!")
        
        # Create a model that outputs the specified layer
        layer_model = Model(inputs=self.model.input, 
                           outputs=self.model.get_layer(layer_name).output)
        
        return layer_model.predict(data)
    
    def get_feature_maps(self, data: np.ndarray, conv_layer_names: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Get feature maps from convolutional layers
        
        Args:
            data: Input data (single sample)
            conv_layer_names: List of layer names (uses all conv layers if None)
            
        Returns:
            Dictionary of feature maps
        """
        if self.model is None:
            raise ValueError("Model not created yet!")
        
        if conv_layer_names is None:
            conv_layer_names = [layer.name for layer in self.model.layers if 'conv2d' in layer.name]
        
        feature_maps = {}
        for layer_name in conv_layer_names:
            try:
                feature_map = self.get_layer_output(data, layer_name)
                feature_maps[layer_name] = feature_map
            except Exception as e:
                logger.warning("Error getting feature map for %s: %s", layer_name, e)
        
        return feature_maps