"""
Main script to run the complete battery cycle life prediction pipeline
"""

import os
import sys
import numpy as np
import argparse
from datetime import datetime

# Import custom modules
from config import Config
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model import BatteryLifeModel
from evaluator import ModelEvaluator
from visualizer import DataVisualizer
from utils import set_global_seed, setup_logging

logger = setup_logging(level=Config.LOG_LEVEL)

class BatteryCycleLifePipeline:
    def __init__(self, config_path=None):
        """Initialize the pipeline with configuration"""
        self.config = Config()
        self.config.create_directories()
        set_global_seed(self.config.RANDOM_SEED)
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.model = BatteryLifeModel(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = DataVisualizer(self.config)
        
        # Data storage
        self.raw_data = None
        self.discharge_data = None
        self.processed_data = {}
        self.results = {}
        
        logger.info("Battery Cycle Life Prediction Pipeline Initialized")
    
    def load_data(self, use_synthetic=True, download_real=False):
        """
        Load battery data
        
        Args:
            use_synthetic: Use synthetic data for testing
            download_real: Download real data from MathWorks
        """
        logger.info("=== Step 1: Loading data ===")

        if download_real:
            logger.info("Downloading real battery data...")
            if self.data_loader.download_data():
                self.raw_data = self.data_loader.load_battery_data()
                if self.raw_data is not None:
                    self.discharge_data = self.preprocessor.extract_discharge_data(self.raw_data)
                else:
                    logger.warning("Failed to load real data, switching to synthetic data")
                    use_synthetic = True
            else:
                logger.warning("Failed to download real data, switching to synthetic data")
                use_synthetic = True

        if use_synthetic:
            self.discharge_data = self.data_loader.create_synthetic_data()

        if not self.data_loader.validate_data(self.discharge_data):
            logger.error("Data validation failed")
            return False

        data_info = self.data_loader.get_battery_info(self.discharge_data)
        self.results['data_info'] = data_info
        logger.info(
            "Data loaded. batteries=%d total_cycles=%d avg_cycles=%.1f",
            data_info['num_batteries'], data_info['total_cycles'],
            data_info['avg_cycles_per_battery'],
        )
        return True
    
    def preprocess_data(self):
        """Preprocess the loaded data"""
        logger.info("=== Step 2: Preprocessing data ===")

        if self.discharge_data is None:
            raise RuntimeError("No data loaded — call load_data() first.")
        
        # Interpolate data
        V_interp, T_interp, Qd_interp = self.preprocessor.linear_interpolation(self.discharge_data)

        # Split data indices (raises on overlap)
        num_batteries = len(self.discharge_data)
        train_indices, val_indices, test_indices = self.preprocessor.split_data_indices(num_batteries)

        # Reshape data for CNN
        train_data, train_rul = self.preprocessor.reshape_for_cnn(V_interp, T_interp, Qd_interp, train_indices)
        val_data, val_rul = self.preprocessor.reshape_for_cnn(V_interp, T_interp, Qd_interp, val_indices)
        test_data, test_rul = self.preprocessor.reshape_for_cnn(V_interp, T_interp, Qd_interp, test_indices)

        if train_data.size == 0:
            raise RuntimeError("Training data is empty after preprocessing.")

        # Fit normalization on train only, apply to val/test, persist params
        train_data, norm_params = self.preprocessor.normalize_data(
            train_data, method=self.config.NORMALIZATION_METHOD
        )
        if val_data.size:
            val_data = self.preprocessor.apply_normalization(val_data, norm_params)
        if test_data.size:
            test_data = self.preprocessor.apply_normalization(test_data, norm_params)
        self.preprocessor.save_norm_params(norm_params)

        self.processed_data = {
            'V_interp': V_interp,
            'T_interp': T_interp,
            'Qd_interp': Qd_interp,
            'train_data': train_data,
            'train_rul': train_rul,
            'val_data': val_data,
            'val_rul': val_rul,
            'test_data': test_data,
            'test_rul': test_rul,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'norm_params': norm_params,
        }

        logger.info("Preprocessing done. Train %s, Val %s, Test %s",
                    train_data.shape, val_data.shape, test_data.shape)
        return True
    
    def train_model(self, epochs=None, batch_size=None):
        """Train the CNN model"""
        logger.info("=== Step 3: Training model ===")
        
        if not self.processed_data:
            raise RuntimeError("Data not preprocessed — call preprocess_data() first.")

        self.model.create_model()
        logger.debug(self.model.get_model_summary())

        history = self.model.train(
            self.processed_data['train_data'],
            self.processed_data['train_rul'],
            self.processed_data['val_data'],
            self.processed_data['val_rul'],
            epochs=epochs,
            batch_size=batch_size,
        )

        self.model.save_model()
        self.results['training_history'] = history
        logger.info("Model training completed.")
        return True
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        logger.info("=== Step 4: Evaluating model ===")
        
        if not self.model.is_trained:
            raise RuntimeError("Model is not trained — run train_model() first.")

        test_data = self.processed_data['test_data']
        if test_data.size == 0:
            logger.warning("Test set is empty — skipping evaluation.")
            return False

        test_predictions = self.model.predict(test_data)
        test_actual = self.processed_data['test_rul'] * self.config.MAX_BATTERY_LIFE

        metrics = self.evaluator.calculate_metrics(test_actual, test_predictions)
        self.results['test_metrics'] = metrics
        self.evaluator.print_metrics(metrics, "Test Set Performance")

        self.results['test_predictions'] = test_predictions
        self.results['test_actual'] = test_actual
        return True
    
    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        logger.info("=== Step 5: Creating visualizations ===")

        try:
            save_dir = self.config.RESULTS_DIR if save_plots else None

            self.visualizer.create_summary_dashboard(
                self.discharge_data,
                self.results['data_info'],
                save_path=os.path.join(save_dir, 'dataset_overview.png') if save_dir else None
            )
            
            self.visualizer.plot_battery_measurements(
                self.discharge_data,
                battery_idx=0,
                cycle_idx=0,
                save_path=os.path.join(save_dir, 'sample_measurements.png') if save_dir else None
            )
            
            if 'V_interp' in self.processed_data:
                self.visualizer.plot_interpolated_data(
                    self.processed_data['V_interp'],
                    self.processed_data['T_interp'],
                    self.processed_data['Qd_interp'],
                    battery_idx=0,
                    cycle_idx=0,
                    save_path=os.path.join(save_dir, 'interpolated_data.png') if save_dir else None
                )
            
            if 'training_history' in self.results:
                self.evaluator.plot_training_history(
                    self.results['training_history'],
                    save_path=os.path.join(save_dir, 'training_history.png') if save_dir else None
                )
            
            if 'test_predictions' in self.results:
                self.evaluator.plot_predictions_vs_actual(
                    self.results['test_actual'],
                    self.results['test_predictions'],
                    save_path=os.path.join(save_dir, 'predictions_vs_actual.png') if save_dir else None
                )
                
                # Residual analysis
                self.evaluator.plot_residuals(
                    self.results['test_actual'],
                    self.results['test_predictions'],
                    save_path=os.path.join(save_dir, 'residual_analysis.png') if save_dir else None
                )
                
                # Error distribution
                self.evaluator.plot_error_distribution(
                    self.results['test_actual'],
                    self.results['test_predictions'],
                    save_path=os.path.join(save_dir, 'error_distribution.png') if save_dir else None
                )
            
            if self.model.model is not None and 'test_data' in self.processed_data:
                try:
                    sample_data = self.processed_data['test_data'][:1]
                    feature_maps = self.model.get_feature_maps(sample_data)
                    if feature_maps:
                        self.visualizer.plot_feature_maps(
                            feature_maps,
                            sample_idx=0,
                            save_path=os.path.join(save_dir, 'feature_maps.png') if save_dir else None
                        )
                except Exception as e:
                    logger.warning("Could not create feature maps visualization: %s", e)

            if save_plots:
                logger.info("Plots saved to %s", save_dir)
            return True
            
        except Exception as e:
            logger.warning("Error creating visualizations: %s", e)
            return False
    
    def generate_report(self, save_report=True):
        """Generate comprehensive evaluation report"""
        logger.info("=== Step 6: Generating report ===")

        try:
            if 'test_predictions' in self.results and 'test_actual' in self.results:
                report_path = os.path.join(self.config.RESULTS_DIR, 'evaluation_report.json') if save_report else None
                
                # Create comprehensive report
                report = self.evaluator.create_evaluation_report(
                    self.results['test_actual'],
                    self.results['test_predictions'],
                    history=self.results.get('training_history'),
                    model_info={
                        'total_parameters': self.model.model.count_params() if self.model.model else None,
                        'architecture': 'CNN with 5 conv layers + layer normalization',
                        'input_shape': f"{self.config.RESHAPE_SIZE}x{self.config.RESHAPE_SIZE}x{self.config.NUM_CHANNELS}",
                        'optimizer': 'Adam',
                        'loss_function': 'Mean Absolute Error',
                        'batch_size': self.config.BATCH_SIZE,
                        'max_epochs': self.config.EPOCHS
                    },
                    save_path=report_path
                )
                
                self.results['evaluation_report'] = report
                logger.info("Report generated.")
                return True
            else:
                logger.warning("No test results available for report generation")
                return False
                
        except Exception as e:
            logger.error("Error generating report: %s", e)
            return False
    
    def run_complete_pipeline(self, use_synthetic=True, download_real=False, 
                            create_plots=True, save_results=True,
                            epochs=None, batch_size=None):
        """
        Run the complete pipeline
        
        Args:
            use_synthetic: Use synthetic data
            download_real: Download real data
            create_plots: Create visualization plots
            save_results: Save results to files
            epochs: Training epochs
            batch_size: Training batch size
        """
        logger.info("Starting pipeline at %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        if not self.load_data(use_synthetic=use_synthetic, download_real=download_real):
            logger.error("Pipeline failed at data loading step")
            return False

        if not self.preprocess_data():
            logger.error("Pipeline failed at data preprocessing step")
            return False

        if not self.train_model(epochs=epochs, batch_size=batch_size):
            logger.error("Pipeline failed at model training step")
            return False

        if not self.evaluate_model():
            logger.error("Pipeline failed at model evaluation step")
            return False

        if create_plots:
            self.create_visualizations(save_plots=save_results)

        if save_results:
            self.generate_report(save_report=save_results)

        logger.info("Pipeline completed successfully.")
        if 'test_metrics' in self.results:
            m = self.results['test_metrics']
            logger.info(
                "Final test metrics — RMSE=%.2f MAE=%.2f MAPE=%.2f%% R2=%.4f",
                m['RMSE'], m['MAE'], m['MAPE'], m['R2_Score'],
            )
        if save_results:
            logger.info("Results dir: %s | Model: %s",
                        self.config.RESULTS_DIR, self.config.MODEL_SAVE_PATH)
        return True
    
    def load_and_predict(self, model_path=None, data=None):
        """Low-level helper: load model and predict on an already-normalized array."""
        model_path = model_path or self.config.MODEL_SAVE_PATH
        if not self.model.load_model(model_path):
            raise RuntimeError(f"Failed to load model from {model_path}")
        if data is None:
            raise ValueError("data is required")
        return self.model.predict(data)

    def predict_from_mat(self, mat_path: str, model_path: str = None,
                         norm_path: str = None, save_csv: bool = True):
        """
        End-to-end prediction: load a .mat file of raw battery data,
        run the same preprocessing + normalization used in training, and
        return (predictions, per-battery output).

        Args:
            mat_path: Path to a .mat file in the same format as the training data.
            model_path: Optional override for model checkpoint path.
            norm_path: Optional override for normalization params path.
            save_csv: If True, write predictions to results/predictions.csv.
        """
        model_path = model_path or self.config.MODEL_SAVE_PATH
        norm_path = norm_path or self.config.NORM_PARAMS_PATH

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Input .mat file not found: {mat_path}")

        # 1. Load raw data
        raw = self.data_loader.load_battery_data(mat_path)
        if raw is None:
            raise RuntimeError(f"Could not read battery data from {mat_path}")

        discharge = self.preprocessor.extract_discharge_data(raw)
        V, T, Qd = self.preprocessor.linear_interpolation(discharge)

        # 2. Reshape every battery (use all indices since this is inference)
        indices = list(range(len(discharge)))
        data, _ = self.preprocessor.reshape_for_cnn(V, T, Qd, indices)
        if data.size == 0:
            raise RuntimeError("No usable cycles found in input data.")

        # 3. Apply saved normalization params
        norm_params = self.preprocessor.load_norm_params(norm_path)
        data = self.preprocessor.apply_normalization(data, norm_params)

        # 4. Load model and predict (rescale to raw cycle count units)
        if not self.model.load_model(model_path):
            raise RuntimeError(f"Failed to load model from {model_path}")
        predictions = self.model.predict(data, rescale=True)

        logger.info("Generated %d predictions from %s", len(predictions), mat_path)

        if save_csv:
            out_path = os.path.join(self.config.RESULTS_DIR, 'predictions.csv')
            os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
            np.savetxt(out_path, predictions, delimiter=',',
                       header='predicted_rul_cycles', comments='')
            logger.info("Predictions written to %s", out_path)

        return predictions


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Battery Cycle Life Prediction Pipeline')
    
    parser.add_argument('--mode', choices=['train', 'predict', 'full'], default='full',
                       help='Pipeline mode: train, predict, or full pipeline')
    parser.add_argument('--use-real-data', action='store_true',
                       help='Download and use real battery data')
    parser.add_argument('--no-synthetic', action='store_true',
                       help='Do not use synthetic data as fallback')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Training batch size')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating visualization plots')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to saved model (for predict mode)')
    parser.add_argument('--input-mat', type=str, default=None,
                       help='Path to a .mat file with new battery data (predict mode)')
    parser.add_argument('--norm-path', type=str, default=None,
                       help='Path to normalization params (.npz) for predict mode')

    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BatteryCycleLifePipeline()
    
    if args.mode == 'full':
        # Run complete pipeline
        success = pipeline.run_complete_pipeline(
            use_synthetic=not args.no_synthetic,
            download_real=args.use_real_data,
            create_plots=not args.no_plots,
            save_results=not args.no_save,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        sys.exit(0 if success else 1)

    elif args.mode == 'train':
        ok = (
            pipeline.load_data(use_synthetic=not args.no_synthetic, download_real=args.use_real_data)
            and pipeline.preprocess_data()
            and pipeline.train_model(epochs=args.epochs, batch_size=args.batch_size)
        )
        if ok:
            logger.info("Training completed successfully.")
            sys.exit(0)
        logger.error("Training failed.")
        sys.exit(1)
        
    elif args.mode == 'predict':
        if not args.input_mat:
            logger.error("--input-mat is required for predict mode")
            sys.exit(2)
        try:
            predictions = pipeline.predict_from_mat(
                mat_path=args.input_mat,
                model_path=args.model_path,
                norm_path=args.norm_path,
            )
            logger.info("First 5 predictions (cycles): %s", predictions[:5].tolist())
            sys.exit(0)
        except Exception as e:
            logger.exception("Predict mode failed: %s", e)
            sys.exit(1)

    else:
        logger.error("Unknown mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()