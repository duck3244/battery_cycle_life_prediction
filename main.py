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

class BatteryCycleLifePipeline:
    def __init__(self, config_path=None):
        """Initialize the pipeline with configuration"""
        self.config = Config()
        self.config.create_directories()
        
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
        
        print("Battery Cycle Life Prediction Pipeline Initialized")
        print("=" * 60)
    
    def load_data(self, use_synthetic=True, download_real=False):
        """
        Load battery data
        
        Args:
            use_synthetic: Use synthetic data for testing
            download_real: Download real data from MathWorks
        """
        print("\n1. LOADING DATA")
        print("-" * 30)
        
        if download_real:
            print("Downloading real battery data...")
            if self.data_loader.download_data():
                self.raw_data = self.data_loader.load_battery_data()
                if self.raw_data is not None:
                    self.discharge_data = self.preprocessor.extract_discharge_data(self.raw_data)
                else:
                    print("Failed to load real data, switching to synthetic data")
                    use_synthetic = True
            else:
                print("Failed to download real data, switching to synthetic data")
                use_synthetic = True
        
        if use_synthetic:
            print("Creating synthetic battery data...")
            self.discharge_data = self.data_loader.create_synthetic_data()
        
        # Validate data
        if self.data_loader.validate_data(self.discharge_data):
            # Get data info
            data_info = self.data_loader.get_battery_info(self.discharge_data)
            self.results['data_info'] = data_info
            
            print(f"\nData loaded successfully!")
            print(f"Number of batteries: {data_info['num_batteries']}")
            print(f"Total cycles: {data_info['total_cycles']}")
            print(f"Average cycles per battery: {data_info['avg_cycles_per_battery']:.1f}")
            
            return True
        else:
            print("Data validation failed!")
            return False
    
    def preprocess_data(self):
        """Preprocess the loaded data"""
        print("\n2. PREPROCESSING DATA")
        print("-" * 30)
        
        if self.discharge_data is None:
            print("No data loaded!")
            return False
        
        try:
            # Interpolate data
            print("Performing linear interpolation...")
            V_interp, T_interp, Qd_interp = self.preprocessor.linear_interpolation(self.discharge_data)
            
            # Split data indices
            num_batteries = len(self.discharge_data)
            train_indices, val_indices, test_indices = self.preprocessor.split_data_indices(num_batteries)
            
            # Reshape data for CNN
            print("Reshaping data for CNN...")
            train_data, train_rul = self.preprocessor.reshape_for_cnn(V_interp, T_interp, Qd_interp, train_indices)
            val_data, val_rul = self.preprocessor.reshape_for_cnn(V_interp, T_interp, Qd_interp, val_indices)
            test_data, test_rul = self.preprocessor.reshape_for_cnn(V_interp, T_interp, Qd_interp, test_indices)
            
            # Store processed data
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
                'test_indices': test_indices
            }
            
            print(f"Data preprocessing completed!")
            print(f"Training data shape: {train_data.shape}")
            print(f"Validation data shape: {val_data.shape}")
            print(f"Test data shape: {test_data.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return False
    
    def train_model(self, epochs=None, batch_size=None):
        """Train the CNN model"""
        print("\n3. TRAINING MODEL")
        print("-" * 30)
        
        if not self.processed_data:
            print("Data not preprocessed!")
            return False
        
        try:
            # Create model
            print("Creating CNN model...")
            self.model.create_model()
            print(self.model.get_model_summary())
            
            # Train model
            print("Starting model training...")
            history = self.model.train(
                self.processed_data['train_data'],
                self.processed_data['train_rul'],
                self.processed_data['val_data'],
                self.processed_data['val_rul'],
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Save model
            self.model.save_model()
            
            self.results['training_history'] = history
            
            print("Model training completed!")
            return True
            
        except Exception as e:
            print(f"Error during training: {e}")
            return False
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\n4. EVALUATING MODEL")
        print("-" * 30)
        
        if not self.model.is_trained:
            print("Model not trained!")
            return False
        
        try:
            # Make predictions
            print("Making predictions on test data...")
            test_predictions = self.model.predict(self.processed_data['test_data'])
            test_actual = self.processed_data['test_rul'] * self.config.MAX_BATTERY_LIFE
            
            # Calculate metrics
            metrics = self.evaluator.calculate_metrics(test_actual, test_predictions)
            self.results['test_metrics'] = metrics
            
            # Print results
            self.evaluator.print_metrics(metrics, "Test Set Performance")
            
            # Store predictions for visualization
            self.results['test_predictions'] = test_predictions
            self.results['test_actual'] = test_actual
            
            return True
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return False
    
    def create_visualizations(self, save_plots=True):
        """Create comprehensive visualizations"""
        print("\n5. CREATING VISUALIZATIONS")
        print("-" * 30)
        
        try:
            save_dir = self.config.RESULTS_DIR if save_plots else None
            
            # 1. Dataset overview
            print("Creating dataset overview...")
            self.visualizer.create_summary_dashboard(
                self.discharge_data,
                self.results['data_info'],
                save_path=os.path.join(save_dir, 'dataset_overview.png') if save_dir else None
            )
            
            # 2. Sample battery measurements
            print("Visualizing sample battery measurements...")
            self.visualizer.plot_battery_measurements(
                self.discharge_data,
                battery_idx=0,
                cycle_idx=0,
                save_path=os.path.join(save_dir, 'sample_measurements.png') if save_dir else None
            )
            
            # 3. Interpolated data visualization
            if 'V_interp' in self.processed_data:
                print("Visualizing interpolated data...")
                self.visualizer.plot_interpolated_data(
                    self.processed_data['V_interp'],
                    self.processed_data['T_interp'],
                    self.processed_data['Qd_interp'],
                    battery_idx=0,
                    cycle_idx=0,
                    save_path=os.path.join(save_dir, 'interpolated_data.png') if save_dir else None
                )
            
            # 4. Training history
            if 'training_history' in self.results:
                print("Plotting training history...")
                self.evaluator.plot_training_history(
                    self.results['training_history'],
                    save_path=os.path.join(save_dir, 'training_history.png') if save_dir else None
                )
            
            # 5. Model predictions
            if 'test_predictions' in self.results:
                print("Creating prediction visualizations...")
                
                # Predictions vs actual
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
            
            # 6. Feature maps (if model is available)
            if self.model.model is not None and 'test_data' in self.processed_data:
                print("Extracting and visualizing feature maps...")
                try:
                    sample_data = self.processed_data['test_data'][:1]  # First test sample
                    feature_maps = self.model.get_feature_maps(sample_data)
                    
                    if feature_maps:
                        self.visualizer.plot_feature_maps(
                            feature_maps,
                            sample_idx=0,
                            save_path=os.path.join(save_dir, 'feature_maps.png') if save_dir else None
                        )
                except Exception as e:
                    print(f"Could not create feature maps visualization: {e}")
            
            print("Visualizations completed!")
            if save_plots:
                print(f"All plots saved to {save_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return False
    
    def generate_report(self, save_report=True):
        """Generate comprehensive evaluation report"""
        print("\n6. GENERATING REPORT")
        print("-" * 30)
        
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
                
                print("\nReport generated successfully!")
                return True
            else:
                print("No test results available for report generation")
                return False
                
        except Exception as e:
            print(f"Error generating report: {e}")
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
        print("Starting Battery Cycle Life Prediction Pipeline")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_data(use_synthetic=use_synthetic, download_real=download_real):
            print("Pipeline failed at data loading step")
            return False
        
        # Step 2: Preprocess data
        if not self.preprocess_data():
            print("Pipeline failed at data preprocessing step")
            return False
        
        # Step 3: Train model
        if not self.train_model(epochs=epochs, batch_size=batch_size):
            print("Pipeline failed at model training step")
            return False
        
        # Step 4: Evaluate model
        if not self.evaluate_model():
            print("Pipeline failed at model evaluation step")
            return False
        
        # Step 5: Create visualizations
        if create_plots:
            self.create_visualizations(save_plots=save_results)
        
        # Step 6: Generate report
        if save_results:
            self.generate_report(save_report=save_results)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Summary
        if 'test_metrics' in self.results:
            print(f"\nFinal Model Performance:")
            print(f"RMSE: {self.results['test_metrics']['RMSE']:.2f}")
            print(f"MAE: {self.results['test_metrics']['MAE']:.2f}")
            print(f"MAPE: {self.results['test_metrics']['MAPE']:.2f}%")
            print(f"R² Score: {self.results['test_metrics']['R2_Score']:.4f}")
        
        if save_results:
            print(f"\nResults saved to: {self.config.RESULTS_DIR}")
            print(f"Model saved to: {self.config.MODEL_SAVE_PATH}")
        
        return True
    
    def load_and_predict(self, model_path=None, data=None):
        """
        Load a trained model and make predictions on new data
        
        Args:
            model_path: Path to saved model
            data: New data for prediction
            
        Returns:
            Predictions array
        """
        model_path = model_path or self.config.MODEL_SAVE_PATH
        
        if self.model.load_model(model_path):
            if data is not None:
                predictions = self.model.predict(data)
                return predictions
            else:
                print("No data provided for prediction")
                return None
        else:
            print("Failed to load model")
            return None


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
        
        if success:
            print("\nPipeline completed successfully!")
            sys.exit(0)
        else:
            print("\nPipeline failed!")
            sys.exit(1)
            
    elif args.mode == 'train':
        # Training mode
        if pipeline.load_data(use_synthetic=not args.no_synthetic, download_real=args.use_real_data):
            if pipeline.preprocess_data():
                if pipeline.train_model(epochs=args.epochs, batch_size=args.batch_size):
                    print("Training completed successfully!")
                    sys.exit(0)
        
        print("Training failed!")
        sys.exit(1)
        
    elif args.mode == 'predict':
        # Prediction mode
        print("Prediction mode - implement your data loading logic here")
        # This would require implementing data loading for new predictions
        # predictions = pipeline.load_and_predict(args.model_path, your_data)
        
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()