"""
Model evaluation and performance metrics for battery cycle life prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from config import Config
from utils import setup_logging, sanitize_for_json

logger = setup_logging()

class ModelEvaluator:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # Basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))

        # R-squared
        r2 = float(r2_score(y_true, y_pred))

        # Mean Absolute Percentage Error (avoiding division by zero)
        mask = y_true != 0
        if np.any(mask):
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float('nan')

        # Mean Error (bias)
        me = float(np.mean(y_pred - y_true))

        # Maximum error
        max_error = float(np.max(np.abs(y_true - y_pred)))

        # Explained variance score
        true_var = np.var(y_true)
        explained_var = float(1 - np.var(y_true - y_pred) / true_var) if true_var > 0 else float('nan')
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2,
            'MAPE': mape,
            'Mean_Error': me,
            'Max_Error': max_error,
            'Explained_Variance': explained_var,
            'Prediction_Std': np.std(y_pred),
            'True_Std': np.std(y_true)
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "Model Performance Metrics"):
        """
        Print evaluation metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics
            title: Title for the metrics display
        """
        print(f"\n{title}")
        print("=" * len(title))
        
        for metric_name, value in metrics.items():
            if metric_name == 'MAPE':
                print(f"{metric_name:20s}: {value:.2f}%")
            else:
                print(f"{metric_name:20s}: {value:.4f}")
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 title: str = "Predicted vs Actual Cycle Life",
                                 save_path: str = None) -> plt.Figure:
        """
        Plot predicted vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate and display R²
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=12)
        
        ax.set_xlabel('Actual Cycle Life')
        ax.set_ylabel('Predicted Cycle Life')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Make plot square
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residual Analysis", save_path: str = None) -> plt.Figure:
        """
        Plot residual analysis
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        residuals = y_pred - y_true
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs Actual
        axes[0, 1].scatter(y_true, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Actual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual plot saved to {save_path}")
        
        return fig
    
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray,
                               title: str = "Error Distribution Analysis", 
                               save_path: str = None) -> plt.Figure:
        """
        Plot error distribution analysis
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        errors = np.abs(y_true - y_pred)
        percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        # Absolute error vs actual values
        axes[0, 0].scatter(y_true, errors, alpha=0.6)
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Absolute Error')
        axes[0, 0].set_title('Absolute Error vs Actual Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Percentage error vs actual values
        axes[0, 1].scatter(y_true, percentage_errors, alpha=0.6)
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Percentage Error (%)')
        axes[0, 1].set_title('Percentage Error vs Actual Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of absolute errors
        axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Absolute Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Absolute Errors')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution of percentage errors
        axes[1, 1].hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Percentage Error (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Percentage Errors')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to {save_path}")
        
        return fig
    
    def plot_training_history(self, history: Dict[str, List], 
                            title: str = "Training History",
                            save_path: str = None) -> plt.Figure:
        """
        Plot training and validation loss/metrics history
        
        Args:
            history: Training history dictionary
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Determine the number of subplots needed
        metrics = [key for key in history.keys() if not key.startswith('val_')]
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16)
        
        for i, metric in enumerate(metrics):
            # Plot training metric
            axes[i].plot(history[metric], label=f'Training {metric.upper()}', linewidth=2)
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                axes[i].plot(history[val_metric], label=f'Validation {metric.upper()}', linewidth=2)
            
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'{metric.upper()} History')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        return fig
    
    def create_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                               history: Dict[str, List] = None,
                               model_info: Dict[str, Any] = None,
                               save_path: str = None) -> Dict[str, Any]:
        """
        Create comprehensive evaluation report
        
        Args:
            y_true: True values
            y_pred: Predicted values
            history: Training history (optional)
            model_info: Model information (optional)
            save_path: Path to save the report (optional)
            
        Returns:
            Complete evaluation report dictionary
        """
        print("Creating comprehensive evaluation report...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Create report
        report = {
            'metrics': metrics,
            'data_info': {
                'n_samples': len(y_true),
                'true_mean': np.mean(y_true),
                'true_std': np.std(y_true),
                'true_min': np.min(y_true),
                'true_max': np.max(y_true),
                'pred_mean': np.mean(y_pred),
                'pred_std': np.std(y_pred),
                'pred_min': np.min(y_pred),
                'pred_max': np.max(y_pred)
            }
        }
        
        # Add model info if provided
        if model_info:
            report['model_info'] = model_info
        
        # Add training history if provided
        if history:
            final_metrics = {}
            for key, values in history.items():
                if values:  # Check if list is not empty
                    final_metrics[f'final_{key}'] = values[-1]
                    final_metrics[f'best_{key}'] = min(values) if 'loss' in key else max(values)
            report['training_history'] = final_metrics
        
        # Calculate error percentiles (guard against zero divisor in MAPE)
        errors = np.abs(y_true - y_pred)
        nonzero = y_true != 0
        if np.any(nonzero):
            percentage_errors = np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]) * 100
        else:
            percentage_errors = np.array([])

        report['error_analysis'] = {
            'error_percentiles': {
                f'{p}th': float(np.percentile(errors, p)) for p in (50, 75, 90, 95, 99)
            },
            'percentage_error_percentiles': (
                {f'{p}th': float(np.percentile(percentage_errors, p)) for p in (50, 75, 90, 95, 99)}
                if percentage_errors.size else {}
            ),
        }

        # Print summary
        self.print_metrics(metrics, "Comprehensive Model Performance")

        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(sanitize_for_json(report), f, indent=2)
            logger.info("Evaluation report saved to %s", save_path)

        return report
    
    def compare_models(self, results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      save_path: str = None) -> pd.DataFrame:
        """
        Compare multiple model results
        
        Args:
            results_dict: Dictionary with model names as keys and (y_true, y_pred) tuples as values
            save_path: Path to save comparison results
            
        Returns:
            DataFrame with comparison results
        """
        comparison_results = []
        
        for model_name, (y_true, y_pred) in results_dict.items():
            metrics = self.calculate_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            comparison_results.append(metrics)
        
        df = pd.DataFrame(comparison_results)
        df = df.set_index('Model')
        
        # Sort by RMSE (lower is better)
        df = df.sort_values('RMSE')
        
        print("\nModel Comparison Results:")
        print("=" * 50)
        print(df.round(4))
        
        if save_path:
            df.to_csv(save_path)
            print(f"Comparison results saved to {save_path}")
        
        return df