"""
Data visualization and analysis tools for battery cycle life prediction
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
from config import Config

class DataVisualizer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_battery_measurements(self, discharge_data: List[Dict[str, List[np.ndarray]]], 
                                battery_idx: int = 0, cycle_idx: int = 0,
                                title: str = "Battery Measurements for One Cycle",
                                save_path: str = None) -> plt.Figure:
        """
        Visualize battery measurements for one cycle
        
        Args:
            discharge_data: Battery discharge data
            battery_idx: Battery index to visualize
            cycle_idx: Cycle index to visualize
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        if (battery_idx >= len(discharge_data) or 
            cycle_idx >= len(discharge_data[battery_idx]['Vd'])):
            print(f"Invalid indices: battery {battery_idx}, cycle {cycle_idx}")
            return None
        
        V = discharge_data[battery_idx]['Vd'][cycle_idx]
        T = discharge_data[battery_idx]['Td'][cycle_idx]
        Qd = discharge_data[battery_idx]['QdClipped'][cycle_idx]
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"{title} (Battery {battery_idx}, Cycle {cycle_idx})", fontsize=14)
        
        # Voltage plot
        axes[0].plot(V, 'b-', linewidth=2, label='Voltage')
        axes[0].set_ylabel('Voltage (V)', fontsize=12)
        axes[0].set_title('Discharge Voltage')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Temperature plot
        axes[1].plot(T, 'r-', linewidth=2, label='Temperature')
        axes[1].set_ylabel('Temperature (°C)', fontsize=12)
        axes[1].set_title('Temperature')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Discharge capacity plot
        axes[2].plot(Qd, 'g-', linewidth=2, label='Discharge Capacity')
        axes[2].set_ylabel('Discharge Capacity (Ah)', fontsize=12)
        axes[2].set_xlabel('Sample Index', fontsize=12)
        axes[2].set_title('Discharge Capacity')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Battery measurements plot saved to {save_path}")
        
        return fig
    
    def plot_interpolated_data(self, V_interp: List[List[np.ndarray]], 
                              T_interp: List[List[np.ndarray]], 
                              Qd_interp: List[List[np.ndarray]],
                              battery_idx: int = 0, cycle_idx: int = 0,
                              title: str = "Interpolated Data (30x30 Matrix)",
                              save_path: str = None) -> plt.Figure:
        """
        Visualize interpolated data as 2D matrices
        
        Args:
            V_interp: Interpolated voltage data
            T_interp: Interpolated temperature data
            Qd_interp: Interpolated discharge capacity data
            battery_idx: Battery index
            cycle_idx: Cycle index
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        if (battery_idx >= len(V_interp) or 
            cycle_idx >= len(V_interp[battery_idx])):
            print(f"Invalid indices: battery {battery_idx}, cycle {cycle_idx}")
            return None
        
        V_matrix = V_interp[battery_idx][cycle_idx]
        T_matrix = T_interp[battery_idx][cycle_idx]
        Qd_matrix = Qd_interp[battery_idx][cycle_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{title} (Battery {battery_idx}, Cycle {cycle_idx})", fontsize=14)
        
        # Voltage matrix
        im1 = axes[0].imshow(V_matrix, cmap='viridis', aspect='equal')
        axes[0].set_title('Voltage (V)')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Temperature matrix
        im2 = axes[1].imshow(T_matrix, cmap='plasma', aspect='equal')
        axes[1].set_title('Temperature (°C)')
        axes[1].set_xlabel('Column')
        axes[1].set_ylabel('Row')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Discharge capacity matrix
        im3 = axes[2].imshow(Qd_matrix, cmap='coolwarm', aspect='equal')
        axes[2].set_title('Discharge Capacity (Ah)')
        axes[2].set_xlabel('Column')
        axes[2].set_ylabel('Row')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Interpolated data plot saved to {save_path}")
        
        return fig
    
    def plot_voltage_temperature_relationship(self, V_interp: List[List[np.ndarray]], 
                                            T_interp: List[List[np.ndarray]],
                                            battery_idx: int = 0, cycle_idx: int = 0,
                                            title: str = "Temperature vs Voltage Relationship",
                                            save_path: str = None) -> plt.Figure:
        """
        Plot temperature and discharge capacity as function of voltage
        
        Args:
            V_interp: Interpolated voltage data
            T_interp: Interpolated temperature data
            battery_idx: Battery index
            cycle_idx: Cycle index
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        if (battery_idx >= len(V_interp) or 
            cycle_idx >= len(V_interp[battery_idx])):
            print(f"Invalid indices: battery {battery_idx}, cycle {cycle_idx}")
            return None
        
        # Flatten matrices to 1D arrays
        voltage = V_interp[battery_idx][cycle_idx].flatten()
        temperature = T_interp[battery_idx][cycle_idx].flatten()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(voltage, temperature, c=temperature, cmap='coolwarm', alpha=0.6)
        ax.set_xlabel('Voltage (V)', fontsize=12)
        ax.set_ylabel('Temperature (°C)', fontsize=12)
        ax.set_title(f"{title} (Battery {battery_idx}, Cycle {cycle_idx})")
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Temperature (°C)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Voltage-temperature relationship plot saved to {save_path}")
        
        return fig
    
    def plot_cycle_life_distribution(self, discharge_data: List[Dict[str, List[np.ndarray]]],
                                   title: str = "Battery Cycle Life Distribution",
                                   save_path: str = None) -> plt.Figure:
        """
        Plot distribution of cycle life across all batteries
        
        Args:
            discharge_data: Battery discharge data
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        # Get cycle counts for each battery
        cycle_counts = [len(battery['Vd']) for battery in discharge_data]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=14)
        
        # Histogram
        axes[0].hist(cycle_counts, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Number of Cycles')
        axes[0].set_ylabel('Number of Batteries')
        axes[0].set_title('Histogram of Cycle Life')
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_cycles = np.mean(cycle_counts)
        std_cycles = np.std(cycle_counts)
        axes[0].axvline(mean_cycles, color='red', linestyle='--', 
                       label=f'Mean: {mean_cycles:.1f}')
        axes[0].axvline(mean_cycles + std_cycles, color='orange', linestyle='--', 
                       label=f'Mean + Std: {mean_cycles + std_cycles:.1f}')
        axes[0].axvline(mean_cycles - std_cycles, color='orange', linestyle='--',
                       label=f'Mean - Std: {mean_cycles - std_cycles:.1f}')
        axes[0].legend()
        
        # Box plot
        axes[1].boxplot(cycle_counts, vert=True)
        axes[1].set_ylabel('Number of Cycles')
        axes[1].set_title('Box Plot of Cycle Life')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
        Mean: {mean_cycles:.1f}
        Std: {std_cycles:.1f}
        Min: {min(cycle_counts)}
        Max: {max(cycle_counts)}
        Median: {np.median(cycle_counts):.1f}"""
        
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cycle life distribution plot saved to {save_path}")
        
        return fig
    
    def plot_data_statistics(self, data_info: Dict[str, Any],
                           title: str = "Dataset Statistics",
                           save_path: str = None) -> plt.Figure:
        """
        Plot dataset statistics
        
        Args:
            data_info: Dictionary containing dataset information
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Battery count
        axes[0, 0].bar(['Total Batteries'], [data_info['num_batteries']], color='skyblue')
        axes[0, 0].set_title('Number of Batteries')
        axes[0, 0].set_ylabel('Count')
        
        # Total cycles
        axes[0, 1].bar(['Total Cycles'], [data_info['total_cycles']], color='lightcoral')
        axes[0, 1].set_title('Total Number of Cycles')
        axes[0, 1].set_ylabel('Count')
        
        # Cycles per battery statistics
        cycle_stats = ['Mean', 'Min', 'Max']
        cycle_values = [data_info['avg_cycles_per_battery'], 
                       data_info['min_cycles'], 
                       data_info['max_cycles']]
        axes[1, 0].bar(cycle_stats, cycle_values, color=['green', 'red', 'blue'])
        axes[1, 0].set_title('Cycles per Battery Statistics')
        axes[1, 0].set_ylabel('Number of Cycles')
        
        # Cycles per battery distribution
        axes[1, 1].hist(data_info['cycles_per_battery'], bins=15, alpha=0.7, 
                       color='purple', edgecolor='black')
        axes[1, 1].set_title('Distribution of Cycles per Battery')
        axes[1, 1].set_xlabel('Number of Cycles')
        axes[1, 1].set_ylabel('Number of Batteries')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dataset statistics plot saved to {save_path}")
        
        return fig
    
    def plot_feature_maps(self, feature_maps: Dict[str, np.ndarray], 
                         sample_idx: int = 0,
                         max_filters: int = 8,
                         title: str = "Feature Maps Visualization",
                         save_path: str = None) -> plt.Figure:
        """
        Visualize feature maps from convolutional layers
        
        Args:
            feature_maps: Dictionary of feature maps from different layers
            sample_idx: Sample index to visualize
            max_filters: Maximum number of filters to show per layer
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        n_layers = len(feature_maps)
        
        fig, axes = plt.subplots(n_layers, max_filters, 
                                figsize=(max_filters * 2, n_layers * 2))
        fig.suptitle(f"{title} (Sample {sample_idx})", fontsize=16)
        
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx, (layer_name, feature_map) in enumerate(feature_maps.items()):
            n_filters = min(feature_map.shape[-1], max_filters)
            
            for filter_idx in range(max_filters):
                ax = axes[layer_idx, filter_idx]
                
                if filter_idx < n_filters:
                    # Show the feature map
                    im = ax.imshow(feature_map[sample_idx, :, :, filter_idx], 
                                  cmap='viridis', aspect='equal')
                    ax.set_title(f'{layer_name}\nFilter {filter_idx}', fontsize=8)
                else:
                    # Hide unused subplots
                    ax.axis('off')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature maps plot saved to {save_path}")
        
        return fig
    
    def plot_data_preprocessing_pipeline(self, original_data: np.ndarray,
                                       interpolated_data: np.ndarray,
                                       battery_idx: int = 0, cycle_idx: int = 0,
                                       title: str = "Data Preprocessing Pipeline",
                                       save_path: str = None) -> plt.Figure:
        """
        Visualize the data preprocessing pipeline
        
        Args:
            original_data: Original raw data
            interpolated_data: Interpolated and reshaped data
            battery_idx: Battery index
            cycle_idx: Cycle index
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{title} (Battery {battery_idx}, Cycle {cycle_idx})", fontsize=14)
        
        # Original data (assuming it's voltage, temperature, capacity)
        for i, (data, name, color) in enumerate(zip([original_data[0], original_data[1], original_data[2]], 
                                                   ['Voltage', 'Temperature', 'Capacity'],
                                                   ['blue', 'red', 'green'])):
            axes[0, i].plot(data, color=color, linewidth=2)
            axes[0, i].set_title(f'Original {name}')
            axes[0, i].set_xlabel('Sample Index')
            axes[0, i].set_ylabel(name)
            axes[0, i].grid(True, alpha=0.3)
        
        # Interpolated data (30x30 matrices)
        cmaps = ['viridis', 'plasma', 'coolwarm']
        for i, (data, name, cmap) in enumerate(zip([interpolated_data[:, :, 0], 
                                                   interpolated_data[:, :, 2], 
                                                   interpolated_data[:, :, 1]], 
                                                  ['Voltage', 'Temperature', 'Capacity'],
                                                  cmaps)):
            im = axes[1, i].imshow(data, cmap=cmap, aspect='equal')
            axes[1, i].set_title(f'Interpolated {name} (30x30)')
            axes[1, i].set_xlabel('Column')
            axes[1, i].set_ylabel('Row')
            plt.colorbar(im, ax=axes[1, i], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Preprocessing pipeline plot saved to {save_path}")
        
        return fig
    
    def create_summary_dashboard(self, discharge_data: List[Dict[str, List[np.ndarray]]],
                               data_info: Dict[str, Any],
                               title: str = "Battery Dataset Summary Dashboard",
                               save_path: str = None) -> plt.Figure:
        """
        Create a comprehensive summary dashboard
        
        Args:
            discharge_data: Battery discharge data
            data_info: Dataset information
            title: Plot title
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(title, fontsize=20)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Dataset overview
        ax1 = fig.add_subplot(gs[0, 0])
        overview_data = [data_info['num_batteries'], data_info['total_cycles']]
        overview_labels = ['Batteries', 'Total Cycles']
        ax1.bar(overview_labels, overview_data, color=['skyblue', 'lightcoral'])
        ax1.set_title('Dataset Overview')
        ax1.set_ylabel('Count')
        
        # 2. Cycle life distribution
        ax2 = fig.add_subplot(gs[0, 1:3])
        cycle_counts = data_info['cycles_per_battery']
        ax2.hist(cycle_counts, bins=15, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Number of Cycles')
        ax2.set_ylabel('Number of Batteries')
        ax2.set_title('Battery Cycle Life Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistics summary
        ax3 = fig.add_subplot(gs[0, 3])
        stats_text = f"""Dataset Statistics:
        
        Batteries: {data_info['num_batteries']}
        Total Cycles: {data_info['total_cycles']}
        
        Cycles per Battery:
        Mean: {data_info['avg_cycles_per_battery']:.1f}
        Min: {data_info['min_cycles']}
        Max: {data_info['max_cycles']}
        Std: {np.std(cycle_counts):.1f}
        
        Data Points: {data_info['total_cycles'] * 900}
        """
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax3.axis('off')
        ax3.set_title('Statistics Summary')
        
        # 4. Sample battery measurements (first battery, first cycle)
        if len(discharge_data) > 0 and len(discharge_data[0]['Vd']) > 0:
            V = discharge_data[0]['Vd'][0]
            T = discharge_data[0]['Td'][0]
            Qd = discharge_data[0]['QdClipped'][0]
            
            # Voltage
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.plot(V, 'b-', linewidth=2)
            ax4.set_title('Sample Voltage Profile')
            ax4.set_ylabel('Voltage (V)')
            ax4.grid(True, alpha=0.3)
            
            # Temperature
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.plot(T, 'r-', linewidth=2)
            ax5.set_title('Sample Temperature Profile')
            ax5.set_ylabel('Temperature (°C)')
            ax5.grid(True, alpha=0.3)
            
            # Capacity
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.plot(Qd, 'g-', linewidth=2)
            ax6.set_title('Sample Capacity Profile')
            ax6.set_ylabel('Capacity (Ah)')
            ax6.grid(True, alpha=0.3)
            
            # Voltage range analysis
            ax7 = fig.add_subplot(gs[1, 3])
            all_voltages = []
            for battery in discharge_data:
                for cycle_data in battery['Vd']:
                    all_voltages.extend(cycle_data)
            ax7.hist(all_voltages, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax7.set_xlabel('Voltage (V)')
            ax7.set_ylabel('Frequency')
            ax7.set_title('Voltage Distribution (All Data)')
            ax7.grid(True, alpha=0.3)
        
        # 5. Data processing info
        ax8 = fig.add_subplot(gs[2, :])
        processing_text = f"""
        Data Processing Pipeline:
        
        1. Raw Data: {data_info['num_batteries']} batteries with varying cycle lengths
        2. Discharge Extraction: Extract discharge portion (3.6V → 2.0V) and apply smoothing
        3. Linear Interpolation: Interpolate to 900 points uniformly distributed over voltage range
        4. Reshape: Convert to 30×30 matrices (3 channels: Voltage, Temperature, Capacity)
        5. CNN Input: Stack as 30×30×3 images for convolutional neural network
        6. Labels: Remaining Useful Life (RUL) normalized by maximum battery life ({self.config.MAX_BATTERY_LIFE})
        """
        ax8.text(0.05, 0.9, processing_text, transform=ax8.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax8.axis('off')
        ax8.set_title('Data Processing Pipeline', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary dashboard saved to {save_path}")
        
        return fig