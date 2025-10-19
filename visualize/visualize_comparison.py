import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

class ShockTubeComparison:
    """Class for comparing analytical and numerical shock tube solutions."""
    
    def __init__(self, results_dir="results", output_dir="plots"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.colors = {
            'analytical': '#DC143C',  # Red for analytical
            'numerical': '#1f77b4'    # Blue for numerical (Lax-Wendroff)
        }
    
    def load_analytical_data(self, filename="results/analytic.dat"):
        """Load analytical solution data."""
        filepath = Path(filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Analytical data file not found: {filepath}")
        
        data = np.loadtxt(filepath, skiprows=1)
        
        # Variables = x, rho, u, p, e, Et, T, c, M, h
        analytical_data = {
            'x': data[:, 0],
            'rho': data[:, 1],
            'u': data[:, 2], 
            'p': data[:, 3],
            'e': data[:, 4],
            'Et': data[:, 5],
            'T': data[:, 6],
            'c': data[:, 7],
            'M': data[:, 8],
            'h': data[:, 9]
        }
        
        return analytical_data
    
    def load_numerical_data(self, filename):
        """Load numerical solution data."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Numerical data file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
            if header_line.startswith('variables'):
                variables = header_line.split('=')[1].strip().split(', ')
                variables = [var.strip() for var in variables]
            else:
                variables = ['position', 'density', 'velocity', 'pressure', 
                           'momentum', 'energy', 'totalEnergy', 'temperature', 
                           'soundVelocity', 'machNumber', 'enthalpy']
        
        
        data = np.loadtxt(filepath, skiprows=1)
        numerical_data = {}
        for i, var in enumerate(variables):
            numerical_data[var] = data[:, i]
        
        if 'position' in numerical_data:
            numerical_data['x'] = numerical_data['position']
        if 'density' in numerical_data:
            numerical_data['rho'] = numerical_data['density']
        if 'pressure' in numerical_data:
            numerical_data['p'] = numerical_data['pressure']
        if 'velocity' in numerical_data:
            numerical_data['u'] = numerical_data['velocity']
        if 'temperature' in numerical_data:
            numerical_data['T'] = numerical_data['temperature']
        if 'machNumber' in numerical_data:
            numerical_data['M'] = numerical_data['machNumber']
        
        return numerical_data
    
    def interpolate_to_common_grid(self, analytical_data, numerical_data):
        x_analytical = analytical_data['x']
        x_numerical = numerical_data['x']
        
        numerical_interp = {}
        for key in ['rho', 'u', 'p', 'T']:
            if key in numerical_data:
                numerical_interp[key] = np.interp(x_analytical, x_numerical, numerical_data[key])
        
        return x_analytical, analytical_data, numerical_interp
    
    
    def plot_density_comparison(self, analytical_data, numerical_data):
        x, analytical, numerical = self.interpolate_to_common_grid(analytical_data, numerical_data)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(x, analytical['rho'], color=self.colors['analytical'], 
               linewidth=2.5, label='Analytical', alpha=0.9)
        ax.plot(x, numerical['rho'], color=self.colors['numerical'], 
               linewidth=2.0, label='Lax-Wendroff', alpha=0.8)
        
    
        ax.set_xlabel('Length of shock tube (m)', fontsize=14)
        ax.set_ylabel('Density (kg/m³)', fontsize=14)
        ax.set_title('Density variation along Shock Tube', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)
        
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        plt.tight_layout()
        
        filename = self.output_dir / 'density_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        return fig
    
    
    def plot_pressure_comparison(self, analytical_data, numerical_data):
        x, analytical, numerical = self.interpolate_to_common_grid(analytical_data, numerical_data)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
        ax.plot(x, analytical['p'], color=self.colors['analytical'], 
               linewidth=2.5, label='Analytical', alpha=0.9)
        ax.plot(x, numerical['p'], color=self.colors['numerical'], 
               linewidth=2.0, label='Lax-Wendroff', alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Length of shock tube (m)', fontsize=14)
        ax.set_ylabel('Pressure (N/m²)', fontsize=14)
        ax.set_title('Pressure variation along Shock Tube', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)
        
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        
        plt.tight_layout()
        
        filename = self.output_dir / 'pressure_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        return fig
    
    
    def plot_velocity_comparison(self, analytical_data, numerical_data):
        x, analytical, numerical = self.interpolate_to_common_grid(analytical_data, numerical_data)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(x, analytical['u'], color=self.colors['analytical'], 
               linewidth=2.5, label='Analytical', alpha=0.9)
        ax.plot(x, numerical['u'], color=self.colors['numerical'], 
               linewidth=2.0, label='Lax-Wendroff', alpha=0.8)
        
        ax.set_xlabel('Length of shock tube (m)', fontsize=14)
        ax.set_ylabel('Velocity (m/s)', fontsize=14)
        ax.set_title('Velocity variation along Shock Tube', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)
        
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        
        plt.tight_layout()
        
        filename = self.output_dir / 'velocity_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        return fig
    
    
    def plot_temperature_comparison(self, analytical_data, numerical_data):
        """Plot temperature comparison."""
        x, analytical, numerical = self.interpolate_to_common_grid(analytical_data, numerical_data)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.plot(x, analytical['T'], color=self.colors['analytical'], 
               linewidth=2.5, label='Analytical', alpha=0.9)
        ax.plot(x, numerical['T'], color=self.colors['numerical'], 
               linewidth=2.0, label='Lax-Wendroff', alpha=0.8)
        
        ax.set_xlabel('Length of shock tube (m)', fontsize=14)
        ax.set_ylabel('Temperature (K)', fontsize=14)
        ax.set_title('Temperature variation along Shock Tube', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)
        
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        
        plt.tight_layout()
        
        filename = self.output_dir / 'temperature_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        
        return fig
    
    
    
    
    def generate_all_comparison_plots(self, analytical_file="results/analytic.dat", 
                                    numerical_file="solution_lax_device21.dat"):
        
        print("Generating analytical vs numerical comparison plots...")
        print(f"Analytical file: {analytical_file}")
        print(f"Numerical file: {numerical_file}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 50)
        
        try:
            analytical_data = self.load_analytical_data(analytical_file)
            numerical_data = self.load_numerical_data(numerical_file)
            
            plots_generated = []
            
            # Density comparison
            fig1 = self.plot_density_comparison(analytical_data, numerical_data)
            plots_generated.append('density_comparison')
            
            # Pressure comparison
            fig2 = self.plot_pressure_comparison(analytical_data, numerical_data)
            plots_generated.append('pressure_comparison')
            
            # Velocity comparison
            fig3 = self.plot_velocity_comparison(analytical_data, numerical_data)
            plots_generated.append('velocity_comparison')
            
            # Temperature comparison
            fig4 = self.plot_temperature_comparison(analytical_data, numerical_data)
            plots_generated.append('temperature_comparison')
            
            print("-" * 50)
            print(f"Generated {len(plots_generated)} comparison plots:")
            for plot in plots_generated:
                print(f"  - {plot}.png")
            print(f"All plots saved to: {self.output_dir}")
            
            return plots_generated
            
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()
            return []

def main():
    comparison = ShockTubeComparison("../results", "../plots")
    plots = comparison.generate_all_comparison_plots(
        analytical_file="../results/analytic.dat",
        numerical_file="solution_lax_device21.dat"
    )


if __name__ == "__main__":
    main()
