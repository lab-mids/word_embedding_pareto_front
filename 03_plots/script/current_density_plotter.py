import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse


class CurrentDensityPlotter:
    def __init__(self, directory, x_col='x', y_col='y', current_col='current_density', figs_per_row=3):
        """
        Initializes the CurrentDensityPlotter class.

        Parameters:
        - directory (str): Path to the directory containing CSV files.
        - x_col (str): Column name for the x-axis positions.
        - y_col (str): Column name for the y-axis positions.
        - current_col (str): Column name for current density values.
        - figs_per_row (int): Number of subplots per row.
        """
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.current_col = current_col
        self.figs_per_row = figs_per_row
        self.dataframes = []
        self.files = []

    def load_data(self):
        """Loads all CSV files from the directory into DataFrames."""
        self.files = glob.glob(os.path.join(self.directory, "*.csv"))
        if not self.files:
            raise ValueError("No CSV files found in the specified directory!")

        for file in self.files:
            try:
                df = pd.read_csv(file)
                if {self.x_col, self.y_col, self.current_col}.issubset(df.columns):
                    self.dataframes.append(df[[self.x_col, self.y_col, self.current_col]])
                else:
                    print(f"Warning: {file} does not contain required columns.")
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not self.dataframes:
            raise ValueError("No valid CSV files were loaded!")

    def plot_subplots(self, output_file=None):
        """Generates subplots for each file with equal axis spacing and universal color scaling."""
        num_files = len(self.dataframes)
        rows = -(-num_files // self.figs_per_row)  # Calculate number of rows

        # Determine the universal color range
        vmin = min(df[self.current_col].min() for df in self.dataframes)
        vmax = max(df[self.current_col].max() for df in self.dataframes)

        fig, axes = plt.subplots(rows, self.figs_per_row, figsize=(4 * self.figs_per_row, 4 * rows))
        axes = np.array(axes).flatten()  # Flatten axes array for easy indexing

        for i, df in enumerate(self.dataframes):
            ax = axes[i]
            ax.scatter(
                df[self.x_col],
                df[self.y_col],
                c=df[self.current_col],
                cmap="viridis",
                vmin=vmin, vmax=vmax,  # Universal color range
                alpha=0.8,
                edgecolor="none",
                s=20
            )
            ax.set_aspect('equal')  # Ensure equal x-y spacing
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.grid(True)

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot current density subplots for multiple CSV files.")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing CSV files.")
    parser.add_argument("--x_col", type=str, default="x", help="Column name for x-axis positions.")
    parser.add_argument("--y_col", type=str, default="y", help="Column name for y-axis positions.")
    parser.add_argument("--current_col", type=str, default="current_density", help="Column name for current density values.")
    parser.add_argument("--figs_per_row", type=int, default=3, help="Number of subplots per row.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the output plot (optional).")
    return parser.parse_args()


def main():
    args = parse_args()

    plotter = CurrentDensityPlotter(
        directory=args.directory,
        x_col=args.x_col,
        y_col=args.y_col,
        current_col=args.current_col,
        figs_per_row=args.figs_per_row
    )
    plotter.load_data()
    plotter.plot_subplots(output_file=args.output_file)


if __name__ == "__main__":
    main()