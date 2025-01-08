import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import argparse


class ScatterPlotWithPareto:
    def __init__(self, original_dir, pareto_dir, output_dir, column_prefix="Current_at_", x_column="x", y_column="y"):
        """
        Initialize the scatter plot class with directories and column names.

        Args:
            original_dir (str): Path to the directory containing original files.
            pareto_dir (str): Path to the directory containing Pareto files.
            output_dir (str): Directory to save the output plots.
            column_prefix (str): Prefix to identify the current density column. Defaults to "Current_at_".
            x_column (str): Column name for x-coordinate. Defaults to "x".
            y_column (str): Column name for y-coordinate. Defaults to "y".
        """
        self.original_dir = original_dir
        self.pareto_dir = pareto_dir
        self.output_dir = output_dir
        self.column_prefix = column_prefix
        self.x_column = x_column
        self.y_column = y_column

    def _find_paired_files(self):
        """
        Find paired files from the original and Pareto directories, including subdirectories.

        Returns:
            list: List of tuples containing the original and corresponding Pareto file paths.
        """
        paired_files = []

        for root, _, files in os.walk(self.original_dir):
            for file in files:
                if file.endswith(".csv"):
                    original_file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(original_file_path, self.original_dir)
                    pareto_file_path = os.path.join(self.pareto_dir, relative_path).replace(".csv", "_pareto_front.csv")

                    if os.path.exists(pareto_file_path):
                        paired_files.append((original_file_path, pareto_file_path))

        return paired_files

    def _find_current_column(self, columns):
        """
        Find the current density column in the provided columns.

        Args:
            columns (list): List of column names.

        Returns:
            str: The current density column name or None if not found.
        """
        for column in columns:
            if column.startswith(self.column_prefix) and column.endswith("mV"):
                return column
        return None

    def _extract_system_name(self, filename):
        """
        Extract the material system name from the filename.

        Args:
            filename (str): The filename.

        Returns:
            str: The extracted system name.
        """
        return re.split(r"_material_system|_dataset", filename)[0]

    def plot_all(self):
        """
        Plot scatter plots for all paired files and save them to the output directory.
        """
        paired_files = self._find_paired_files()
        os.makedirs(self.output_dir, exist_ok=True)

        for original_file, pareto_file in paired_files:
            original_data = pd.read_csv(original_file)
            pareto_data = pd.read_csv(pareto_file)

            current_column = self._find_current_column(original_data.columns)
            if not current_column or current_column != self._find_current_column(pareto_data.columns):
                print(f"Skipping files due to unmatched current density column: {original_file}, {pareto_file}")
                continue

            system_name = self._extract_system_name(os.path.basename(original_file))
            output_path = os.path.join(self.output_dir, f"{system_name}_pareto_front_scatter_plot.pdf")

            # Generate the plot
            fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

            # Get the shared color range for the colorbar
            vmin = original_data[current_column].min()
            vmax = original_data[current_column].max()

            # First subplot: Original data
            axs[0].scatter(
                original_data[self.x_column],
                original_data[self.y_column],
                c=original_data[current_column],
                cmap="viridis",
                alpha=0.8,
                edgecolor="none",
                vmin=vmin,
                vmax=vmax
            )
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("y")
            axs[0].set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
            axs[0].text(-0.1, 1.05, "(a)", transform=axs[0].transAxes, size=14, weight="bold", va="top", ha="right")

            # Second subplot: Pareto data
            axs[1].scatter(
                original_data[self.x_column],
                original_data[self.y_column],
                c="lightgray",
                alpha=0.5,
                edgecolor="none"
            )
            axs[1].scatter(
                pareto_data[self.x_column],
                pareto_data[self.y_column],
                c=pareto_data[current_column],
                cmap="viridis",
                alpha=0.8,
                edgecolor="none",
                vmin=vmin,
                vmax=vmax
            )
            axs[1].set_xlabel("x")
            axs[1].set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
            axs[1].text(-0.1, 1.05, "(b)", transform=axs[1].transAxes, size=14, weight="bold", va="top", ha="right")

            # Add a single shared colorbar
            cbar = fig.colorbar(
                plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax)),
                ax=axs,
                orientation="vertical",
                fraction=0.03,
                pad=0.1
            )
            cbar.set_label(f"{current_column} (mA/cm²)")

            plt.subplots_adjust(right=0.85, wspace=0.2)
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate scatter plots with Pareto data.")
    parser.add_argument("--original_dir", type=str, required=True, help="Directory containing the original data.")
    parser.add_argument("--pareto_dir", type=str, required=True, help="Directory containing the Pareto front data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output plots.")
    parser.add_argument("--column_prefix", type=str, default="Current_at_", help="Prefix for the current density column.")
    parser.add_argument("--x_column", type=str, default="x", help="Column name for x-coordinate.")
    parser.add_argument("--y_column", type=str, default="y", help="Column name for y-coordinate.")
    return parser.parse_args()


def main():
    args = parse_args()
    plotter = ScatterPlotWithPareto(
        original_dir=args.original_dir,
        pareto_dir=args.pareto_dir,
        output_dir=args.output_dir,
        column_prefix=args.column_prefix,
        x_column=args.x_column,
        y_column=args.y_column
    )
    plotter.plot_all()


if __name__ == "__main__":
    main()