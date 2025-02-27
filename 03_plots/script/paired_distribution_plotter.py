import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import matplotlib.ticker as mtick
import numpy as np


class AutoPairedDistributionPlotter:
    def __init__(self, original_dir, pareto_dir, save_dir=None):
        """
        Initialize the AutoPairedDistributionPlotter with the directories containing original and pareto data.

        Args:
            original_dir (str): Path to the directory containing original processed data.
            pareto_dir (str): Path to the directory containing pareto results.
            save_dir (str, optional): Directory to save the generated figures. Defaults to None.
        """
        self.original_dir = original_dir
        self.pareto_dir = pareto_dir
        self.save_dir = save_dir

    def _get_numeric_columns(self, original_data, pareto_data):
        """
        Get the numeric columns that are common to both datasets.

        Args:
            original_data (pd.DataFrame): DataFrame for the original data.
            pareto_data (pd.DataFrame): DataFrame for the pareto data.

        Returns:
            list: List of numeric column names present in both datasets.
        """
        original_numeric = original_data.select_dtypes(include='number').columns
        pareto_numeric = pareto_data.select_dtypes(include='number').columns
        return list(set(original_numeric).intersection(set(pareto_numeric)))

    def _find_paired_files(self):
        """
        Find paired files from the original and pareto directories, including nested subdirectories.

        Returns:
            list: List of tuples containing the original and corresponding pareto file paths.
        """
        paired_files = []

        for root, dirs, files in os.walk(self.original_dir):
            for file in files:
                if file.endswith(".csv"):
                    # Build the path to the original CSV
                    original_file_path = os.path.join(root, file)

                    # Derive the relative path from the original root to match the pareto structure
                    relative_path = os.path.relpath(original_file_path, self.original_dir)

                    # Replace ".csv" with "_pareto_front.csv" to find the corresponding Pareto file
                    pareto_file_path = os.path.join(self.pareto_dir, relative_path).replace(".csv", "_pareto_front.csv")

                    # Check if the corresponding Pareto file exists
                    if os.path.exists(pareto_file_path):
                        paired_files.append((original_file_path, pareto_file_path))

        return paired_files

    def _label_subplots(self, axes):
        """
        Add (a), (b), (c), ... labels to each subplot.

        Args:
            axes (list): List of axes (subplots).
        """
        labels = 'abcdefghijklmnopqrstuvwxyz'
        for idx, ax in enumerate(axes):
            ax.text(-0.1, 1.05, f'({labels[idx]})', transform=ax.transAxes,
                    size=12, weight='bold', va='top', ha='right')



    def plot_paired_distributions(self, selected_columns=None, subplots_per_row=2,
                                  text_scale=1.5):
        """
        Plot the distribution comparisons for each system, arranging each system's plots in a grid.

        Args:
            selected_columns (list, optional): List of column names to plot. If None, all common numeric columns are plotted.
            subplots_per_row (int, optional): Number of subplots per row. Defaults to 2.
            text_scale (float, optional): Scaling factor for text size. Defaults to 1.5.

        Returns:
            None
        """

        # Set the text scale globally for Matplotlib
        plt.rcParams.update({
            'font.size': 10 * text_scale,
            'axes.titlesize': 12 * text_scale,
            'axes.labelsize': 11 * text_scale,
            'xtick.labelsize': 10 * text_scale,
            'ytick.labelsize': 10 * text_scale,
            'legend.fontsize': 9 * text_scale
        })

        paired_files = self._find_paired_files()

        for original_file, pareto_file in paired_files:
            # Read the original and pareto CSV files
            original_data = pd.read_csv(original_file)
            pareto_data = pd.read_csv(pareto_file)

            # Automatically detect the common numeric columns
            numeric_columns = self._get_numeric_columns(original_data, pareto_data)

            # Use selected columns if provided, otherwise plot all numeric columns
            if selected_columns:
                columns_to_plot = [col for col in selected_columns if
                                   col in numeric_columns]
            else:
                columns_to_plot = numeric_columns

            if not columns_to_plot:
                continue  # Skip if there are no valid columns to plot

            # Determine the total number of rows based on how many subplots per row
            num_columns = len(columns_to_plot)
            nrows = (
                                num_columns + subplots_per_row - 1) // subplots_per_row  # Ceiling division

            # Create subplots dynamically based on the number of valid columns
            fig, axes = plt.subplots(nrows, subplots_per_row,
                                     figsize=(5 * subplots_per_row, 4 * nrows))
            axes = axes.flatten() if num_columns > 1 else [
                axes]  # Ensure axes is always a list

            for idx, column_name in enumerate(columns_to_plot):
                ax = axes[idx]

                # Get the data for the selected column
                original_values = original_data[column_name]
                pareto_values = pareto_data[column_name]

                # Plot histograms for original and pareto data with custom colors
                ax.hist(original_values, bins=20, alpha=0.5, label='Original',
                        color='blue')
                ax.hist(pareto_values, bins=20, alpha=0.5, label='Pareto Front',
                        color='red')

                # Set labels and title
                ax.set_xlabel(column_name.replace("_", " "))
                ax.set_ylabel('Count')

                # Determine optimal decimal places for x-axis labels
                x_ticks = ax.get_xticks()
                unique_rounded_1 = len(set(np.round(x_ticks, 1)))
                unique_rounded_2 = len(set(np.round(x_ticks, 2)))
                unique_rounded_3 = len(set(np.round(x_ticks, 3)))

                if unique_rounded_1 == len(x_ticks):
                    decimal_places = 1  # If rounding to 1 decimal place works, use it
                elif unique_rounded_2 == len(x_ticks):
                    decimal_places = 2  # If rounding to 2 works, use it
                else:
                    decimal_places = 3  # Otherwise, use 3 decimal places

                # Format x-axis labels
                ax.xaxis.set_major_formatter(
                    mtick.FormatStrFormatter(f'%.{decimal_places}f'))

                # Add legend
                ax.legend()

            # Label each subplot (a), (b), (c), etc., with larger text
            for idx, ax in enumerate(axes[:num_columns]):
                label = f"({chr(97 + idx)})"  # Generate labels (a), (b), (c), ...
                ax.text(
                    -0.1, 1.1, label,
                    transform=ax.transAxes,
                    fontsize=12 * text_scale,
                    fontweight='bold',
                    va='top',
                    ha='left'
                )

            # Remove any empty subplots if there are fewer columns than expected
            if len(columns_to_plot) < len(axes):
                for empty_ax in axes[len(columns_to_plot):]:
                    fig.delaxes(empty_ax)

            # Adjust layout
            plt.tight_layout()

            # Get system name from the file name
            system_name = os.path.basename(original_file).replace(
                "_material_system_with_similarity.csv", "")

            # Save figure if save_dir is provided
            if self.save_dir:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                fig.savefig(
                    os.path.join(self.save_dir, f'{system_name}_distribution.pdf'),
                    bbox_inches='tight'
                )

            # Close the figure to free up memory
            plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paired distribution plots.")
    parser.add_argument("--original_dir", type=str, required=True, help="Directory containing the original data.")
    parser.add_argument("--pareto_dir", type=str, required=True, help="Directory containing the pareto data.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the generated plots.")
    parser.add_argument("--selected_columns", nargs="*", help="List of columns to plot. Defaults to all common numeric columns.")
    parser.add_argument("--subplots_per_row", type=int, default=2, help="Number of subplots per row. Defaults to 2.")
    return parser.parse_args()


def main():
    args = parse_args()
    plotter = AutoPairedDistributionPlotter(args.original_dir, args.pareto_dir, args.save_dir)
    plotter.plot_paired_distributions(selected_columns=args.selected_columns, subplots_per_row=args.subplots_per_row)


if __name__ == "__main__":
    main()