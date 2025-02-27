
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

class ParetoIllustrationPlot:
    def __init__(self, output_dir):
        """
        Initialize the class with the output directory for saving the plots.

        Args:
            output_dir (str): Directory where the output plot will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_pareto_front(self, points, direction="lower_left"):
        """
        Calculate the Pareto front for the given points in the specified direction.

        Args:
            points (ndarray): 2D array of points (x, y).
            direction (str): "lower_left" for HER & ORR, "upper_right" for OER.

        Returns:
            ndarray: Pareto front points.
        """
        pareto_points = []
        if direction == "lower_left":
            # Sort by x ascending, then y ascending
            points = points[np.argsort(points[:, 0])]
            for point in points:
                if not pareto_points or point[1] < pareto_points[-1][1]:
                    pareto_points.append(point)
        elif direction == "upper_right":
            # Sort by x descending, then y descending
            points = points[np.argsort(-points[:, 0])]
            for point in points:
                if not pareto_points or point[1] > pareto_points[-1][1]:
                    pareto_points.append(point)
        return np.array(pareto_points)

    def create_plot(self):
        """
        Create the Pareto front illustration plot with dots and Pareto lines.
        """
        # Simulate random points within the range (0.2, 0.8)
        np.random.seed(42)  # For reproducibility
        x = 0.2 + 0.6 * np.random.rand(100)
        y = 0.2 + 0.6 * np.random.rand(100)
        points = np.column_stack((x, y))

        # Calculate Pareto fronts
        pareto_lower_left = self.calculate_pareto_front(points, direction="lower_left")
        pareto_upper_right = self.calculate_pareto_front(points, direction="upper_right")

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all points
        ax.scatter(points[:, 0], points[:, 1], color="gray", alpha=0.5)

        # Plot Pareto fronts
        ax.plot(
            pareto_lower_left[:, 0], pareto_lower_left[:, 1],
            color="red", linewidth=2
        )
        ax.plot(
            pareto_upper_right[:, 0], pareto_upper_right[:, 1],
            color="blue", linewidth=2
        )

        # Annotate Pareto fronts
        ax.text(
            pareto_lower_left[-1, 0], pareto_lower_left[-1, 1] + 0.02,
            "HER & ORR",
            color="red", fontsize=12, fontweight="bold",
            ha="left", va="center"
        )
        ax.text(
            pareto_upper_right[0, 0], pareto_upper_right[0, 1],
            "OER",
            color="blue", fontsize=12, fontweight="bold",
            ha="right", va="center"
        )

        # Set axis labels
        ax.set_xlabel("Similarity to Conductivity", fontsize=14)
        ax.set_ylabel("Similarity to Dielectric", fontsize=14)

        # Adjust axis limits
        ax.set_xlim(0.2, 0.8)
        ax.set_ylim(0.2, 0.8)

        # Save the plot
        output_path = os.path.join(self.output_dir, "pareto_front_illustration.pdf")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Illustration saved to {output_path}")


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a Pareto front illustration plot.")
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        help="Directory where the output plot will be saved."
    )
    return parser.parse_args()


def main():
    """
    Main function to generate the Pareto front illustration plot.
    """
    args = parse_args()
    plotter = ParetoIllustrationPlot(output_dir=args.output_dir)
    plotter.create_plot()


if __name__ == "__main__":
    main()