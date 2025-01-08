import argparse
import os
import pandas as pd
from scipy import stats


class PairedSystemStatisticsCalculator:
    periodic_table = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
        'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
        'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38,
        'Y': 39, 'Zr': 40,
        'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
        'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60,
        'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68,
        'Tm': 69, 'Yb': 70,
        'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88,
        'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98,
        'Es': 99, 'Fm': 100,
        'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107,
        'Hs': 108, 'Mt': 109, 'Ds': 110,
        'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
        'Og': 118
    }

    def __init__(self, original_dir, pareto_dir, output_dir, column_prefix="Current_at_"):
        self.original_dir = original_dir
        self.pareto_dir = pareto_dir
        self.output_dir = output_dir
        self.column_prefix = column_prefix

    def _find_paired_files(self):
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

    def find_current_column(self, columns):
        for column in columns:
            if column.startswith(self.column_prefix) and column.endswith("mV"):
                return column
        return None

    def extract_potential(self, current_column):
        return current_column.split("_")[-1].replace("mV", "")

    def extract_element_columns(self, columns):
        return [col for col in columns if col in self.periodic_table]

    def calculate_stats(self, data, columns, round_digits=True):
        stats = {}
        for column in columns:
            stats[column] = {
                "Max": int(data[column].max()) if round_digits else data[column].max(),
                "Min": int(data[column].min()) if round_digits else data[column].min(),
                "Mean": round(data[column].mean(), 3),
                "Std Dev": round(data[column].std(), 3),
                "Entries": len(data[column])
            }
        return stats

    def analyze_files(self):
        element_results = []
        current_results = []

        paired_files = self._find_paired_files()

        for original_file_path, pareto_file_path in paired_files:
            original_data = pd.read_csv(original_file_path)
            pareto_data = pd.read_csv(pareto_file_path)

            current_column = self.find_current_column(original_data.columns)
            if not current_column:
                continue

            potential = self.extract_potential(current_column)

            original_elements = self.extract_element_columns(original_data.columns)
            pareto_elements = self.extract_element_columns(pareto_data.columns)

            system_name = os.path.basename(original_file_path).split("_material_system")[0]

            all_elements = set(original_elements).union(pareto_elements)
            for element in all_elements:
                original_stats = self.calculate_stats(original_data, [element]) if element in original_elements else {}
                pareto_stats = self.calculate_stats(pareto_data, [element]) if element in pareto_elements else {}

                stats = {
                    "System": system_name,
                    "Element": element,
                    "Max (Original)": original_stats.get(element, {}).get("Max"),
                    "Max (Pareto)": pareto_stats.get(element, {}).get("Max"),
                    "Min (Original)": original_stats.get(element, {}).get("Min"),
                    "Min (Pareto)": pareto_stats.get(element, {}).get("Min"),
                    "Std Dev (Original)": original_stats.get(element, {}).get("Std Dev"),
                    "Std Dev (Pareto)": pareto_stats.get(element, {}).get("Std Dev")
                }
                element_results.append(stats)

            original_current_stats = self.calculate_stats(original_data, [current_column])
            pareto_current_stats = self.calculate_stats(pareto_data, [current_column])

            current_stats = {
                "System": system_name,
                "Potential (mV)": potential,
                "Max Current Density (Original, mA/cm²)": f"{original_current_stats[current_column]['Max']:.3f}",
                "Max Current Density (Pareto, mA/cm²)": f"{pareto_current_stats[current_column]['Max']:.3f}",
                "Min Current Density (Original, mA/cm²)": f"{original_current_stats[current_column]['Min']:.3f}",
                "Min Current Density (Pareto, mA/cm²)": f"{pareto_current_stats[current_column]['Min']:.3f}",
                "Mean Current Density (Original, mA/cm²)": f"{original_current_stats[current_column]['Mean']:.3f}",
                "Mean Current Density (Pareto, mA/cm²)": f"{pareto_current_stats[current_column]['Mean']:.3f}",
                "Std Dev Current Density (Original)": f"{original_current_stats[current_column]['Std Dev']:.3f}",
                "Std Dev Current Density (Pareto)": f"{pareto_current_stats[current_column]['Std Dev']:.3f}",
                "Entries (Original)": original_current_stats[current_column]["Entries"],
                "Entries (Pareto)": pareto_current_stats[current_column]["Entries"],
            }
            current_results.append(current_stats)

        element_df = pd.DataFrame(element_results)
        current_df = pd.DataFrame(current_results)

        self.save_results(element_df, current_df)

    def save_results(self, element_df, current_df):
        os.makedirs(self.output_dir, exist_ok=True)
        element_df.to_csv(os.path.join(self.output_dir, "element_statistics.csv"), index=False)
        current_df.to_csv(os.path.join(self.output_dir, "current_density_statistics.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="Calculate statistics for paired material systems.")
    parser.add_argument("--original_dir", required=True, help="Directory containing original material system files.")
    parser.add_argument("--pareto_dir", required=True, help="Directory containing Pareto front files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output files.")
    parser.add_argument("--column_prefix", default="Current_at_", help="Prefix for the current density column.")
    args = parser.parse_args()

    calculator = PairedSystemStatisticsCalculator(
        original_dir=args.original_dir,
        pareto_dir=args.pareto_dir,
        output_dir=args.output_dir,
        column_prefix=args.column_prefix
    )
    calculator.analyze_files()


if __name__ == "__main__":
    main()