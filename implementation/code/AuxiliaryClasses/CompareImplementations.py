import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


class CompareImplementations:
    def __init__(self, implementation1_name: str, implementation2_name: str, scenario: str):
        self.scenario = scenario
        self.implementation_1_name = implementation1_name
        self.implementation_2_name = implementation2_name
        self.implementation_1 = {}
        self.implementation_2 = {}
        self.std_implementation_1 = {}
        self.std_implementation_2 = {}
        self._raw_dfs_1 = []
        self._raw_dfs_2 = []

    def load_and_process_folders(self, folder1, folder2):
        self._raw_dfs_1 = self._load_folder_data(folder1)
        self._raw_dfs_2 = self._load_folder_data(folder2)
        self.implementation_1, self.std_implementation_1 = self._compute_mean_std(self._raw_dfs_1)
        self.implementation_2, self.std_implementation_2 = self._compute_mean_std(self._raw_dfs_2)

    def _load_folder_data(self, folder_path):
        dataframes = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                dataframes.append(df)
        if not dataframes:
            raise ValueError(f"No CSV files found in {folder_path}")
        return dataframes

    import numpy as np
    import pandas as pd

    def _compute_mean_std(self, dfs):
        mean_dict = {}
        std_dict = {}

        # Find the minimum number of rows among all dataframes
        min_len = min(len(df) for df in dfs)

        # Clip all dataframes to the same number of rows
        clipped_dfs = [df.iloc[:min_len].copy() for df in dfs]

        for column in clipped_dfs[0].columns:
            try:
                # Collect column data across all dataframes
                data = [df[column].values for df in clipped_dfs]

                # Convert to float and mask invalid values (NaN, inf)
                clean_data = []
                for arr in data:
                    arr = arr.astype(float)
                    arr = np.where(np.isfinite(arr), arr, np.nan)  # Replace inf and -inf with NaN
                    clean_data.append(arr)

                stacked = np.stack(clean_data)

                # Use nanmean and nanstd to ignore NaNs
                mean_dict[column] = np.nanmean(stacked, axis=0).tolist()
                std_dict[column] = np.nanstd(stacked, axis=0).tolist()

            except Exception:
                continue  # Skip non-numeric or incompatible columns

        return mean_dict, std_dict

    def plot_comparison(self, column_name, save_path=None):
        if column_name not in self.implementation_1 or column_name not in self.implementation_2:
            raise ValueError(f"Column '{column_name}' not found in one of the implementations.")

        y1 = self.implementation_1[column_name]
        y2 = self.implementation_2[column_name]
        std1 = self.std_implementation_1[column_name]
        std2 = self.std_implementation_2[column_name]
        # Find the minimum length among all lists
        min_len = min(len(y1), len(y2), len(std1), len(std2))

        # Clip all lists to the minimum length
        y1 = y1[:min_len]
        y2 = y2[:min_len]
        std1 = std1[:min_len]
        std2 = std2[:min_len]
        x = list(range(min_len))

        plt.figure(figsize=(12, 6))
        plt.plot(x, y1, label=self.implementation_1_name, color='skyblue')
        plt.fill_between(x,
                         np.array(y1) - np.array(std1),
                         np.array(y1) + np.array(std1),
                         color='skyblue', alpha=0.3)

        plt.plot(x, y2, label=self.implementation_2_name, color='salmon')
        plt.fill_between(x,
                         np.array(y2) - np.array(std2),
                         np.array(y2) + np.array(std2),
                         color='salmon', alpha=0.3)

        plt.xlabel("Row Index")
        plt.ylabel(column_name)
        plt.title(f"Comparison of {column_name} on {self.scenario}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            filename = os.path.join(save_path, f"{self.scenario}/{column_name}"
                                               f"_{self.implementation_1_name}_vs_{self.implementation_2_name}.png")
            plt.savefig(filename)
        else:
            plt.show()

    def compare_statistically(self, column_name):
        if column_name not in self.implementation_1 or column_name not in self.implementation_2:
            raise ValueError(f"Column '{column_name}' not found.")

        data1 = np.array(self.implementation_1[column_name])
        data2 = np.array(self.implementation_2[column_name])

        # Clip both arrays to the minimum length
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]

        # Wilcoxon signed-rank test
        stat, p_value = wilcoxon(data1, data2)

        # Calculate mean of pairwise differences
        differences = data1 - data2
        mean_diff = np.mean(differences)

        print(f"\nStatistical Comparison of '{column_name}' Using Wilcoxon Signed-Rank Test:")
        print(f"  Wilcoxon statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Mean difference ({self.implementation_1_name} vs {self.implementation_2_name}): {mean_diff:.4f}")

        if p_value < 0.05:
            print("  Result: Statistically significant difference between implementations.")
            if mean_diff > 0:
                print(f"  Interpretation: {self.implementation_1_name} performs better on average.")
            elif mean_diff < 0:
                print(f"  Interpretation: {self.implementation_2_name} performs better on average.")
            else:
                print("  Interpretation: No practical difference detected.")
        else:
            print("  Result: No statistically significant difference.")


if __name__ == "__main__":
    comparison = CompareImplementations("Fixed-Size Controllers", "Dynamic-Size Controllers", "CaveCrawler-v0")
    comparison.load_and_process_folders(
        '/Users/sjmendes/Documents/Universidade/Mestrado/1_ano/2_semestre/EC/Project/implementation/evolve_both/static/data/runs/CaveCrawler-v0/250'
        ,
        '/Users/sjmendes/Documents/Universidade/Mestrado/1_ano/2_semestre/EC/Project/implementation/evolve_both/dynamic/data/runs/CaveCrawler-v0/250')
    for column in ['Average Fitness', 'Average Reward', 'Best Fitness', 'Best Reward']:
        comparison.plot_comparison(column, save_path="/Users/sjmendes/Documents/Universidade/Mestrado/1_ano/2_semestre/EC/Project/implementation/plots")
        comparison.compare_statistically(column)
