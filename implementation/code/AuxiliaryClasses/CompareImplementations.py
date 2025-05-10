import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon


class CompareImplementations:
    def __init__(self, implementation1_name: str, implementation2_name: str):
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

    def _compute_mean_std(self, dfs):
        combined = pd.concat(dfs, axis=0, keys=range(len(dfs)))
        mean_dict = {}
        std_dict = {}

        for column in dfs[0].columns:
            stacked = np.stack([df[column].values for df in dfs])
            mean_dict[column] = stacked.mean(axis=0).tolist()
            std_dict[column] = stacked.std(axis=0).tolist()

        return mean_dict, std_dict

    def plot_comparison(self, column_name, save_path=None):
        if column_name not in self.implementation_1 or column_name not in self.implementation_2:
            raise ValueError(f"Column '{column_name}' not found in one of the implementations.")

        x = list(range(len(self.implementation_1[column_name])))
        y1 = self.implementation_1[column_name]
        y2 = self.implementation_2[column_name]
        std1 = self.std_implementation_1[column_name]
        std2 = self.std_implementation_2[column_name]

        plt.figure(figsize=(12, 6))
        plt.plot(x, y1, label='Implementation 1', color='skyblue')
        plt.fill_between(x,
                         np.array(y1) - np.array(std1),
                         np.array(y1) + np.array(std1),
                         color='skyblue', alpha=0.3)

        plt.plot(x, y2, label='Implementation 2', color='salmon')
        plt.fill_between(x,
                         np.array(y2) - np.array(std2),
                         np.array(y2) + np.array(std2),
                         color='salmon', alpha=0.3)

        plt.xlabel("Row Index")
        plt.ylabel(column_name)
        plt.title(f"Row-wise Comparison of {column_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            filename = os.path.join(save_path, f"{column_name} comparison.png")
            plt.savefig(filename)
            print(f"Saved per-seed plot to {filename}")
        else:
            plt.show()

    def compare_statistically(self, column_name):
        if column_name not in self.implementation_1 or column_name not in self.implementation_2:
            raise ValueError(f"Column '{column_name}' not found.")

        data1 = np.array(self.implementation_1[column_name])
        data2 = np.array(self.implementation_2[column_name])

        if len(data1) != len(data2):
            raise ValueError("Data lengths must match for Wilcoxon signed-rank test.")

        # Wilcoxon signed-rank test
        stat, p_value = wilcoxon(data1, data2)

        # Calculate mean of pairwise differences
        differences = data1 - data2
        mean_diff = np.mean(differences)

        print(f"\nStatistical Comparison of '{column_name}' Using Wilcoxon Signed-Rank Test:")
        print(f"  Wilcoxon statistic: {stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Mean difference (Implementation 1 - Implementation 2): {mean_diff:.4f}")

        if p_value < 0.05:
            print("  Result: Statistically significant difference between implementations.")
            if mean_diff > 0:
                print("  Interpretation: Implementation 1 performs better on average.")
            elif mean_diff < 0:
                print("  Interpretation: Implementation 2 performs better on average.")
            else:
                print("  Interpretation: No practical difference detected.")
        else:
            print("  Result: No statistically significant difference.")

if __name__ == "__main__":
    comparison = CompareImplementations()
    comparison.load_and_process_folders('/Users/sjmendes/Documents/Universidade/Mestrado/1_ano/2_semestre/EC/Project/implementation/evolve_controller/DE/runs/DownStepper-v0/250'
                                        , '/Users/sjmendes/Documents/Universidade/Mestrado/1_ano/2_semestre/EC/Project/implementation/evolve_controller/DE/runs/ObstacleTraverser-v0/250')
    comparison.plot_comparison('Best Fitness')
    comparison.compare_statistically('Best Fitness')