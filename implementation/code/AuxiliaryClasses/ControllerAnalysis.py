import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from itertools import combinations


class ControllerAnalysisTemporal:
    def __init__(self, base_path, metrics=None):
        self.base_path = base_path
        self.metrics = metrics if metrics else ["Best Reward", "Best Fitness"]
        self.temporal_data = {}  # controller -> list of DataFrames

    def load_temporal_data(self):
        """
        Loads all rows from every CSV file, organizing them by controller type.
        """
        for seed in os.listdir(self.base_path):
            seed_path = os.path.join(self.base_path, seed)
            if not os.path.isdir(seed_path):
                continue

            for controller_type in os.listdir(seed_path):
                ctrl_path = os.path.join(seed_path, controller_type)
                if not os.path.isdir(ctrl_path):
                    continue

                for scenario in os.listdir(ctrl_path):
                    sc_path = os.path.join(ctrl_path, scenario)
                    if not os.path.isdir(sc_path):
                        continue

                    for file in glob(os.path.join(sc_path, "*.csv")):
                        try:
                            df = pd.read_csv(file)

                            if df.empty or any(m not in df.columns for m in self.metrics):
                                continue

                            df = df[self.metrics].apply(pd.to_numeric, errors='coerce')
                            if df.isnull().values.any():
                                continue

                            self.temporal_data.setdefault(controller_type, []).append(df)

                        except Exception as e:
                            print(f"Error reading {file}: {e}")

    def summarize_temporal_metrics(self):
        """
        Calculate per-iteration mean and std for each controller.
        Returns a dictionary: { controller: { metric: { "mean": [...], "std": [...] } } }
        """
        summary = {}

        for controller, runs in self.temporal_data.items():
            combined = np.stack([df.values for df in runs])  # shape: (n_runs, n_steps, n_metrics)
            per_metric = {}

            for i, metric in enumerate(self.metrics):
                values = combined[:, :, i]  # shape: (n_runs, n_steps)
                per_metric[metric] = {
                    "mean": np.mean(values, axis=0),
                    "std": np.std(values, axis=0)
                }

            summary[controller] = per_metric

        return summary

    def plot_temporal_curves(self, save_path=None):
        """
        Plots mean ± std curves for each metric and controller in a single horizontal figure.
        """
        summary = self.summarize_temporal_metrics()
        num_metrics = len(self.metrics)
        fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 6), sharey=False)

        if num_metrics == 1:
            axes = [axes]  # Make iterable if only one metric

        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            for controller, stats in summary.items():
                mean = stats[metric]["mean"]
                std = stats[metric]["std"]
                x = np.arange(len(mean))
                ax.plot(x, mean, label=f"{controller}")
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.set_title(f"{metric} over Time")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True)

        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, "compare_controllers_metrics.png")
            plt.savefig(path)
            print(f"Saved: {path}")
        else:
            plt.show()

    def statistical_test(self):
        """
        Perform Wilcoxon signed-rank test on final values of each metric across controllers.
        Assumes data is paired (e.g., same seeds for each controller).
        """
        print("\nStatistical Comparison of Final Metric Values (Wilcoxon Signed-Rank Test):")
        for metric in self.metrics:
            print(f"\n--- Metric: {metric} ---")
            controller_values = {}
            for controller, dfs in self.temporal_data.items():
                controller_values[controller] = [df[metric].iloc[-1] for df in dfs]

            # Ensure all controllers have same number of runs for pairing
            lengths = [len(v) for v in controller_values.values()]
            if len(set(lengths)) != 1:
                print("⚠️ Unequal number of runs across controllers — cannot perform paired test.")
                continue

            for (c1, c2) in combinations(controller_values.keys(), 2):
                data1 = controller_values[c1]
                data2 = controller_values[c2]
                try:
                    stat, p_val = wilcoxon(data1, data2)
                    print(f"{c1} vs {c2} → W={stat:.2f}, p={p_val:.4f} → "
                          f"{'Significant' if p_val < 0.05 else 'Not Significant'}")
                except ValueError as e:
                    print(f"{c1} vs {c2} → Test failed: {e}")


if __name__ == "__main__":
    analyzer = ControllerAnalysisTemporal(base_path="../../evolve_structure/GP/testing/fixed_controller/")
    analyzer.load_temporal_data()
    analyzer.plot_temporal_curves(save_path="../../evolve_structure/plots")
    analyzer.statistical_test()
