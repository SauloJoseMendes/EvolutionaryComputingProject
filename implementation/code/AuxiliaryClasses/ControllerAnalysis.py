import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

class ControllerAnalysis:
    def __init__(self, base_path="../../evolve_structure/GP/testing/fixed_controller/", metrics=None):
        self.base_path = base_path
        self.metrics = metrics if metrics else ["Best Reward", "Best Fitness"]
        self.data = None
        self.summary_by_controller = None
        self.summary_by_seed = None

    def load_data(self):
        """
        Walks through base_path/{seed}/{controller_type}/{scenario}/*.csv,
        reads only the last row of each CSV, and stores it in self.data.
        """
        records = []

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
                            if df.empty:
                                continue

                            # Grab the last row
                            last = df.iloc[-1]

                            # Convert to numeric, coerce errors to NaN
                            fitness = pd.to_numeric(last["Best Fitness"], errors="coerce")
                            reward = pd.to_numeric(last["Best Reward"], errors="coerce")

                            # Skip if conversion failed
                            if pd.isna(fitness) or pd.isna(reward):
                                print(f"Warning: non-numeric in {file}, skipping")
                                continue

                            # Build a lightweight record
                            records.append({
                                "Seed": seed,
                                "Controller": controller_type,
                                "Scenario": scenario,
                                "Best Fitness": last["Best Fitness"],
                                "Best Reward": last["Best Reward"]
                            })

                        except Exception as e:
                            print(f"Error reading {file}: {e}")

        if not records:
            raise RuntimeError("No CSV files were loaded.")

            # Create a single DataFrame of just these four columns + metadata
        self.data = pd.DataFrame(records)

    def summarize_by_controller(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Only aggregate the two numeric columns
        metric_cols = ["Best Fitness", "Best Reward"]
        self.summary_by_controller = (
            self.data
                .groupby("Controller")[metric_cols]
                .agg(["mean", "std"])
                .reset_index()
        )

    def summarize_by_seed(self):
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        metric_cols = ["Best Fitness", "Best Reward"]
        self.summary_by_seed = (
            self.data
                .groupby("Seed")[metric_cols]
                .agg(["mean", "std"])
                .reset_index()
        )

    def compare_controllers_statistically(self):
        """
        Perform non-parametric statistical comparison between controllers
        using Kruskal-Wallis and pairwise Mann-Whitney U tests with Bonferroni correction.
        """
        if not hasattr(self, 'data') or self.data.empty:
            raise ValueError("No data loaded. Call load_data() first.")

        print("\nStatistical Comparison of Controllers:")

        for metric in ["Best Fitness", "Best Reward"]:  # Or any other metrics you have
            print(f"\n--- Metric: {metric} ---")

            # Prepare data: list of arrays (one array per controller with all observations)
            controllers = self.data['Controller'].unique()
            data = [self.data[self.data['Controller'] == c][metric].values for c in controllers]

            # 1. Kruskal-Wallis test (omnibus test)
            h_stat, p_value = kruskal(*data)  # Fixed import here
            print(f"Kruskal-Wallis H-test: H={h_stat:.3f}, p={p_value:.4f}")

            if p_value < 0.05:
                print("Significant differences found between controllers (p < 0.05)")

                # 2. Pairwise Mann-Whitney U tests with Bonferroni correction
                n_comparisons = len(controllers) * (len(controllers) - 1) // 2
                alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

                print("\nPairwise comparisons (Mann-Whitney U with Bonferroni correction):")
                for (c1, c2) in combinations(controllers, 2):  # Now properly imported
                    group1 = self.data[self.data['Controller'] == c1][metric]
                    group2 = self.data[self.data['Controller'] == c2][metric]

                    u_stat, p = mannwhitneyu(group1, group2, alternative='two-sided')
                    significant = "YES" if p < alpha else "NO"
                    print(f"{c1} vs {c2}: U={u_stat:.1f}, p={p:.4f} (adj. α={alpha:.4f}), significant={significant}")

                    # Calculate effect size (Cliff's delta)
                    delta = (2 * u_stat / (len(group1) * len(group2))) - 1
                    print(f"   Effect size (Cliff's delta): {delta:.3f}")
            else:
                print("No significant differences found between controllers (p ≥ 0.05)")

    def plot_metrics(self, save_path=None):
        if self.summary_by_controller is None:
            raise ValueError("Summary not computed. Call summarize_by_controller() first.")

        os.makedirs(save_path, exist_ok=True) if save_path else None
        fig, axes = plt.subplots(1, len(self.metrics), figsize=(7 * len(self.metrics), 5))

        if len(self.metrics) == 1:
            axes = [axes]  # Make iterable

        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            means = self.summary_by_controller[(metric, "mean")]
            stds = self.summary_by_controller[(metric, "std")]
            controllers = self.summary_by_controller["Controller"]

            bars = ax.bar(controllers, means, yerr=stds, capsize=5)
            for bar, mean, std in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.05 * means.max(),
                    f"{mean:.2f}±{std:.2f}",
                    ha="center", va="bottom", fontsize=9
                )
            ax.set_title(f"{metric} (mean ± std)")
            ax.set_ylabel(metric)
            ax.set_xlabel("Controller Type")
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            filename = os.path.join(save_path, "controller_metrics.png")
            plt.savefig(filename)
            print(f"Plot saved to {filename}")
        else:
            plt.show()

    def plot_by_seed(self, controller_type, save_path=None):
        """
        For a given controller_type:
        - Group data by Seed to get mean±std of Best Reward & Best Fitness.
        - Plot those as bar charts.
        - Print the overall mean±std across seeds.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 1) Filter
        df_ctr = self.data[self.data["Controller"] == controller_type]
        if df_ctr.empty:
            raise ValueError(f"No data found for controller '{controller_type}'")

        # 2) Group by Seed
        metric_cols = ["Best Reward", "Best Fitness"]
        by_seed = (
            df_ctr
            .groupby("Seed")[metric_cols]
            .agg(["mean", "std"])
            .sort_index()
        )

        # 3) Print overall stats across seeds
        overall = by_seed[("Best Reward", "mean")].agg(["mean", "std"])
        print(f"\nOverall Best Reward across seeds for '{controller_type}': "
              f"{overall['mean']:.3f} ± {overall['std']:.3f}")
        overall = by_seed[("Best Reward", "mean")].agg(["mean", "std"])
        print(f"Overall Best Fitness across seeds for '{controller_type}': "
              f"{overall['mean']:.3f} ± {overall['std']:.3f}\n")

        # 4) Plot
        os.makedirs(save_path, exist_ok=True) if save_path else None
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), tight_layout=True)

        seeds = by_seed.index.astype(str)
        for i, metric in enumerate(metric_cols):
            ax = axes[i]
            means = by_seed[(metric, "mean")]
            stds = by_seed[(metric, "std")]

            bars = ax.bar(seeds, means, yerr=stds, capsize=5)
            ax.set_title(f"{metric} for '{controller_type}'\n(μ ± σ by seed)")
            ax.set_xlabel("Seed")
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)

            # annotate each bar
            for bar, m, s in zip(bars, means, stds):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.05 * means.max(),
                    f"{m:.2f}±{s:.2f}",
                    ha="center", va="bottom", fontsize=8
                )

        # Save or show
        if save_path:
            filename = os.path.join(save_path, f"{controller_type}_by_seed.png")
            plt.savefig(filename)
            print(f"Saved per-seed plot to {filename}")
        else:
            plt.show()


if __name__ == "__main__":
    analyzer = ControllerAnalysis()
    analyzer.load_data()
    analyzer.summarize_by_controller()
    print(analyzer.summary_by_controller)
    # analyzer.plot_metrics(save_path="./plots")
    analyzer.summarize_by_seed()
    analyzer.plot_by_seed("alternating_gait", save_path="../../evolve_structure/plots")
    analyzer.compare_controllers_statistically()
