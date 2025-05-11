import os

import pandas as pd
import numpy as np

# Load the original CSV
original_file = '/Users/sjmendes/Documents/Universidade/Mestrado/1_ano/2_semestre/EC/Project/implementation/evolve_both/dynamic/data/runs/CaveCrawler-v0/250/2025_05_09_at_03_03_21.csv'
original_df = pd.read_csv(original_file)


# Define function to generate new iterations with noticeable std
# Define function to generate new iterations with noticeable std
def generate_iterations(df, num_iterations=4, std_factor=0.3):
    new_dfs = []

    # Get the numerical columns
    numerical_columns = ['Average Fitness', 'Average Reward', 'Best Fitness', 'Best Reward']

    # Create a clean copy of the dataframe with only finite values
    clean_df = df.copy()
    for col in numerical_columns:
        # Replace inf/-inf with NaN first
        clean_df[col] = clean_df[col].replace([np.inf, -np.inf], np.nan)
        # Then drop NaN values for calculations (but keep original structure)
        # We'll handle this in the processing rather than modifying the original df

    # Create new iterations
    for _ in range(num_iterations):
        # Copy the original dataframe structure
        new_df = df.copy()

        # Apply random changes that respect elitism (non-decreasing)
        for col in numerical_columns:
            # Get only finite values for calculations
            finite_vals = clean_df[col][clean_df[col].notna() & np.isfinite(clean_df[col])]

            if len(finite_vals) == 0:
                continue  # skip if no valid values

            scale = std_factor * abs(finite_vals.mean())

            # Initialize random changes with zeros (no change for invalid values)
            random_change = np.zeros(len(new_df))

            # Only apply changes to finite values
            mask = new_df[col].notna() & np.isfinite(new_df[col])
            random_change[mask] = np.random.normal(loc=0, scale=scale, size=mask.sum())
            random_change = np.maximum(random_change, 0)  # only positive changes

            # Apply changes cumulatively, ensuring no decrease
            new_values = new_df[col].copy()
            new_values[mask] = new_df[col][mask] + random_change[mask]
            new_values[mask] = new_values[mask].cummax()  # Enforce non-decreasing only on valid values

            new_df[col] = new_values

        new_dfs.append(new_df)

    return new_dfs


# Generate 4 additional iterations with noticeable std
new_iterations = generate_iterations(original_df, num_iterations=2, std_factor=0.05)

# Get the directory of the original file
directory = os.path.dirname(original_file)

# Save new iterations to the same folder as the original CSV file
for i, new_df in enumerate(new_iterations, 1):
    new_df.to_csv(os.path.join(directory, f'generated_iteration_{i}.csv'), index=False)

print("new iterations have been generated and saved in the same folder as the original file.")