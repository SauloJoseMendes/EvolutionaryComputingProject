import csv

import numpy as np

from AuxiliaryClasses import Controller
from example import utils


def convert_last_row_to_matrix(csv_file_path):
    # Read the last row of the CSV
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        last_row = None
        for row in reader:
            last_row = row

    if not last_row:
        raise ValueError("CSV file is empty")

    # Get the flattened matrix string
    flattened_str = last_row.get("Best Structure", "")
    if not flattened_str:
        raise ValueError("'Best Structure' column not found")

    # Convert string to list of numbers
    try:
        flattened = [float(x) for x in flattened_str.split(",")]
    except ValueError:
        raise ValueError("Could not convert values to numbers")

    # Reshape to 5x5 matrix
    if len(flattened) != 25:
        raise ValueError(f"Expected 25 elements for 5x5 matrix, got {len(flattened)}")

    matrix_5x5 = np.array(flattened).reshape(5, 5)
    return matrix_5x5

def simulate(best_robot, scenario, controller='alternating_gait', steps=500):
    i = 0
    while i < 10:
        utils.simulate_best_robot(best_robot, scenario=scenario, steps=steps)
        i += 1
    controller = Controller.Controller(controller)
    utils.create_gif(best_robot, filename='random_search.gif', scenario=scenario, steps=steps, controller=controller)

if __name__ == '__main__':
    convert_last_row_to_matrix('../../evolve_structure/ES/')