import csv

import imageio
from evogym.envs import *

from AuxiliaryClasses import Controller


def simulate_best_robot(robot_structure, scenario=None, steps=500, controller='alternating_gait'):
    connectivity = get_full_connectivity(robot_structure)
    # if not isinstance(connectivity, np.ndarray):
    #    connectivity = np.zeros(robot_structure.shape, dtype=int)
    env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
    env.reset()
    sim = env.sim
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    action_size = sim.get_dim_action_space('robot')  # Get correct action size
    t_reward = 0
    controller = Controller.Controller(controller_type=controller, action_size=action_size)
    for t in range(200):  # Simulate for 200 timesteps
        # Update actuation before stepping
        action = controller.run(t)

        ob, reward, terminated, truncated, info = env.step(action)
        t_reward += reward
        if terminated or truncated:
            env.reset()
            break
        viewer.render('screen')
    viewer.close()
    env.close()

    return t_reward


def create_gif(robot_structure, filename='best_robot.gif', duration=0.066, scenario=None, steps=500,
               controller_type='alternating_gait'):
    try:
        """Create a smooth GIF of the robot simulation at 30fps."""
        connectivity = get_full_connectivity(robot_structure)
        # if not isinstance(connectivity, np.ndarray):
        #    connectivity = np.zeros(robot_structure.shape, dtype=int)
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        env.reset()
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')
        frames = []
        action_size = sim.get_dim_action_space('robot')  # Get correct action size
        t_reward = 0
        controller = Controller.Controller(controller_type=controller_type, action_size=action_size)
        for t in range(steps):

            viewer.render('screen')
            action = controller.run(t)
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break
            frame = viewer.render('rgb_array')
            frames.append(frame)

        viewer.close()
        imageio.mimsave(filename, frames, duration=duration, optimize=True)
    except ValueError as e:
        print('Invalid, ' + str(e))


def convert_last_row_to_matrix(csv_file_path):
    # Read the last row of the CSV
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty")

    # Get headers and last row
    headers = rows[0]
    last_row = rows[-1]

    # Find the "Best Structure" column index
    try:
        col_index = headers.index("Best Structure")
    except ValueError:
        raise ValueError("'Best Structure' column not found")

    # Get the flattened matrix string
    flattened_str = last_row[col_index]
    if not flattened_str:
        raise ValueError("'Best Structure' value is empty")

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


def simulate(best_robot, scenario, save_path, controller='alternating_gait', steps=500):
    simulate_best_robot(best_robot, scenario=scenario, steps=steps, controller=controller)
    create_gif(best_robot, filename=save_path, scenario=scenario, steps=steps)


if __name__ == '__main__':
    robot = convert_last_row_to_matrix(
        '/Users/sjmendes/Documents/Universidade/Mestrado/1_ano/2_semestre/EC/Project/implementation/evolve_structure/ES/data/fixed_controller/alternating_gait/BridgeWalker-v0/2025_05_08_at_13_20_30.csv')
    simulate(robot, "BridgeWalker-v0", "/Users/sjmendes/Documents/Universidade/Mestrado/1_ano/2_semestre/EC/Project/implementation/evolve_structure/ES/data/fixed_controller/alternating_gait/BridgeWalker-v0/2025_05_08_at_13_20_30.gif")
