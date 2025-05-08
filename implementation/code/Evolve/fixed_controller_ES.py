import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import evogym
import numpy as np
import pandas as pd
import torch
from evogym.envs import *

from AuxiliaryClasses import Controller

# ---- PARAMETERS ----
NUM_GENERATIONS = 50  # Number of generations to evolve
MIN_GRID_SIZE = (5, 5)  # Minimum size of the robot grid
MAX_GRID_SIZE = (5, 5)  # Maximum size of the robot grid

# ---- VOXEL TYPES ----
VOXEL_TYPES = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)


def evaluate_fitness(robot_structure, scenario: str, steps: int = 500, view=False):
    connectivity = evogym.get_full_connectivity(robot_structure)
    if (evogym.is_connected(robot_structure) is False or
            connectivity.shape[1] == 0 or
            evogym.has_actuator(robot_structure) is False):
        return np.nan
    try:
        env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
        env.reset()
        current_sim = env.sim
        current_viewer = evogym.EvoViewer(current_sim)
        current_viewer.track_objects(('robot',))
        action_size = current_sim.get_dim_action_space('robot')
        t, t_reward, final_pos, average_speed = 0, 0, 0, 0
        controller = Controller.Controller(controller_type='alternating_gait', action_size=action_size)
        initial_pos = current_sim.object_pos_at_time(current_sim.get_time(), 'robot').mean()
        for t in range(steps):
            if view:
                current_viewer.render('screen')
            action = controller.run(t)
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            average_speed += current_sim.object_vel_at_time(current_sim.get_time(), 'robot').mean()
            if terminated or truncated:
                final_pos = current_sim.object_pos_at_time(current_sim.get_time(), 'robot').mean()
                env.reset()
                break
        distance = final_pos - initial_pos
        if t != 0:
            average_speed /= t
        final_fitness = (distance * 5) + average_speed * 2
        current_viewer.close()
        env.close()
        return final_fitness, t_reward
    except ValueError as error_fitness:
        print(f"Error during environment creation, discarding unusable structure: {error_fitness}")
        return -np.inf, -np.inf


def create_random_robot():
    """Generate a valid random robot structure."""

    grid_size = (random.randint(MIN_GRID_SIZE[0], MAX_GRID_SIZE[0]), random.randint(MIN_GRID_SIZE[1], MAX_GRID_SIZE[1]))
    random_robot, _ = sample_robot(grid_size)
    return random_robot


def mutate_robot(parent, mutation_rate=0.1):
    """Randomly flip a small fraction of voxels to a random type."""
    child = copy.deepcopy(parent)
    # For each voxel, with small probability, reassign its type:
    for i in range(child.shape[0]):
        for j in range(child.shape[1]):
            if random.random() < mutation_rate:
                child[i, j] = random.choice(VOXEL_TYPES)
    return child


def evolutionary_search(
        seed,
        controller,
        scenario,
        mu=1,
        lamb=10,
):
    def filter_results(results: List):
        fitness_list = []
        reward_list = []
        for res in results:
            if isinstance(res, tuple):
                fitness, reward = res
                fitness_list.append(fitness)
                reward_list.append(reward)
            else:
                # print("Invalid fitness or reward")
                fitness_list.append(0)
                reward_list.append(0)
        return fitness_list, reward_list

    # Initialize μ parents
    population = [create_random_robot() for _ in range(mu)]
    # Evaluate initial population in parallel
    evaluator = partial(evaluate_fitness,
                        scenario=scenario)

    # 2) Evaluate initial parents
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        init_results = list(executor.map(evaluator, population))
    fitnesses, rewards = zip(*init_results)

    # storage
    best_structures = [copy.deepcopy(np.array(population))]
    best_fitnesses = [max(fitnesses)]
    best_rewards = [rewards[np.argmax(fitnesses)]]
    avg_fitness = [float(np.mean(fitnesses))]
    avg_reward = [float(np.mean(rewards))]

    print(f"Init  | Best fit: {best_fitnesses[-1]:.3f}, avg fit: {avg_fitness[-1]:.3f}")

    # 3) Now the loop
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for gen in range(NUM_GENERATIONS):
            # a) generate offspring
            offspring = [mutate_robot(random.choice(population))
                         for _ in range(lamb)]

            # b) **evaluate the offspring** (not the old population)
            off_results = list(executor.map(evaluator, offspring))
            off_fits, off_rews = filter_results(off_results)

            # c) combine parent + offspring
            all_individuals = population + offspring
            all_fits = list(fitnesses) + off_fits
            all_rews = list(rewards) + off_rews

            # d) select top μ …
            idx_sorted = np.argsort(all_fits)[-mu:]
            population = [copy.deepcopy(all_individuals[i])
                          for i in idx_sorted]
            fitnesses = [all_fits[i] for i in idx_sorted]
            rewards = [all_rews[i] for i in idx_sorted]

            # e) record & print …
            best_idx = int(np.argmax(fitnesses))
            best_fit = fitnesses[best_idx]
            best_rew = rewards[best_idx]
            avg_fit = float(np.mean(fitnesses))
            avg_rew = float(np.mean(rewards))

            best_structures.append(np.array(copy.deepcopy(population[best_idx])))
            best_fitnesses.append(best_fit)
            best_rewards.append(best_rew)
            avg_fitness.append(avg_fit)
            avg_reward.append(avg_rew)

            print(f"Gen {gen:3d} | Best fit: {best_fit:.3f}, Best reward: {best_rew:.3f}")

    return best_structures, best_fitnesses, best_rewards, avg_fitness, avg_reward


def save_to_csv(data_csv, seed, controller, scenario, testing):
    # Create a DataFrame
    df = pd.DataFrame(data_csv)
    if testing:
        path = f"../../evolve_structure/ES/testing/fixed_controller/{seed}/{controller}/{scenario}/"
    else:
        path = f"../../evolve_structure/ES/data/fixed_controller/{controller}/{scenario}/"
    # Create all intermediate directories if they don't exist
    os.makedirs(path, exist_ok=True)
    filename = path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".csv"
    # Save to CSV
    df.to_csv(filename, index=False)


def simulate(best_robot, scenario, controller, steps):
    i = 0
    while i < 10:
        utils.simulate_best_robot(best_robot, scenario=scenario, steps=steps)
        i += 1
    controller = Controller.Controller(controller)
    utils.create_gif(best_robot, filename='random_search.gif', scenario=scenario, steps=steps, controller=controller)


def run(batches, seed, controller, scenario, testing=False):
    for iteration in range(batches):
        best_structures, best_fitnesses, best_rewards, avg_fitness, avg_reward = evolutionary_search(seed=seed,
                                                                                                     controller=controller,
                                                                                                     scenario=scenario)
        print(f"===== Iteration {iteration} =====")
        print(f"Best Fitness Achieved: {best_fitnesses[-1]}")
        print(f"Best Reward Achieved: {best_rewards[-1]}")
        data = {
            "Average Fitness": avg_fitness,
            "Average Reward": avg_reward,
            "Best Fitness": best_fitnesses,
            "Best Reward": best_rewards,
            "Best Structure": [",".join(map(str, mat.flatten())) for mat in best_structures],
        }
        save_to_csv(data, seed, controller, scenario, testing)
        # Visualize the best structure
        # print("Visualizing the best robot...")
        # evaluate_structure_fitness(best_structures[-1], view=True)
        print("============================")


if __name__ == "__main__":
    _SCENARIOS = ['BridgeWalker-v0', 'Walker-v0']
    for _scenario in _SCENARIOS:
        run(batches=1, seed=271828, controller='alternating_gait', scenario=_scenario)
