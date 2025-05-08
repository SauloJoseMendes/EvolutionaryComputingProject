import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import evogym
import pandas as pd
import torch
from evogym.envs import *

from AuxiliaryClasses.NeuralController import NeuralController, initialize_weights, get_weights, set_weights
import numpy as np

SCENARIOS = ['DownStepper-v0', 'ObstacleTraverser-v0']
SEEDS = [42, 0, 123, 987, 314159, 271828, 2 ** 32 - 1]

# EA Parameters
BATCH_SIZE = 1
NUM_GENERATIONS = 250
POPULATION_SIZE = 100
MUTATION_RATE = 0.4
ELITISM_SIZE = 2

# Evogym Parameter
STEPS = 500
robot = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

connections = evogym.get_full_connectivity(robot)


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


# === EA Helper Functions ===


def create_population(scenario):
    env = gym.make(scenario, max_episode_steps=STEPS, body=robot, connections=connections)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    env.close()
    population = []
    for _ in range(POPULATION_SIZE):
        model = NeuralController(input_size, output_size)
        model.apply(initialize_weights)
        population.append(model)
    return population


def de_mutation(population: List[NeuralController], F=0.5):
    mutated = []
    for i in range(len(population)):
        a, b, c = population[np.random.choice(len(population), 3, replace=False)]
        donor = a + F * (b - c)  # DE-style mutation
        mutated.append(donor)
    return mutated


def evaluate_controller_fitness(controller, scenario, view=False):
    try:
        # 1. Create the environment
        env = gym.make(scenario,
                       max_episode_steps=STEPS,
                       body=robot,
                       connections=connections)

        # 2. Reset & get initial state
        state = env.reset()

        # if Gym returns (obs, info) tuple, unpack it
        if isinstance(state, tuple):
            state = state[0]

        current_sim = env.sim
        current_viewer = evogym.EvoViewer(current_sim)
        current_viewer.track_objects(('robot',))

        # 3. Record initial position
        initial_pos = current_sim.object_pos_at_time(
            current_sim.get_time(), 'robot'
        ).mean()

        # 4. Main loop
        t = 0
        total_reward = 0.0
        speed_accum = 0.0
        final_pos = initial_pos

        for t in range(STEPS):
            if view:
                current_viewer.render('screen')

            # ── Here's the key part ──
            # Convert the observation to a torch tensor,
            # feed it through your NN controller,
            # and pull out the action as a NumPy array.
            obs_tensor = torch.tensor(state,
                                      dtype=torch.float32).unsqueeze(0)
            action_tensor = controller(obs_tensor)
            action = action_tensor.detach().cpu().numpy().squeeze()
            # ─────────────────────────
            # Step the sim
            state, reward, terminated, truncated, info = env.step(action)
            # if it’s (obs, info), unpack
            if isinstance(state, tuple):
                state = state[0]

            total_reward += reward
            speed_accum += current_sim.object_vel_at_time(
                current_sim.get_time(), 'robot').mean()

            if terminated or truncated:
                final_pos = current_sim.object_pos_at_time(
                    current_sim.get_time(), 'robot').mean()
                break

        # 5. Compute fitness from distance & speed
        distance = final_pos - initial_pos
        avg_speed = speed_accum / (t + 1) if t >= 0 else 0.0
        final_fitness = (distance * 5.0) + (avg_speed * 2.0)

        # 6. Clean up
        current_viewer.close()
        env.close()

        return final_fitness, total_reward

    except ValueError as error_fitness:
        print(f"Error during environment creation, discarding unusable controller: {error_fitness}")
        return np.nan, np.nan


def tournament_selection(population: List[NeuralController],
                         fitnesses: List[float],
                         tournament_size: int = 3,
                         num_winners: int = 1) -> List[NeuralController]:
    winners: List[NeuralController] = []
    pop_size = len(population)
    for _ in range(num_winners):
        indices = np.random.choice(pop_size, size=tournament_size, replace=False)
        best_idx = max(indices, key=lambda idx: fitnesses[idx])
        winners.append(copy.deepcopy(population[best_idx]))
    return winners


def crossover(parent1: NeuralController,
              parent2: NeuralController,
              crossover_rate: float = 0.5) -> NeuralController:
    child = copy.deepcopy(parent1)
    w1 = [p.detach().numpy() for p in parent1.parameters()]
    w2 = [p.detach().numpy() for p in parent2.parameters()]
    new_weights = [np.where(np.random.rand(*a.shape) < crossover_rate, a, b)
                   for a, b in zip(w1, w2)]
    set_weights(child, new_weights)
    return child


def elitism(population: List[NeuralController],
            fitnesses: List[float],
            num_elites: int = 2) -> List[NeuralController]:
    sorted_idx = np.argsort(fitnesses)[::-1][:num_elites]
    return [copy.deepcopy(population[i]) for i in sorted_idx]


# === Main Evolutionary Algorithm ===

def evolve(scenario: str):
    """
    Runs an EA to evolve NeuralController weights for the given scenario.
    Returns the final population of controllers.
    """
    # Data
    best_weights = np.empty(NUM_GENERATIONS, dtype=object)
    best_fitnesses = np.empty(NUM_GENERATIONS)
    best_rewards = np.empty(NUM_GENERATIONS)

    avg_fitness = np.full(NUM_GENERATIONS, np.nan)
    avg_rewards = np.full(NUM_GENERATIONS, np.nan)

    # Reproducibility
    np.random.seed(None)
    random.seed(None)
    torch.manual_seed(int(time.time()))

    # Initialize population
    population = create_population(scenario)

    with (ProcessPoolExecutor(max_workers=os.cpu_count()) as executor):
        for generation in range(NUM_GENERATIONS):
            # Parallel fitness evaluation
            evaluator = partial(evaluate_controller_fitness,
                                scenario=scenario)

            fitnesses, rewards = filter_results(list(executor.map(evaluator, population)))

            # Track best individual
            current_best_fitness_idx = np.argmax(fitnesses)
            current_best_reward_idx = np.argmax(rewards)

            best_fitnesses[generation] = fitnesses[current_best_fitness_idx]
            best_weights[generation] = get_weights(population[current_best_fitness_idx])
            best_rewards[generation] = rewards[current_best_reward_idx]

            # print(graph_to_matrix(population[current_best_fitness_idx]))

            # Track average
            fitnesses = fitnesses[np.isfinite(fitnesses)]
            rewards = rewards[np.isfinite(rewards)]
            avg_fitness[generation] = np.nanmean(fitnesses)
            avg_rewards[generation] = np.nanmean(rewards)

            # Elitism: keep top 2
            elites = elitism(population, list(fitnesses), num_elites=2)

            # Generate children until full size
            children: List[NeuralController] = []
            for _ in range(len(population) - ELITISM_SIZE):
                # mutate here
                print("hi")

            # New population
            population = elites + children
            # print(f"Generation {generation + 1}/{NUM_GENERATIONS} best fitness: {max(fitnesses):.3f}")

    return best_weights, best_fitnesses, best_rewards, avg_fitness, avg_rewards


def save(data_csv, scenario):
    best_weights = data_csv.pop("Best Weights")
    # Create a DataFrame
    df = pd.DataFrame(data_csv)
    run_path = f"../../evolve_controllerdata/runs/{scenario}/{NUM_GENERATIONS}"
    weights_path = f"../../evolve_controller/GA+ES/data/weights/{scenario}/{NUM_GENERATIONS}"
    # Create all intermediate directories if they don't exist
    os.makedirs(run_path, exist_ok=True)
    run_filename = run_path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".csv"
    # Save to CSV
    df.to_csv(run_filename, index=False)
    # Save weights
    os.makedirs(weights_path, exist_ok=True)
    weights_filename = weights_path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".pt"
    torch.save(best_weights, weights_filename)


def run(scenario, testing=False, batches=1):
    for iteration in range(batches):
        best_weights, best_fitnesses, best_rewards, avg_fitness, avg_reward = evolve(scenario=scenario)
        print(f"===== Iteration {iteration} =====")
        print(f"Best Fitness Achieved: {best_fitnesses[-1]}")
        print(f"Best Reward Achieved: {best_rewards[-1]}")
        data = {
            "Average Fitness": avg_fitness,
            "Average Reward": avg_reward,
            "Best Fitness": best_fitnesses,
            "Best Reward": best_rewards,
            "Best Weights": best_weights,
        }
        save(data, scenario)
        # Visualize the best structure
        # print("Visualizing the best robot...")
        # evaluate_structure_fitness(best_structures[-1], view=True)
        print("============================")


if __name__ == '__main__':
    for _scenario in ['ObstacleTraverser-v0']:
        run(batches=3, scenario=_scenario)
