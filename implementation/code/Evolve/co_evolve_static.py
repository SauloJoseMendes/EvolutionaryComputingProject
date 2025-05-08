import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import evogym
import pandas as pd
import torch
from evogym.envs import *

import fixed_controller_GP as es
import fixed_structure_hybrid as ec
from AuxiliaryClasses.Genome import Genome

SCENARIOS = ['GapJumper-v0', 'CaveCrawler-v0']
SEEDS = [42, 0, 123, 987, 314159, 271828, 2 ** 32 - 1]

# EA Parameters
BATCH_SIZE = 1
NUM_GENERATIONS = 50
POPULATION_SIZE = 50
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.4
ELITISM_SIZE = 2

# Evogym Parameter
STEPS = 500


def save(data_csv, scenario):
    best_genomes = data_csv.pop("Best Genomes")
    # Create a DataFrame
    df = pd.DataFrame(data_csv)
    run_path = f"../../evolve_both/testing/runs/{scenario}/{NUM_GENERATIONS}"
    genomes_path = f"../../evolve_both/testing/genomes/{scenario}/{NUM_GENERATIONS}"
    # Create all intermediate directories if they don't exist
    os.makedirs(run_path, exist_ok=True)
    run_filename = run_path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".csv"
    # Save to CSV
    df.to_csv(run_filename, index=False)
    # Save weights
    os.makedirs(genomes_path, exist_ok=True)
    genomes_filename = genomes_path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".pt"
    Genome.save_genomes(best_genomes, genomes_filename)


def initialize_population(scenario) -> List[Genome]:
    """Initialize a population with randomly generated individuals."""

    genomes = []
    for _ in range(POPULATION_SIZE):
        genomes.append(Genome(scenario=scenario))
    return genomes


def evaluate(individual: Genome, scenario: str, view: bool = False) -> Tuple[float, float]:
    def check_compatibility(env) -> Tuple[bool, float]:
        # genome declares how many inputs/outputs it expects:
        x = individual.input_size
        y = individual.output_size

        # environment actually provides:
        w = env.observation_space.shape[0]
        z = env.action_space.shape[0]

        # if they match exactly, OK
        if x == w and y == z:
            return True, 0.0

        # otherwise return False plus the negative L₁‐distance penalty
        penalty = - (abs(x - w) + abs(y - z))
        return False, penalty

    """Evaluate the fitness of a single individual by decoding and testing its joint components."""
    try:
        controller = individual.get_controller()
        individual_structure = individual.get_structure()
        robot, connections = es.graph_to_matrix(individual_structure)
        # print("OLA")
        # 1. Create the environment
        env = gym.make(scenario,
                       max_episode_steps=STEPS,
                       body=robot,
                       connections=connections)

        # Check compatibility
        # print("OLA2")
        isCompatible, penalty = check_compatibility(env)
        # print("OLA3")
        if isCompatible is False:
            return penalty, 0
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
        return -np.inf, -np.inf


def select_parents(population, fitness_scores, num_winners=1):
    """Select individuals from the population based on their fitness to be parents."""
    winners: List[Genome] = []

    for _ in range(num_winners):
        indices = np.random.choice(POPULATION_SIZE, size=TOURNAMENT_SIZE, replace=False)
        best_idx = max(indices, key=lambda idx: fitness_scores[idx])
        winners.append(copy.deepcopy(population[best_idx]))
    return winners


def crossover(parent1: Genome, parent2: Genome) -> Genome:
    """Perform crossover between two parent individuals to produce offspring."""
    parent1_graph = parent1.get_structure()
    parent2_graph = parent2.get_structure()
    parent1_controller = parent1.get_controller()
    parent2_controller = parent2.get_controller()

    offspring = copy.deepcopy(parent1)

    offspring_graph = es.grafting_crossover(parent1_graph, parent2_graph)
    offspring_controller = ec.crossover(parent1_controller, parent2_controller)
    offspring.set_structure(offspring_graph)
    offspring.set_controller(offspring_controller)

    return offspring


def mutate(individual: Genome) -> Genome:
    """Apply mutation to an individual based on a given mutation rate."""
    individual_graph = individual.get_structure()
    individual_weights = individual.get_weights()

    resulting_graph = es.mutate_structure(individual_graph, mutation_rate=MUTATION_RATE)
    resulting_weights = ec.mutate(weights=individual_weights, mutation_strength=MUTATION_RATE)

    individual.set_structure(resulting_graph)
    individual.set_weights(resulting_weights)
    return individual


def elitism(population: List[Genome], fitness_scores: List[float]) -> List[Genome]:
    """Selects the best fitted individuals"""
    sorted_idx = np.argsort(fitness_scores)[::-1][:ELITISM_SIZE]
    return [copy.deepcopy(population[i]) for i in sorted_idx]


def evolve(scenario: str, debug=False):
    """
    Runs an EA to evolve NeuralController weights for the given scenario.
    Returns the final population of controllers.
    """

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

    # Data
    best_genomes = np.empty(NUM_GENERATIONS, dtype=object)
    best_fitness_scores = np.empty(NUM_GENERATIONS)
    best_rewards = np.empty(NUM_GENERATIONS)

    avg_fitness = np.full(NUM_GENERATIONS, np.nan)
    avg_rewards = np.full(NUM_GENERATIONS, np.nan)

    # Reproducibility
    np.random.seed(None)
    random.seed(None)
    torch.manual_seed(int(time.time()))

    # Initialize population
    population = initialize_population(scenario)

    with (ProcessPoolExecutor(max_workers=os.cpu_count()) as executor):
        for generation in range(NUM_GENERATIONS):
            # Parallel fitness evaluation
            evaluator = partial(evaluate,
                                scenario=scenario)

            fitness_scores, rewards = filter_results(list(executor.map(evaluator, population)))

            # Track best individual
            current_best_fitness_idx = np.argmax(fitness_scores)
            current_best_reward_idx = np.argmax(rewards)

            best_fitness_scores[generation] = fitness_scores[current_best_fitness_idx]
            best_genomes[generation] = population[current_best_fitness_idx]
            best_rewards[generation] = rewards[current_best_reward_idx]

            # print(graph_to_matrix(population[current_best_fitness_idx]))

            # Track average
            fitness_scores = fitness_scores[np.isfinite(fitness_scores)]
            rewards = rewards[np.isfinite(rewards)]
            avg_fitness[generation] = np.nanmean(fitness_scores)
            avg_rewards[generation] = np.nanmean(rewards)

            # Elitism: keep top 2
            elites = elitism(population, list(fitness_scores))

            # Generate children until full size
            offspring: List[Genome] = []
            while len(offspring) < POPULATION_SIZE - len(elites):
                # Tournament select two parents
                parents = select_parents(population=population,
                                         fitness_scores=list(fitness_scores),
                                         num_winners=2)
                # Crossover
                child = crossover(parents[0], parents[1])
                mutate(child)
                offspring.append(child)

            # New population
            population = elites + offspring
            if debug:
                print(f"Generation {generation + 1}/{NUM_GENERATIONS} best fitness: {max(fitness_scores):.3f}")

    return best_genomes, best_fitness_scores, best_rewards, avg_fitness, avg_rewards


def run(scenario, batches=1):
    for iteration in range(batches):
        best_genomes, best_fitnesses, best_rewards, avg_fitness, avg_reward = evolve(scenario=scenario)
        print(f"===== Iteration {iteration} =====")
        print(f"Best Fitness Achieved: {best_fitnesses[-1]}")
        print(f"Best Reward Achieved: {best_rewards[-1]}")
        data = {
            "Average Fitness": avg_fitness,
            "Average Reward": avg_reward,
            "Best Fitness": best_fitnesses,
            "Best Reward": best_rewards,
            "Best Genomes": best_genomes,
        }
        save(data, scenario)
        # Visualize the best structure
        # print("Visualizing the best robot...")
        # evaluate_structure_fitness(best_structures[-1], view=True)
        print("============================")


if __name__ == '__main__':
    for _scenario in ['GapJumper-v0']:
        run(batches=1, scenario=_scenario)
