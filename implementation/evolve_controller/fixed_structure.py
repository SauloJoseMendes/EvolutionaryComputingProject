import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import evogym
import torch
from evogym.envs import *

from example.neural_controller import NeuralController, initialize_weights, get_weights, set_weights
import numpy as np

from implementation.evolve_structure.fixed_controller import filter_results

# Parameter

population_size = 20
generations = 50
input_size = 10
output_size = 5


STEPS = 500
robot = np.array([
    [1, 3, 1, 0, 0],
    [4, 1, 3, 2, 2],
    [3, 4, 4, 4, 4],
    [3, 0, 0, 3, 2],
    [0, 0, 0, 0, 2]
])

connections = evogym.get_full_connectivity(robot)


def create_population(size):
    population = []
    for _ in range(size):
        model = NeuralController(input_size, output_size)
        model.apply(initialize_weights)
        population.append(model)
    return population


def crossover(parent1_weights, parent2_weights):
    children_weights_ar = []
    for w1, w2 in zip(parent1_weights, parent2_weights):
        mask = np.random.rand(*w1.shape) < 0.5
        child_w = np.where(mask, w1, w2)
        children_weights_ar.append(child_w)
    return children_weights_ar


def mutate(weights, mutation_rate=0.1, mutation_strength=0.5):
    new_weights = []
    for w in weights:
        noise = np.random.randn(*w.shape) * mutation_strength
        mask = np.random.rand(*w.shape) < mutation_rate
        new_w = w + noise * mask
        new_weights.append(new_w)
    return new_weights


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
        viewer = evogym.EvoViewer(current_sim)
        viewer.track_objects(('robot',))

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
                viewer.render('screen')

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
        viewer.close()
        env.close()

        return final_fitness, total_reward

    except ValueError as error_fitness:
        print(f"Error during environment creation, discarding unusable controller: {error_fitness}")
        return np.nan

# === EA Helper Functions ===

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
    for param, nw in zip(child.parameters(), new_weights):
        param.data = torch.tensor(nw, dtype=torch.float32)
    return child


def elitism(population: List[NeuralController],
            fitnesses: List[float],
            num_elites: int = 2) -> List[NeuralController]:
    sorted_idx = np.argsort(fitnesses)[::-1][:num_elites]
    return [copy.deepcopy(population[i]) for i in sorted_idx]


# === Main Evolutionary Algorithm ===

def evolutionary_algorithm(seed: int,
                           scenario: str,
                           robot,
                           connections,
                           input_size: int,
                           output_size: int,
                           population_size: int,
                           generations: int,
                           STEPS: int = 1000) -> List[NeuralController]:
    """
    Runs an EA to evolve NeuralController weights for the given scenario.
    Returns the final population of controllers.
    """
    # Reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Initialize population
    population = []
    for _ in range(population_size):
        ctrl = NeuralController(input_size, output_size)
        ctrl.apply(initialize_weights)
        population.append(ctrl)

    for gen in range(generations):
        # Parallel fitness evaluation
        evaluator = partial(evaluate_controller_fitness,
                            scenario=scenario,
                            robot=robot,
                            connections=connections,
                            STEPS=STEPS,
                            view=False)
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(evaluator, population))
        fitnesses, _ = zip(*filter_results(results))

        # Elitism: keep top 2
        elites = elitism(population, list(fitnesses), num_elites=2)

        # Generate children until full size
        children: List[NeuralController] = []
        while len(children) < population_size - len(elites):
            # Tournament select two parents
            parents = tournament_selection(population,
                                           list(fitnesses),
                                           tournament_size=3,
                                           num_winners=2)
            # Crossover
            child = crossover(parents[0], parents[1], crossover_rate=0.5)
            # Mutation
            w = get_weights(child)
            w_mut = mutate(w)
            set_weights(child, w_mut)
            children.append(child)

        # New population
        population = elites + children
        print(f"Generation {gen+1}/{generations} best fitness: {max(fitnesses):.3f}")

    return population
