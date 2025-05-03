import copy
from collections import deque
from functools import partial
import torch
import networkx as nx
import evogym
from evogym.envs import *
from concurrent.futures import ProcessPoolExecutor
from Controller import Controller
import numpy as np
import random
import pandas as pd

from implementation.evolve_structure.MutationHandler import MutationHandler

# ===== GP PARAMETERS =====
BATCH_SIZE = 1
NUM_GENERATIONS = 50
POPULATION_SIZE = 50
MUTATION_RATE = 0.4
ELITISM_SIZE = 2

# ===== EVOGYM PARAMETERS =====
TESTING = False
STEPS = 500
SCENARIOS = ['Walker-v0', 'BridgeWalker-v0']
CONTROLLERS = ['alternating_gait', 'sinusoidal_wave', 'hopping_motion']
SEEDS = [42, 0, 123, 987, 314159, 271828, 2 ** 32 - 1]


# ===== STRUCTURE REPRESENTATION =====
def enforce_max_nodes(graph: nx.DiGraph, max_nodes: int = 25) -> nx.DiGraph:
    """Ensure the graph has at most `max_nodes` nodes.
    If it exceeds, remove nodes arbitrarily to meet the limit."""
    if graph.number_of_nodes() <= max_nodes:
        return graph

    # If too many nodes, remove extras
    nodes_to_remove = list(graph.nodes)[max_nodes:]
    graph.remove_nodes_from(nodes_to_remove)

    return graph


def generate_fully_connected_graph(grid_size=(5, 5)):
    rows, _ = grid_size
    num_nodes = random.randint(3, 15)  # Generate a small number of nodes
    graph = nx.DiGraph()
    # Add nodes
    for i in range(num_nodes):
        graph.add_node(i, type=random.choice([1, 2, 3, 4]))

    # Connect the nodes to form a connected component (e.g., a chain or a small tree)
    nodes = list(graph.nodes())
    if nodes:
        start_node = random.choice(nodes)
        remaining_nodes = set(nodes)
        remaining_nodes.remove(start_node)
        connected_component = {start_node}

        while remaining_nodes:
            source_node = random.choice(list(connected_component))
            target_node = random.choice(list(remaining_nodes))
            graph.add_edge(source_node, target_node)
            connected_component.add(target_node)
            remaining_nodes.remove(target_node)

    return enforce_max_nodes(graph)


# ===== EVOLUTIONARY OPERATORS =====
def tournament_selection(population, fitnesses, tournament_size=3):
    """Selects parents using tournament selection."""
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), size=tournament_size, replace=False)
        winner = candidates[np.argmax([fitnesses[c] for c in candidates])]
        selected.append(copy.deepcopy(population[winner]))
    return selected


def grafting_crossover(parent1: nx.DiGraph, parent2: nx.DiGraph) -> nx.DiGraph:
    """
    Performs directed graph grafting crossover inspired by Karl Sims' technique.

    This function takes two parent graphs representing robot structures and
    grafts a randomly selected subgraph from one parent onto a randomly
    selected node in the other parent.

    Parameters:
      parent1 (nx.DiGraph): The first parent's graph (acts as the base structure).
      parent2 (nx.DiGraph): The second parent's graph (provides a subgraph to graft).

    Returns:
      nx.DiGraph: A new graph resulting from grafting a branch from parent2 onto parent1.
    """

    # Ensure both parents are directed graphs.
    if not isinstance(parent1, nx.DiGraph) or not isinstance(parent2, nx.DiGraph):
        raise ValueError("Both parents must be directed graphs (DiGraph).")

    # Create a copy of parent1 to serve as the child graph.
    child = copy.deepcopy(parent1)

    # If parent1 is empty, simply return a copy of parent2.
    if len(child) == 0:
        return copy.deepcopy(parent2)

    # Step 1: Select an attachment point in parent1 where the graft will be added.
    attachment_point = random.choice(list(child.nodes()))

    # Step 2: Select a random node from parent2 to serve as the root of the grafted subgraph.
    graft_root = random.choice(list(parent2.nodes()))

    # Step 3: Extract the entire subgraph (descendants) of the chosen root from parent2.
    graft_subgraph = nx.descendants(parent2, graft_root)  # Get all nodes reachable from graft_root.
    graft_subgraph.add(graft_root)  # Ensure the root itself is included.

    # Step 4: Create a copy of this subgraph from parent2.
    graft_branch = parent2.subgraph(graft_subgraph).copy()

    # Step 5: Relabel the nodes in the graft branch to avoid conflicts with existing node IDs in parent1.
    max_child_node = max(child.nodes()) if child.nodes() else 0  # Find the highest node ID in parent1.

    # Create a mapping of old node IDs to new unique IDs.
    mapping = {node: node + max_child_node + 1 for node in graft_branch.nodes()}

    # Apply the relabeling to the grafted subgraph.
    graft_branch = nx.relabel_nodes(graft_branch, mapping)

    # Step 6: Merge the grafted subgraph into the child graph.
    child = nx.compose(child, graft_branch)

    # Step 7: Create a new edge connecting the attachment point in parent1 to the graft root from parent2.
    new_graft_root = mapping[graft_root]  # Get the newly mapped ID of the graft root.
    child.add_edge(attachment_point, new_graft_root)  # Add a directed edge from the attachment point.

    return child


def mutate_structure(robot_graph: nx.DiGraph, mutation_rate=MUTATION_RATE):
    """
    Mutates a directed graph-based robot structure by randomly choosing one or two
    of the five possible mutation techniques (defined in MutationHandler) to apply.

    Parameters:
      robot_graph (nx.DiGraph): The directed graph representing the robot.
      mutation_rate (float): The base probability of mutation.

    Returns:
      nx.DiGraph: The mutated graph.
    """

    num_elements = len(robot_graph.nodes()) + len(robot_graph.edges())
    scaling = 1.0 / num_elements if num_elements > 0 else 1.0
    mutation_prob = mutation_rate * scaling

    handler = MutationHandler(mutation_prob)

    mutation_techniques = [
        handler.mutate_node_parameters,
        handler.mutate_connection_parameters,
        handler.add_remove_connections
    ]

    if len(robot_graph.nodes()) <= 25:
        mutation_techniques.append(handler.add_new_node)

    num_to_apply = random.choice([1, 2])
    techniques_to_apply = random.sample(mutation_techniques, num_to_apply)

    mutated_graph = robot_graph.copy()

    for technique in techniques_to_apply:
        mutated_graph = technique(mutated_graph)

    mutated_graph = MutationHandler.garbage_collect_nodes(mutated_graph)
    return enforce_max_nodes(mutated_graph)


def elitism(population, fitnesses, elite_size=2):
    """Preserves the top `elite_size` individuals."""
    elite_indices = np.argsort(fitnesses)[-elite_size:]
    return [copy.deepcopy(population[i]) for i in elite_indices]


def remove_duplicates_and_replace(
        graphs: List[nx.DiGraph],
        grid_size=(5, 5),
        max_nodes: int = 25) -> List[nx.DiGraph]:
    unique_graphs = []
    counter = 0
    for graph in graphs:
        is_duplicate = any(nx.is_isomorphic(graph, other,
                                            node_match=lambda x, y: x['type'] == y['type']) for other in unique_graphs)
        if not is_duplicate:
            unique_graphs.append(graph)
        else:
            counter += 1
            new_graph = generate_fully_connected_graph(grid_size)
            new_graph = enforce_max_nodes(new_graph, max_nodes)

            # Ensure new_graph is not a duplicate either
            while any(nx.is_isomorphic(new_graph, g,
                                       node_match=lambda x, y: x['type'] == y['type']) for g in unique_graphs):
                new_graph = generate_fully_connected_graph(grid_size)
                new_graph = enforce_max_nodes(new_graph, max_nodes)

            unique_graphs.append(new_graph)
    # print(f"Replaced {counter} duplicates")
    return unique_graphs


# ===== FITNESS EVALUATION =====

def graph_to_matrix(graph, grid_size=(5, 5)):
    """
    Converts a directed graph into a 5x5 matrix representation using BFS-based placement.
    Ensures that:
    - The graph remains a connected component in the grid.
    - Nodes are placed adjacent to their neighbors where possible.
    - The matrix does not exceed the 5x5 limit.

    Args:
        graph (networkx.DiGraph): The directed graph representing the structure.
        grid_size (tuple): The size of the grid (default is 5x5).

    Returns:
        tuple: A 5x5 numpy array representing the structure and its connectivity.
    """
    # Remove disconnected nodes
    graph = MutationHandler.garbage_collect_nodes(graph)

    rows, cols = grid_size
    grid = np.zeros((rows, cols), dtype=int)
    visited = set()

    # Start BFS from a central position to avoid running out of space
    start_node = list(graph.nodes())[0]  # Pick an arbitrary start node
    queue = deque([(start_node, (rows // 2, cols // 2))])  # Start at center of grid
    visited.add(start_node)
    grid[rows // 2, cols // 2] = graph.nodes[start_node]['type']  # Store block type

    # Directions for right, down, left, up (ensures adjacency)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    while queue:
        node, (r, c) = queue.popleft()

        # Iterate over node's successors (maintaining adjacency structure)
        for neighbor in graph.successors(node):
            if neighbor in visited:
                continue
            # look for any empty neighbor cell
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                if (0 <= rr < rows and 0 <= cc < cols
                        and grid[rr, cc] == 0):
                    grid[rr, cc] = graph.nodes[neighbor]['type']
                    visited.add(neighbor)
                    queue.append((neighbor, (rr, cc)))
                    break

    # Compute full connectivity for Evogym
    connectivity = evogym.get_full_connectivity(grid)
    return grid, connectivity


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


def evaluate_structure_fitness(robot_graph: nx.DiGraph, controller_type, scenario, view=False):
    structure, connectivity = graph_to_matrix(robot_graph)
    if (evogym.is_connected(structure) is False or
            connectivity.shape[1] == 0 or
            evogym.has_actuator(structure) is False):
        return np.nan
    try:
        env = gym.make(scenario, max_episode_steps=STEPS, body=structure, connections=connectivity)
        env.reset()
        current_sim = env.sim
        current_viewer = evogym.EvoViewer(current_sim)
        current_viewer.track_objects(('robot',))
        action_size = current_sim.get_dim_action_space('robot')
        t, t_reward, final_pos, average_speed = 0, 0, 0, 0
        controller = Controller(controller_type=controller_type, action_size=action_size)
        initial_pos = current_sim.object_pos_at_time(current_sim.get_time(), 'robot').mean()
        for t in range(STEPS):
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
        return np.nan


def count_duplicate_digraphs(graph_list):
    """
    Counts the number of duplicate directed graphs in a list of nx.DiGraph objects.

    Args:
        graph_list (list): A list of networkx.DiGraph objects

    Returns:
        tuple: (total_duplicates, duplicate_groups)
            total_duplicates: total number of duplicate graphs (excluding originals)
            duplicate_groups: a dictionary with {graph: count} for duplicates
    """
    duplicate_groups = {}

    for graph in graph_list:
        # Create a hashable representation of the graph
        # Convert all lists to tuples to ensure hashability
        edge_data = tuple(sorted(
            (u, v, tuple(sorted(d.items())) if d else ())
            for u, v, d in graph.edges(data=True)
        ))
        node_data = tuple(sorted(
            (n, tuple(sorted(d.items())) if d else ())
            for n, d in graph.nodes(data=True)
        ))
        graph_hash = (edge_data, node_data)

        # Count occurrences
        duplicate_groups[graph_hash] = duplicate_groups.get(graph_hash, 0) + 1

    # Calculate total duplicates (occurrences beyond the first)
    total_duplicates = sum(count - 1 for count in duplicate_groups.values() if count > 1)

    return total_duplicates


# ===== EVOLUTIONARY ALGORITHM =====
def evolutionary_algorithm(seed, controller, scenario):
    """Main EA loop with modular operators (parallelized fitness evaluation)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    population = [generate_fully_connected_graph() for _ in range(POPULATION_SIZE)]
    best_structures = np.empty((NUM_GENERATIONS, 5, 5))
    best_fitnesses = np.empty(NUM_GENERATIONS)
    best_rewards = np.empty(NUM_GENERATIONS)

    avg_fitness = np.full(NUM_GENERATIONS, np.nan)
    avg_rewards = np.full(NUM_GENERATIONS, np.nan)
    with (ProcessPoolExecutor(max_workers=os.cpu_count()) as executor):
        for generation in range(NUM_GENERATIONS):
            population = remove_duplicates_and_replace(population)
            # Parallel fitness evaluation
            evaluator = partial(evaluate_structure_fitness,
                                controller_type=controller,
                                scenario=scenario,
                                view=False)

            fitnesses, rewards = filter_results(list(executor.map(evaluator, population)))

            # Track best individual
            current_best_fitness_idx = np.argmax(fitnesses)
            current_best_reward_idx = np.argmax(rewards)

            best_fitnesses[generation] = fitnesses[current_best_fitness_idx]
            best_structures[generation] = graph_to_matrix(population[current_best_fitness_idx])[0]
            best_rewards[generation] = rewards[current_best_reward_idx]

            # print(graph_to_matrix(population[current_best_fitness_idx]))

            # Track average
            avg_fitness[generation] = np.nanmean(fitnesses)
            avg_rewards[generation] = np.nanmean(rewards)

            # Selection
            parents = tournament_selection(population, fitnesses)

            # Crossover
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1 = grafting_crossover(parents[i], parents[i + 1])
                    child2 = grafting_crossover(parents[i + 1], parents[i])
                    offspring.extend([child1, child2])
                else:
                    offspring.append(parents[i].copy())

            # Mutation
            offspring = [mutate_structure(child) for child in offspring]

            # Elitism
            elites = elitism(population, fitnesses, ELITISM_SIZE)
            elites.extend(offspring)

            # Clear and replace last population
            population.clear()
            del population
            population = elites[:POPULATION_SIZE]

            # print(f"Gen {generation + 1}: Best Fitness = {best_fitnesses[generation]:.2f}, "
            #       f"Its Reward = {best_rewards[generation]:.2f}")

    return best_structures, best_fitnesses, best_rewards, avg_fitness, avg_rewards


def save_to_csv(data_csv, seed, controller, scenario, testing):
    # Create a DataFrame
    df = pd.DataFrame(data_csv)
    if testing:
        path = f"./testing/fixed_controller/{seed}/{controller}/{scenario}/"
    else:
        path = f"./data/fixed_controller/{seed}/{controller}/{scenario}/"
    # Create all intermediate directories if they don't exist
    os.makedirs(path, exist_ok=True)
    filename = path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".csv"
    # Save to CSV
    df.to_csv(filename, index=False)


def run(seed, controller, scenario, testing=False):
    for iteration in range(BATCH_SIZE):
        best_structures, best_fitnesses, best_rewards, avg_fitness, avg_reward = evolutionary_algorithm(seed,
                                                                                                        controller,
                                                                                                        scenario)
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


def test_seeds():
    for seed in [271828, 2 ** 32 - 1]:
        for scenario in SCENARIOS:
            for controller in CONTROLLERS:
                run(seed=seed, controller=controller, scenario=scenario, testing=True)


# ===== RUN AND VISUALIZE =====
if __name__ == "__main__":
    test_seeds()
