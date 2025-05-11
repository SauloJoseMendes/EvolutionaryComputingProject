import copy
from collections import deque
from functools import partial
import torch
import networkx as nx
import evogym
from evogym.envs import *
from concurrent.futures import ProcessPoolExecutor
from AuxiliaryClasses import Controller
import numpy as np
import random
import pandas as pd

from implementation.code.AuxiliaryClasses.MutationHandler import MutationHandler


BATCH_SIZE = 1
NUM_GENERATIONS = 250
POPULATION_SIZE = 100
MUTATION_RATE = 0.4
ELITISM_SIZE = 2

TESTING = False
STEPS = 500
SCENARIOS = ['Walker-v0', 'BridgeWalker-v0']
CONTROLLERS = ['alternating_gait', 'sinusoidal_wave', 'hopping_motion']

def enforce_max_nodes(graph: nx.DiGraph, max_nodes: int = 25) -> nx.DiGraph:
    """
       Ensures that the graph doesn't have to many nodes, aka the robot isn't too big.
       If it exceeds, removes nodes arbitrarily to meet the limit.

       Parameters:
           graph (nx.DiGraph): The directed graph to be modified.
           max_nodes (int): The maximum number of nodes allowed in the graph. Default is 25.

       Returns:
           nx.DiGraph: The graph after removing excess nodes, if any.
       """
    if graph.number_of_nodes() <= max_nodes:
        return graph
    nodes_to_remove = list(graph.nodes)[max_nodes:]
    graph.remove_nodes_from(nodes_to_remove)
    return graph

def generate_fully_connected_graph():
    """
       Generates a randomly connected directed graph with a variable number of nodes (between 3 and 15).

       Returns:
           nx.DiGraph: A directed graph (DiGraph) with a random number of nodes, randomly assigned node types,
           and edges that connect all nodes into a single component. The graph is subject to the maximum node limit.
       """
    rows, _ = (5, 5)
    num_nodes = random.randint(3, 15)
    graph = nx.DiGraph()
    for i in range(num_nodes):
        graph.add_node(i, type=random.choice([1, 2, 3, 4]))
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

def tournament_selection(population, fitnesses, tournament_size=3):
    """
      Selects parents using tournament selection.

      Parameters:
          population (list): A list of individuals (robot structures) in the population.
          fitnesses (list): A list of fitness values corresponding to each individual in the population.
          tournament_size (int): The number of candidates randomly chosen for each tournament. Default is 3.

      Returns:
          list: A list of selected parents (individuals with highest fitness from each tournament).
      """
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), size=tournament_size, replace=False)
        winner = candidates[np.argmax([fitnesses[c] for c in candidates])]
        selected.append(copy.deepcopy(population[winner]))
    return selected


def grafting_crossover(parent1: nx.DiGraph, parent2: nx.DiGraph) -> nx.DiGraph:
    """
     Performs directed graph grafting crossover inspired by Karl Sims.

     Parameters:
       parent1 (nx.DiGraph): The first parent's graph (acts as the base structure).
       parent2 (nx.DiGraph): The second parent's graph (provides a subgraph to graft).

     Returns:
       nx.DiGraph: A new graph resulting from grafting a branch from parent2 onto parent1.
     """
    if not isinstance(parent1, nx.DiGraph) or not isinstance(parent2, nx.DiGraph):
        raise ValueError("Both parents must be directed graphs (DiGraph).")
    child = copy.deepcopy(parent1)
    if len(child) == 0:
        return copy.deepcopy(parent2)
    attachment_point = random.choice(list(child.nodes()))
    graft_root = random.choice(list(parent2.nodes()))
    graft_subgraph = nx.descendants(parent2, graft_root)
    graft_subgraph.add(graft_root)
    graft_branch = parent2.subgraph(graft_subgraph).copy()
    max_child_node = max(child.nodes()) if child.nodes() else 0
    mapping = {node: node + max_child_node + 1 for node in graft_branch.nodes()}
    graft_branch = nx.relabel_nodes(graft_branch, mapping)
    child = nx.compose(child, graft_branch)
    new_graft_root = mapping[graft_root]
    child.add_edge(attachment_point, new_graft_root)
    return child

def mutate_structure(robot_graph: nx.DiGraph, mutation_rate=MUTATION_RATE):
    """
      Mutates a given robot by altering node parameters, connection parameters, and adding or removing connections.
      A new node may be added if the graph has a small number of nodes.

      Args:
          robot_graph (nx.DiGraph): The directed graph representing the robot's structure.
          mutation_rate (float, optional): The rate at which mutations should occur. Default is defined by MUTATION_RATE.

      Returns:
          nx.DiGraph: A new mutated graph that adheres to the mutation constraints.
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
    """
      Selects the top individuals from the population.

      Parameters:
      - population (list): A list of individuals in the population (e.g., chromosomes, solutions).
      - fitnesses (list or numpy array): A list or array of fitness values corresponding to the individuals.
      - elite_size (int, optional): The number of top individuals to select for elitism. Default is 2.

      Returns:
      - list: A list of the top 'elite_size' individuals based on their fitness values.
       """
    elite_indices = np.argsort(fitnesses)[-elite_size:]
    return [copy.deepcopy(population[i]) for i in elite_indices]

def remove_duplicates_and_replace(
        graphs: List[nx.DiGraph],
        max_nodes: int = 25) -> List[nx.DiGraph]:
    """
      This function takes a list of directed graphs and removes any duplicate graphs that existsin the list,
      based on the node type.
      Args:
          graphs (List[nx.DiGraph]): A list of directed graphs to be processed.
          max_nodes (int): The maximum number of nodes allowed in a generated graph. Default is 25.

      Returns:
          List[nx.DiGraph]: A list of unique directed graphs, where no two graphs are isomorphic
          (based on node type).
      """
    unique_graphs = []
    counter = 0
    for graph in graphs:
        is_duplicate = any(nx.is_isomorphic(graph, other,
                                            node_match=lambda x, y: x['type'] == y['type']) for other in unique_graphs)
        if not is_duplicate:
            unique_graphs.append(graph)
        else:
            counter += 1
            new_graph = generate_fully_connected_graph()
            new_graph = enforce_max_nodes(new_graph, max_nodes)
            while any(nx.is_isomorphic(new_graph, g,
                                       node_match=lambda x, y: x['type'] == y['type']) for g in unique_graphs):
                new_graph = generate_fully_connected_graph()
                new_graph = enforce_max_nodes(new_graph, max_nodes)
            unique_graphs.append(new_graph)
    return unique_graphs

def graph_to_matrix(graph, grid_size=(5, 5)):
    """
    Converts a graph into a matrix representation.

    Parameters:
    - graph (networkx.Graph): The graph to be converted. The nodes of the graph should have a 'type' attribute.
    - grid_size (tuple): The dimensions (rows, cols) of the grid to represent the graph. Default is (5, 5).

    Returns:
    - grid (numpy.ndarray): A matrix representation of the graph where each position corresponds to a node in the graph.
    - connectivity (numpy.ndarray): A connectivity matrix that represents how the grid positions are connected.
    """
    graph = MutationHandler.garbage_collect_nodes(graph)
    rows, cols = grid_size
    grid = np.zeros((rows, cols), dtype=int)
    visited = set()
    start_node = list(graph.nodes())[0]
    queue = deque([(start_node, (rows // 2, cols // 2))])
    visited.add(start_node)
    grid[rows // 2, cols // 2] = graph.nodes[start_node]['type']
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    while queue:
        node, (r, c) = queue.popleft()
        for neighbor in graph.successors(node):
            if neighbor in visited:
                continue
            for dr, dc in directions:
                rr, cc = r + dr, c + dc
                if (0 <= rr < rows and 0 <= cc < cols
                        and grid[rr, cc] == 0):
                    grid[rr, cc] = graph.nodes[neighbor]['type']
                    visited.add(neighbor)
                    queue.append((neighbor, (rr, cc)))
                    break
    connectivity = evogym.get_full_connectivity(grid)
    return grid, connectivity

def evaluate_structure_fitness(robot_graph: nx.DiGraph, controller_type, scenario, view=False):
    """
    Evaluates the fitness of a robot structure in a given scenario by simulating its performance in a gym environment.

    Parameters:
    robot_graph (nx.DiGraph): The graph representation of the robot's structure (nodes and edges).
    controller_type (str): The type of controller to be used for the robot (e.g., PID, Neural Network).
    scenario (str): The name of the scenario to be used in the simulation environment.
    view (bool): Whether or not to display the simulation during the evaluation. Default is False.

    Returns:
    tuple: A tuple containing:
        - final_fitness (float): The computed fitness score based on robot movement and speed.
        - t_reward (float): The total reward accumulated during the simulation.
    """

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
        controller = Controller.Controller(controller_type=controller_type, action_size=action_size)
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
        return np.nan, np.nan

def count_duplicate_digraphs(graph_list):
    """
    Count the number of duplicate directed graphs in a list of graphs.

    Parameters:
    graph_list (list): A list of directed graphs (e.g., from NetworkX library).

    Returns:
    int: The total number of duplicate graph pairs in the list.
    """
    duplicate_groups = {}
    for graph in graph_list:
        edge_data = tuple(sorted(
            (u, v, tuple(sorted(d.items())) if d else ())
            for u, v, d in graph.edges(data=True)
        ))
        node_data = tuple(sorted(
            (n, tuple(sorted(d.items())) if d else ())
            for n, d in graph.nodes(data=True)
        ))
        graph_hash = (edge_data, node_data)
        duplicate_groups[graph_hash] = duplicate_groups.get(graph_hash, 0) + 1
    total_duplicates = sum(count - 1 for count in duplicate_groups.values() if count > 1)
    return total_duplicates

def evolutionary_algorithm(controller, scenario, debug=True):
    """
    Executes the evolutionary algorithm for optimizing structures through genetic operations.

    Parameters:
    - controller: The type of controller used to evaluate the fitness of the structures.
    - scenario: The scenario in which the structures will be evaluated.
    - debug: A boolean flag to enable or disable debugging output. Defaults to True.

    Returns:
    - best_structures: The best structures found at each generation.
    - best_fitnesses: The fitness values of the best structures at each generation.
    - best_rewards: The reward values of the best structures at each generation.
    - avg_fitness: The average fitness values at each generation.
    - avg_rewards: The average reward values at each generation.
    """
    def filter_results(results: List):
        """
               Filters and separates the fitness and reward values from the evaluation results.

               Parameters:
               - results: A list of evaluation results, each either a tuple of (fitness, reward) or a non-tuple result.

               Returns:
               - fitness_list: A list of fitness values.
               - reward_list: A list of reward values.
               """
        fitness_list = []
        reward_list = []
        for res in results:
            if isinstance(res, tuple):
                fitness, reward = res
                fitness_list.append(fitness)
                reward_list.append(reward)
            else:
                fitness_list.append(0)
                reward_list.append(0)
        return fitness_list, reward_list

    np.random.seed(None)
    random.seed(None)
    torch.manual_seed(int(time.time()))
    population = [generate_fully_connected_graph() for _ in range(POPULATION_SIZE)]
    best_structures = np.empty((NUM_GENERATIONS, 5, 5))
    best_fitnesses = np.empty(NUM_GENERATIONS)
    best_rewards = np.empty(NUM_GENERATIONS)
    avg_fitness = np.full(NUM_GENERATIONS, np.nan)
    avg_rewards = np.full(NUM_GENERATIONS, np.nan)
    with (ProcessPoolExecutor(max_workers=os.cpu_count()) as executor):
        for generation in range(NUM_GENERATIONS):
            population = remove_duplicates_and_replace(population)
            evaluator = partial(evaluate_structure_fitness,
                                controller_type=controller,
                                scenario=scenario,
                                view=False)
            fitnesses, rewards = filter_results(list(executor.map(evaluator, population)))
            current_best_fitness_idx = np.argmax(fitnesses)
            current_best_reward_idx = np.argmax(rewards)
            best_fitnesses[generation] = fitnesses[current_best_fitness_idx]
            best_structures[generation] = graph_to_matrix(population[current_best_fitness_idx])[0]
            best_rewards[generation] = rewards[current_best_reward_idx]
            fitnesses = np.array(fitnesses)
            rewards = np.array(rewards)
            fitnesses = fitnesses[np.isfinite(fitnesses)]
            rewards = rewards[np.isfinite(rewards)]
            avg_fitness[generation] = np.nanmean(fitnesses)
            avg_rewards[generation] = np.nanmean(rewards)
            parents = tournament_selection(population, fitnesses)
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1 = grafting_crossover(parents[i], parents[i + 1])
                    child2 = grafting_crossover(parents[i + 1], parents[i])
                    offspring.extend([child1, child2])
                else:
                    offspring.append(parents[i].copy())
            offspring = [mutate_structure(child) for child in offspring]
            elites = elitism(population, fitnesses, ELITISM_SIZE)
            elites.extend(offspring)
            population.clear()
            del population
            population = elites[:POPULATION_SIZE]
            if debug:
                print(f"Gen {generation + 1}: Best Fitness = {best_fitnesses[generation]:.2f}, "
                      f"Its Reward = {best_rewards[generation]:.2f}")
    return best_structures, best_fitnesses, best_rewards, avg_fitness, avg_rewards

def save_to_csv(data_csv, controller, scenario):
    """
       Saves the results of an evolutionary run to CSV and .pt files.

       Parameters:
         data_csv (list of dicts or pandas DataFrame): The data to save as CSV.
         controller (str): The name or identifier for the controller to be included in the path.
         scenario (str): The scenario name or identifier for the path.
       """

    df = pd.DataFrame(data_csv)
    path = f"../../evolve_structure/GP/data/fixed_controller/{controller}/{scenario}/{NUM_GENERATIONS}/"
    os.makedirs(path, exist_ok=True)
    filename = path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".csv"
    df.to_csv(filename, index=False)

def run(batches, controller, scenario):
    """
      Runs the evolutionary algorithm.

      Parameters:
          scenario (str): Name of the Evogym scenario.
          controller (str): Type of movement that the robot will do.
          batches (int): Number of independent runs to execute.
      """
    for iteration in range(batches):
        best_structures, best_fitnesses, best_rewards, avg_fitness, avg_reward = evolutionary_algorithm(controller,
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
        save_to_csv(data, controller, scenario)
        print("============================")

if __name__ == "__main__":

    _SCENARIOS = ['BridgeWalker-v0', 'Walker-v0']
    for _scenario in _SCENARIOS:
        run(batches=5, controller='alternating_gait', scenario=_scenario)
