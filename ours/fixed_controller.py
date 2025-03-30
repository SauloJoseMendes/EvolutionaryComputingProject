from collections import deque

import networkx as nx
import evogym
from evogym.envs import *

from Controller import Controller
import numpy as np
import random

from ours.MutationHandler import MutationHandler

# ===== PARAMETERS =====
NUM_GENERATIONS = 50
POPULATION_SIZE = 20
STEPS = 500
SCENARIO = 'DownStepper-v0'  # or 'BridgeWalker-v0'
CONTROLLER_TYPE = 'alternating_gait'  # Options: 'alternating_gait', 'sinusoidal_wave', 'hopping_motion'
MUTATION_RATE = 0.1
ELITISM_SIZE = 2  # Number of top individuals to preserve
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ===== STRUCTURE REPRESENTATION =====
# grafo em cadeia
def generate_fully_connected_graph(grid_size=(5, 5)):
    rows, _ = grid_size
    num_nodes = random.randint(3, 8)  # Generate a small number of nodes
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

    return graph


# ===== EVOLUTIONARY OPERATORS =====
def tournament_selection(population, fitnesses, tournament_size=3):
    """Selects parents using tournament selection."""
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), size=tournament_size, replace=False)
        winner = candidates[np.argmax([fitnesses[c] for c in candidates])]
        selected.append(population[winner])
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
    child = parent1.copy()

    # If parent1 is empty, simply return a copy of parent2.
    if len(child) == 0:
        return parent2.copy()

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
        handler.add_new_node,
        handler.mutate_connection_parameters,
        handler.add_remove_connections
    ]

    num_to_apply = random.choice([1, 2])
    techniques_to_apply = random.sample(mutation_techniques, num_to_apply)

    mutated_graph = robot_graph.copy()

    for technique in techniques_to_apply:
        mutated_graph = technique(mutated_graph)

    mutated_graph = MutationHandler.garbage_collect_nodes(mutated_graph)
    return mutated_graph


def elitism(population, fitnesses, elite_size):
    """Preserves the top `elite_size` individuals."""
    elite_indices = np.argsort(fitnesses)[-elite_size:]
    return [population[i] for i in elite_indices]


# ===== FITNESS EVALUATION =====

def convert_to_evogym_format(graph, grid_size=(5, 5)):
    # Remove disconnected nodes
    graph = MutationHandler.garbage_collect_nodes(graph)

    rows, cols = grid_size
    grid = np.zeros((rows, cols), dtype=int)
    visited = set()

    start_node = list(graph.nodes())[0]  # Pick an arbitrary starting node
    queue = deque([(start_node, (0, 0))])  # Start BFS from (0,0)

    while queue:
        node, (r, c) = queue.popleft()

        if node in visited or not (0 <= r < rows and 0 <= c < cols):
            continue

        grid[r, c] = node  # Assign node index to grid
        visited.add(node)

        # Process neighbors (ensuring correct adjacency)
        neighbors = list(graph.successors(node))  # Directed edges
        possible_positions = [(r, c + 1), (r + 1, c), (r, c - 1), (r - 1, c)]  # Right, Down, Left, Up

        for neighbor, pos in zip(neighbors, possible_positions):
            if neighbor not in visited:
                queue.append((neighbor, pos))  # Add next node with its grid position

    connectivity = evogym.get_full_connectivity(grid)
    return grid, connectivity


def evaluate_structure_fitness(robot_graph: nx.DiGraph, view=False):
    structure, connectivity = convert_to_evogym_format(robot_graph)
    if (evogym.is_connected(structure) is False or
            connectivity.shape[1] == 0 or
            evogym.has_actuator(structure) is False):
        return -np.inf
    try:
        env = gym.make('Walker-v0', max_episode_steps=STEPS, body=structure, connections=connectivity)
        action_size = env.action_space.shape[0]
        controller = Controller(controller_type=CONTROLLER_TYPE, action_size=action_size)

        current_sim = env.sim
        current_viewer = evogym.EvoViewer(current_sim)
        current_viewer.track_objects(('robot',))

        state = env.reset()[0]  # Get initial state
        t_reward = 0
        for t in range(STEPS):
            if view:
                viewer.render('screen')
            action = controller.run(t)
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break

        if view:
            current_viewer.close()
        env.close()
        return t_reward
    except ValueError as error_fitness:
        print(f"Error during environment creation, discarding unusable structure: {error_fitness}")
        return -np.inf


# ===== EVOLUTIONARY ALGORITHM =====
def evolutionary_algorithm():
    """Main EA loop with modular operators."""
    population = [generate_fully_connected_graph() for _ in range(POPULATION_SIZE)]
    current_best_structure = None
    current_best_fitness = -np.inf

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        fitnesses = [evaluate_structure_fitness(ind) for ind in population]

        # Track best individual
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > current_best_fitness:
            current_best_fitness = fitnesses[current_best_idx]
            current_best_structure = population[current_best_idx]

        # Selection
        parents = tournament_selection(population, fitnesses)

        # Crossover
        offspring = []
        for i in range(0, len(parents), 2):
            # Handle the case where there's an odd number of parents
            if i + 1 < len(parents):
                child1 = grafting_crossover(parents[i], parents[i + 1])
                child2 = grafting_crossover(parents[i + 1], parents[i])
                offspring.extend([child1, child2])
            else:
                offspring.append(parents[i].copy())  # If odd, just copy the last parent

        # Mutation
        offspring = [mutate_structure(child) for child in offspring]

        # Elitism: Preserve top individuals
        elites = elitism(population, fitnesses, ELITISM_SIZE)

        # Survivor Selection (Elites + Offspring)
        combined = elites + offspring
        # Ensure we don't exceed population size due to elitism
        combined = combined[:POPULATION_SIZE]
        combined_fitnesses = [evaluate_structure_fitness(ind) for ind in combined]

        # Take the top POPULATION_SIZE individuals based on fitness
        sorted_indices = np.argsort(combined_fitnesses)[::-1]  # Sort in descending order of fitness
        population = [combined[i] for i in sorted_indices[:POPULATION_SIZE]]

        print(f"Gen {generation + 1}: Best Fitness = {current_best_fitness:.2f}")

    return current_best_structure, current_best_fitness


# ===== RUN AND VISUALIZE =====
if __name__ == "__main__":
    best_structure, best_fitness = evolutionary_algorithm()
    print(f"Best Fitness Achieved: {best_fitness}")

    # Visualize the best structure
    print("Visualizing the best robot...")
    evaluate_structure_fitness(best_structure, view=True)
