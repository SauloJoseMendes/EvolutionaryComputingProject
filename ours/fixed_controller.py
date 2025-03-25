import networkx as nx
import numpy as np
import random
import gymnasium as gym
from evogym import EvoViewer, get_full_connectivity
from students import fixed_controllers as fc
from students.random_structure import CONTROLLER

# ===== PARAMETERS =====
NUM_GENERATIONS = 50
POPULATION_SIZE = 20
STEPS = 500
SCENARIO = 'Walker-v0'  # or 'BridgeWalker-v0'
CONTROLLER_TYPE = 'alternating_gait'  # Options: 'alternating_gait', 'sinusoidal_wave', 'hopping_motion'
MUTATION_RATE = 0.1
ELITISM_SIZE = 2  # Number of top individuals to preserve
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ===== STRUCTURE REPRESENTATION =====
def generate_random_structure():
    """Creates a random directed graph-based robot structure."""
    G = nx.DiGraph()  # Ensuring it's directed

    # Create nodes (voxels)
    for i in range(25):
        G.add_node(i, type=random.choice([0, 1, 2, 3, 4]))

    # Randomly add directed edges (parent → child)
    for i in range(25):
        for j in range(i + 1, 25):
            if random.random() < 0.2:
                G.add_edge(i, j)  # Directed edge

    return G


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


def mutate_structure(robot_graph: nx.DiGraph):
    """
    Mutates a directed graph-based robot structure following Karl Sims' mutation approach.

    The mutation is performed in several steps:
      1. Mutate internal parameters of each node.
      2. Add a new random node.
      3. Mutate connection (edge) parameters and possibly reassign targets.
      4. Add new random connections and remove some existing ones.
      5. Garbage collect any unconnected nodes.

    A temporary scaling of the mutation probability by the inverse graph size is used
    to ensure at least one mutation occurs per graph on average.

    Parameters:
      robot_graph (nx.DiGraph): The directed graph representing the robot.

    Returns:
      nx.DiGraph: The mutated graph.
    """

    # To ensure that even small graphs get mutated, scale mutation probability inversely by graph size.
    num_elements = len(robot_graph.nodes()) + len(robot_graph.edges())
    if num_elements == 0:
        scaling = 1.0
    else:
        scaling = 1.0 / num_elements

    # Effective mutation probability:
    mutation_prob = MUTATION_RATE * scaling

    # Step 1: Mutate internal parameters of each node.
    # Here, we assume each node has a dictionary of parameters,
    # e.g. 'type' (could be boolean or discrete), 'weight' (scalar), etc.
    for node, data in robot_graph.nodes(data=True):
        # Example: Mutate 'type' if present (discrete parameter)
        if "type" in data and random.random() < mutation_prob:
            # Flip among available types (here, 0 means empty; we might want to avoid that)
            data["type"] = random.choice([1, 2, 3, 4])

        # Example: Mutate a 'weight' parameter (scalar parameter)
        if "weight" in data and random.random() < mutation_prob:
            # Add small Gaussian noise
            adjustment = np.random.normal(0, 0.1)
            data["weight"] += adjustment
            # Optionally, also allow negation with some probability:
            if random.random() < 0.05:
                data["weight"] = -data["weight"]
            # Clamp to legal bounds (example bounds: 0.0 to 10.0)
            data["weight"] = max(0.0, min(data["weight"], 10.0))

    # Step 2: Add a new random node to the graph.
    if random.random() < mutation_prob:
        new_node_id = max(robot_graph.nodes()) + 1 if robot_graph.nodes() else 0
        # New node with random parameters; initially, it's disconnected.
        robot_graph.add_node(new_node_id, type=random.choice([1, 2, 3, 4]), weight=random.uniform(0.5, 5.0))

    # Step 3: Mutate connection parameters.
    # For each edge, we might adjust its weight or reassign its target.
    for (src, tgt, data) in list(robot_graph.edges(data=True)):
        if random.random() < mutation_prob:
            # For example, mutate an edge weight if present.
            if "weight" in data:
                adjustment = np.random.normal(0, 0.05)
                data["weight"] += adjustment
                data["weight"] = max(0.1, data["weight"])  # Ensure weight doesn't drop too low.
            # With some frequency, change the target of the connection:
            if random.random() < 0.5 * mutation_prob:
                possible_targets = list(robot_graph.nodes())
                new_tgt = random.choice(possible_targets)
                if new_tgt != src:
                    # Remove old edge and add new edge with the same data.
                    robot_graph.remove_edge(src, tgt)
                    robot_graph.add_edge(src, new_tgt, **data)

    # Step 4: Add new random connections and remove existing ones.
    for node in list(robot_graph.nodes()):
        # Chance to add a new outgoing edge from this node.
        if random.random() < mutation_prob:
            target = random.choice(list(robot_graph.nodes()))
            if node != target and not robot_graph.has_edge(node, target):
                # Optionally, assign a random weight to the edge.
                robot_graph.add_edge(node, target, weight=random.uniform(0.1, 1.0))

        # Chance to remove an existing outgoing edge.
        if random.random() < mutation_prob and robot_graph.out_degree > 0:
            edge_to_remove = random.choice(list(robot_graph.out_edges(node)))
            robot_graph.remove_edge(*edge_to_remove)

    # Step 5: Garbage collection: remove any nodes that are not connected
    # to the main body. Here we assume node 0 is the root of the morphology.
    if 0 in robot_graph.nodes():
        # Find all nodes reachable from the root (including root itself)
        reachable_nodes = set(nx.descendants(robot_graph, 0))
        reachable_nodes.add(0)
        # Remove nodes that are not reachable.
        for node in list(robot_graph.nodes()):
            if node not in reachable_nodes:
                robot_graph.remove_node(node)

    return robot_graph


def elitism(population, fitnesses, elite_size):
    """Preserves the top `elite_size` individuals."""
    elite_indices = np.argsort(fitnesses)[-elite_size:]
    return [population[i] for i in elite_indices]


# ===== FITNESS EVALUATION =====

def convert_graph_to_env_params(robot_graph: nx.DiGraph):
    """
    Converts a directed graph representing the robot structure into the format
    expected by the simulation environment. This function extracts two items:
      - structure: A dictionary (or other format) that encodes node parameters.
      - connectivity: A list of edges or a connectivity matrix derived from the graph.

    For example purposes, this function simply creates:
      - 'structure': a dictionary mapping node IDs to their attributes.
      - 'connectivity': a list of (source, target) pairs.

    You may need to adapt this conversion to match your environment's API.
    """
    structure = {node: data for node, data in robot_graph.nodes(data=True)}
    connectivity = list(robot_graph.edges())
    return structure, connectivity


def evaluate_structure_fitness(robot_graph: nx.DiGraph, view=False, alpha=0.01):
    """
    Evaluates the fitness of a robot's structure (morphology) based on its performance
    in the simulation environment when using a fixed controller.

    This version assumes your environment expects a 'body' and 'connections' argument,
    which we derive from the directed graph representing the robot.

    Additionally, a complexity penalty (scaled by alpha) is subtracted, where the penalty
    is proportional to the number of nodes in the graph.

    Parameters:
      robot_graph (nx.DiGraph): The directed graph representing the robot morphology.
      view (bool): Whether to render the simulation.
      alpha (float): Penalty coefficient for complexity (number of nodes).

    Returns:
      float: The overall fitness value. Higher values indicate better structure performance.
             Returns 0.0 if the evaluation fails.
    """
    try:
        # Convert the directed graph into the environment-specific format.
        structure, connectivity = convert_graph_to_env_params(robot_graph)

        # Create the simulation environment using the structure and connectivity.
        env = gym.make(SCENARIO, max_episode_steps=STEPS, body=structure, connections=connectivity)
        env.reset()
        sim = env.sim

        # Initialize a viewer (optional).
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        t_reward = 0
        # Get the action space size from the simulation.
        action_size = sim.get_dim_action_space('robot')

        # Run the simulation for a fixed number of steps.
        for t in range(STEPS):
            # Use the fixed controller to generate actions.
            actuation = CONTROLLER(action_size, t)
            if view:
                viewer.render('screen')
            ob, reward, terminated, truncated, info = env.step(actuation)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()

        # Apply a penalty for complexity: more nodes yield a higher penalty.
        complexity_penalty = alpha * robot_graph.number_of_nodes()

        fitness = t_reward - complexity_penalty
        return fitness
    except (ValueError, IndexError) as e:
        return 0.0


# ===== EVOLUTIONARY ALGORITHM =====
def evolutionary_algorithm():
    """Main EA loop with modular operators."""
    population = [generate_random_structure() for _ in range(POPULATION_SIZE)]
    best_structure = None
    best_fitness = -np.inf

    for generation in range(NUM_GENERATIONS):
        # Evaluate fitness
        fitnesses = [evaluate_structure_fitness(ind) for ind in population]

        # Track best individual
        current_best_idx = np.argmax(fitnesses)
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_structure = population[current_best_idx]

        # Selection
        parents = tournament_selection(population, fitnesses)

        # Crossover
        offspring = []
        for i in range(0, len(parents), 2):
            child1 = grafting_crossover(parents[i], parents[i + 1])
            child2 = grafting_crossover(parents[i + 1], parents[i])
            offspring.extend([child1, child2])

        # Mutation
        offspring = [mutate_structure(child) for child in offspring]

        # Elitism: Preserve top individuals
        elites = elitism(population, fitnesses, ELITISM_SIZE)

        # Survivor Selection (Elites + Offspring)
        combined = elites + offspring
        combined_fitnesses = [evaluate_structure_fitness(ind) for ind in combined]
        top_indices = np.argsort(combined_fitnesses)[-POPULATION_SIZE:]
        population = [combined[i] for i in top_indices]

        print(f"Gen {generation + 1}: Best Fitness = {best_fitness:.2f}")

    return best_structure, best_fitness


# ===== RUN AND VISUALIZE =====
if __name__ == "__main__":
    best_structure, best_fitness = evolutionary_algorithm()
    print(f"Best Fitness Achieved: {best_fitness}")

    # Visualize the best structure
    print("Visualizing the best robot...")
    evaluate_structure_fitness(best_structure, render=True)
