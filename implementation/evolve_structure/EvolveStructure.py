import copy
from collections import deque
from functools import partial
from typing import TypedDict

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


class EvolveStructure:
    class Result(TypedDict):
        average_fitness: np.ndarray
        average_reward: np.ndarray
        best_fitness: np.ndarray
        best_reward: np.ndarray
        best_structure: List[str]

    def __init__(self,
                 population_size: int = 10,
                 mutation_rate: float = 0.4,
                 elitism_count: int = 2,
                 grid_size=(5, 5),
                 scenario: str = 'Walker-v0',
                 controller_type: str = 'alternating_gait',
                 steps: int = 500,
                 tournament_size: int = 2,
                 seed: int = 271828,
                 num_generations: int = 1,
                 testing: bool = False):
        """
        Initialize the evolutionary algorithm with parameters.

        Args:
            population_size: Number of individuals in the population
            mutation_rate: Probability of mutation for each gene
            elitism_count: Number of top individuals to carry over unchanged
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.grid_size = grid_size
        self.scenario = scenario
        self.controller_type = controller_type
        self.steps = steps
        self.tournament_size = tournament_size
        self.seed = seed
        self.num_generations = num_generations
        self.population = []
        self.initialize_population()
        self.result = None
        self.testing = testing

        self.initialize_population()

    # ===== AUXILIARY FUNCTIONS =====
    @staticmethod
    def enforce_max_nodes(individual: nx.DiGraph, max_nodes: int = 25) -> nx.DiGraph:
        """Ensure the graph has at most `max_nodes` nodes.
        If it exceeds, remove nodes arbitrarily to meet the limit."""
        if individual.number_of_nodes() <= max_nodes:
            return individual

        # If too many nodes, remove extras
        nodes_to_remove = list(individual.nodes)[max_nodes:]
        individual.remove_nodes_from(nodes_to_remove)

        return individual

    def graph_to_matrix(self, individual: nx.DiGraph) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts a directed graph into a 5x5 matrix representation using BFS-based placement.
        Ensures that:
        - The graph remains a connected component in the grid.
        - Nodes are placed adjacent to their neighbors where possible.
        - The matrix does not exceed the 5x5 limit.

        Args:
            individual (networkx.DiGraph): The directed graph representing the structure.

        Returns:
            tuple: A 5x5 numpy array representing the structure and its connectivity.
        """
        # Remove disconnected nodes
        individual = MutationHandler.garbage_collect_nodes(individual)

        rows, cols = self.grid_size
        grid = np.zeros((rows, cols), dtype=int)
        visited = set()

        # Start BFS from a central position to avoid running out of space
        start_node = list(individual.nodes())[0]  # Pick an arbitrary start node
        queue = deque([(start_node, (rows // 2, cols // 2))])  # Start at center of grid
        visited.add(start_node)
        grid[rows // 2, cols // 2] = individual.nodes[start_node]['type']  # Store block type

        # Directions for right, down, left, up (ensures adjacency)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            node, (r, c) = queue.popleft()

            # Iterate over node's successors (maintaining adjacency structure)
            for neighbor in individual.successors(node):
                if neighbor in visited:
                    continue
                # look for any empty neighbor cell
                for dr, dc in directions:
                    rr, cc = r + dr, c + dc
                    if (0 <= rr < rows and 0 <= cc < cols
                            and grid[rr, cc] == 0):
                        grid[rr, cc] = individual.nodes[neighbor]['type']
                        visited.add(neighbor)
                        queue.append((neighbor, (rr, cc)))
                        break

        # Compute full connectivity for Evogym
        connectivity = evogym.get_full_connectivity(grid)
        return np.array(grid), connectivity

    def generate_graph(self) -> nx.DiGraph:
        rows, _ = self.grid_size
        num_nodes = random.randint(3, 15)
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

    def remove_duplicates(self, max_nodes: int = 25):
        unique_graphs = []
        counter = 0
        for graph in self.population:
            is_duplicate = any(nx.is_isomorphic(graph, other,
                                                node_match=lambda x, y: x['type'] == y['type']) for other in
                               unique_graphs)
            if not is_duplicate:
                unique_graphs.append(graph)
            else:
                counter += 1
                new_graph = self.generate_graph()
                new_graph = self.enforce_max_nodes(new_graph, max_nodes)

                # Ensure new_graph is not a duplicate either
                while any(nx.is_isomorphic(new_graph, g,
                                           node_match=lambda x, y: x['type'] == y['type']) for g in unique_graphs):
                    new_graph = self.generate_graph()
                    new_graph = self.enforce_max_nodes(new_graph, max_nodes)

                unique_graphs.append(new_graph)

        self.population = unique_graphs

    def save_to_csv(self):
        if self.result is None:
            raise ValueError("Result not available yet")

        result_to_dict = {
            "Average Fitness": self.result["average_fitness"],
            "Average Reward": self.result["average_reward"],
            "Best Fitness": self.result["best_fitness"],
            "Best Reward": self.result["best_reward"],
            "Best Structure": self.result["best_structure"]
        }

        # Create dataframe
        df = pd.DataFrame(result_to_dict)

        # Verify if it's testing seeds or running
        if self.testing:
            path = f"./testing/fixed_controller/{self.seed}/{self.controller_type}/{self.scenario}/"
        else:
            path = f"./data/fixed_controller/{self.controller_type}/{self.scenario}/"

        # Create all intermediate directories if they don't exist
        os.makedirs(path, exist_ok=True)
        filename = path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".csv"

        # Save to CSV
        df.to_csv(filename, index=False)

    def get_result(self) -> Result:
        return self.result

    # ===== EA FUNCTIONS =====

    def initialize_population(self):
        """
        Create initial random population.

        Returns:
            List of individuals, where each individual is a list of genes (floats)
        """
        population = []
        for _ in range(self.population_size):
            graph = self.generate_graph()
            population.append(graph)

        self.population = population

    def evaluate_fitness(self, individual: nx.DiGraph, view: bool = False) -> Tuple[float, float]:
        """
        Mock fitness function - evaluates how good an individual is.
        In a real implementation, this would be your problem-specific function.

        Args:
            individual: The structure to evaluate

        Returns:
            Fitness score (higher is better)
            :param individual:
            :param view:
        """
        structure, connectivity = self.graph_to_matrix(individual)
        if (evogym.is_connected(structure) is False or
                connectivity.shape[1] == 0 or
                evogym.has_actuator(structure) is False):
            return np.nan, np.nan
        try:
            env = gym.make(self.scenario, max_episode_steps=self.steps, body=structure, connections=connectivity)
            env.reset()
            current_sim = env.sim
            current_viewer = evogym.EvoViewer(current_sim)
            current_viewer.track_objects(('robot',))
            action_size = current_sim.get_dim_action_space('robot')
            t, t_reward, final_pos, average_speed = 0, 0, 0, 0
            controller = Controller(controller_type=self.controller_type, action_size=action_size)
            initial_pos = current_sim.object_pos_at_time(current_sim.get_time(), 'robot').mean()
            for t in range(self.steps):
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

    def select_parents(self, fitness_scores: List[float]) -> List[nx.DiGraph]:
        """
        Select parents for reproduction using tournament selection.

        Args:
            fitness_scores: List of fitness scores for the current population

        Returns:
            Selected parents
        """
        selected = []
        for _ in range(len(self.population)):
            candidates = np.random.choice(len(self.population), size=self.tournament_size, replace=False)
            winner = candidates[np.argmax([fitness_scores[c] for c in candidates])]
            selected.append(copy.deepcopy(self.population[winner]))
        return selected

    def crossover(self, parent1: nx.DiGraph, parent2: nx.DiGraph) -> nx.DiGraph:
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

    def mutate(self, individual: nx.DiGraph) -> nx.DiGraph:
        """
        Mutates a directed graph-based robot structure by randomly choosing one or two
        of the five possible mutation techniques (defined in MutationHandler) to apply.

        Parameters:
          individual (nx.DiGraph): The directed graph representing the robot.

        Returns:
          nx.DiGraph: The mutated graph.
        """

        num_elements = len(individual.nodes()) + len(individual.edges())
        scaling = 1.0 / num_elements if num_elements > 0 else 1.0
        mutation_prob = self.mutation_rate * scaling

        handler = MutationHandler(mutation_prob)

        mutation_techniques = [
            handler.mutate_node_parameters,
            handler.mutate_connection_parameters,
            handler.add_remove_connections
        ]

        if len(individual.nodes()) <= 25:
            mutation_techniques.append(handler.add_new_node)

        num_to_apply = random.choice([1, 2])
        techniques_to_apply = random.sample(mutation_techniques, num_to_apply)

        mutated_individual = individual.copy()

        for technique in techniques_to_apply:
            mutated_individual = technique(mutated_individual)

        mutated_individual = MutationHandler.garbage_collect_nodes(mutated_individual)
        return self.enforce_max_nodes(mutated_individual)

    def elitism(self, fitness_scores: List[float]) -> List[nx.DiGraph]:
        """Preserves the top `elite_size` individuals."""
        elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
        return [copy.deepcopy(self.population[i]) for i in elite_indices]

    def evolve(self, show_progress=False):
        """
            Run the evolutionary algorithm for a given number of generations.
        """

        def track_best():
            try:
                # Track best individual
                current_best_fitness_idx = np.nanargmax(fitness_scores)
                current_best_reward_idx = np.nanargmax(rewards)
                fitness_scores[generation] = fitness_scores[current_best_fitness_idx]
                best_structures[generation] = self.graph_to_matrix(self.population[current_best_fitness_idx])[0]
                best_rewards[generation] = rewards[current_best_reward_idx]
                # Track average
                avg_fitness[generation] = np.nanmean(fitness_scores)
                avg_rewards[generation] = np.nanmean(rewards)
            except ValueError:
                fitness_scores[generation] = 0
                best_structures[generation] = 0
                best_rewards[generation] = 0
                avg_fitness[generation] = 0
                avg_rewards[generation] = 0

        # Update Randomness
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Initialize data structures
        best_structures = np.empty((self.num_generations, 5, 5))
        best_fitness_scores = np.empty(self.num_generations, dtype=np.float64)
        best_rewards = np.empty(self.num_generations, dtype=np.float64)
        avg_fitness = np.empty(self.num_generations, dtype=np.float64)
        avg_rewards = np.empty(self.num_generations, dtype=np.float64)

        with (ProcessPoolExecutor(max_workers=os.cpu_count()) as executor):
            for generation in range(self.num_generations):
                # Remove and Replace duplicates in population
                self.remove_duplicates()

                # Evaluate fitness with parallelism
                evaluation_func = partial(EvolveStructure.evaluate_fitness, self)
                results = list(executor.map(evaluation_func, self.population))
                fitness_scores, rewards = zip(*results)
                fitness_scores = list(fitness_scores)
                rewards = list(rewards)

                # Track results
                track_best()

                # Selection
                parents = self.select_parents(fitness_scores)

                # Crossover
                offspring = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        child1 = self.crossover(parents[i], parents[i + 1])
                        child2 = self.crossover(parents[i + 1], parents[i])
                        offspring.extend([child1, child2])
                    else:
                        offspring.append(parents[i].copy())

                # Mutation
                offspring = [self.mutate(child) for child in offspring]

                # Elitism
                elites = self.elitism(fitness_scores)
                offspring = elites + offspring

                # Update population
                self.population = offspring[:self.population_size]

                if show_progress:
                    print(f"Gen {generation + 1}: Best Fitness = {best_fitness_scores[generation]:.2f}, "
                          f"Its Reward = {best_rewards[generation]:.2f}")

        data = {
            "average_fitness": avg_fitness,
            "average_reward": avg_rewards,
            "best_fitness": best_fitness_scores,
            "best_reward": best_rewards,
            "best_structure": [",".join(map(str, mat.flatten())) for mat in best_structures],
        }

        self.result = data


def run(batches, seed, controller, scenario, testing=False, view=False):
    for iteration in range(batches):
        EA = EvolveStructure(seed=seed, controller_type=controller, scenario=scenario, testing=testing)
        EA.evolve()
        result = EA.get_result()

        print(f"===== Iteration {iteration} =====")
        print(f"Best Fitness Achieved: {result['best_fitness'][-1]}")
        print(f"Best Reward Achieved: {result['best_reward'][-1]}")
        print("============================")

        EA.save_to_csv()

        # Visualize the best structure
        if view:
            print("Visualizing the best robot...")
            # structure_in_plain_text = result['best_structure'][-1]
            # TODO: implement plain text to matrix conversion
            # EA.evaluate_fitness(best_structures[-1], view=True)


def seeds_():
    SCENARIOS = ['BridgeWalker-v0', 'Walker-v0']
    CONTROLLERS = ['alternating_gait', 'sinusoidal_wave', 'hopping_motion']
    SEEDS = [42, 0, 123, 987, 314159, 271828, 2 ** 32 - 1]
    for seed in SEEDS:
        for scenario in SCENARIOS:
            for controller in CONTROLLERS:
                run(batches=1, seed=seed, controller=controller, scenario=scenario, testing=True)


if __name__ == "__main__":
    _SCENARIOS = ['BridgeWalker-v0', 'Walker-v0']
    for _scenario in _SCENARIOS:
        run(batches=5, seed=271828, controller='alternating_gait', scenario=_scenario)
