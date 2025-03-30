import networkx as nx
import random
import numpy as np


class MutationHandler:
    def __init__(self, mutation_prob):
        self.mutation_prob = mutation_prob

    def mutate_node_parameters(self, graph):
        for node, data in graph.nodes(data=True):
            if "type" in data and random.random() < self.mutation_prob:
                data["type"] = random.choice([1, 2, 3, 4])
        return graph

    def add_new_node(self, graph):
        if random.random() < self.mutation_prob:
            new_node_id = max(graph.nodes()) + 1 if graph.nodes() else 0
            graph.add_node(new_node_id, type=random.choice([1, 2, 3, 4]), weight=random.uniform(0.5, 5.0))

            # Add a connection to an existing node if the graph is not empty
            existing_nodes = list(graph.nodes())
            if existing_nodes and new_node_id in graph.nodes():  # Ensure the new node was actually added
                existing_node = random.choice(existing_nodes)
                # Add a directed edge from the new node to the existing node
                graph.add_edge(new_node_id, existing_node, weight=random.uniform(0.1, 1.0))
                # Or, add a directed edge from the existing node to the new node
                # graph.add_edge(existing_node, new_node_id, weight=random.uniform(0.1, 1.0))
                # Or, add an undirected edge (if your graph is treated as undirected for connectivity)
                # graph.add_edge(existing_node, new_node_id, weight=random.uniform(0.1, 1.0)) # For NetworkX DiGraph, this is still a directed edge

        return graph

    def mutate_connection_parameters(self, graph):
        for src, tgt, data in list(graph.edges(data=True)):
            if random.random() < self.mutation_prob:
                if "weight" in data:
                    adjustment = np.random.normal(0, 0.05)
                    data["weight"] += adjustment
                    data["weight"] = max(0.1, data["weight"])
                if random.random() < 0.5 * self.mutation_prob:
                    possible_targets = list(graph.nodes())
                    new_tgt = random.choice(possible_targets)
                    if new_tgt != src:
                        graph.remove_edge(src, tgt)
                        graph.add_edge(src, new_tgt, **data)
        return graph

    def add_remove_connections(self, graph):
        for node in list(graph.nodes()):
            if random.random() < self.mutation_prob:
                target = random.choice(list(graph.nodes()))
                if node != target and not graph.has_edge(node, target):
                    graph.add_edge(node, target, weight=random.uniform(0.1, 1.0))
            if random.random() < self.mutation_prob:
                if graph.out_degree[node] > 0:
                    edge_to_remove = random.choice(list(graph.out_edges(node)))
                    source, target = edge_to_remove
                    graph.remove_edge(source, target)
                    break
        return graph

    @staticmethod
    def garbage_collect_nodes(graph):
        if 0 in graph.nodes():
            reachable_nodes = set(nx.descendants(graph, 0))
            reachable_nodes.add(0)
            for node in list(graph.nodes()):
                if node not in reachable_nodes:
                    graph.remove_node(node)
        return graph
