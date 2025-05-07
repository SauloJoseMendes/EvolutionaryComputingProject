import networkx as nx
import torch
from evogym.envs import *
from AuxiliaryClasses.NeuralController import NeuralController, get_weights, set_weights
from Evolve.fixed_controller import generate_fully_connected_graph, graph_to_matrix


class Genome:
    def __init__(self, scenario: str, steps: int = 500, seed=271828):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.scenario = scenario
        self.steps = steps
        self.structure = self.initialize_structure()
        self.controller = self.controller = initialize_controller(self.structure, self.scenario, self.steps)

    @staticmethod
    def initialize_structure() -> nx.DiGraph:

        structure = generate_fully_connected_graph()
        return structure

    def get_structure(self) -> nx.DiGraph:
        return self.structure

    def set_structure(self, structure: nx.DiGraph):
        self.structure = structure

    def get_weights(self):
        return get_weights(self.controller)

    def set_weights(self, weights):
        set_weights(self.controller, weights)

    def set_controller(self, controller: NeuralController):
        self.controller = controller

    def get_controller(self):
        return self.controller

    def to_dict(self):
        """Convert to serializable dictionary"""
        return {
            'scenario': self.scenario,
            'steps': self.steps,
            'structure': nx.node_link_data(self.structure),
            'weights': get_weights(self.controller)
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstruct from dictionary"""
        genome = cls(data['scenario'], data['steps'])
        genome.structure = nx.node_link_graph(data['structure'])
        genome.controller = Genome.initialize_controller(genome.structure, data['scenario'], data['steps'])
        set_weights(genome.controller, data['weights'])
        return genome

    @staticmethod
    def save_genomes(genomes, path):
        """One-liner with basic safety"""
        torch.save([g.to_dict() for g in genomes], path)

    @staticmethod
    def load_genomes(path):
        return [Genome.from_dict(d) for d in torch.load(path)]


def initialize_controller(structure: nx.DiGraph, scenario: str, steps: int = 500) -> NeuralController:
    robot, connections = graph_to_matrix(structure)

    env = gym.make(scenario, max_episode_steps=steps, body=robot, connections=connections)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    env.close()

    return NeuralController(input_size, output_size)
