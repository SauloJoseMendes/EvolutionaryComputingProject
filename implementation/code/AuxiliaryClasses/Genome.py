import evogym
import networkx as nx
import torch
from evogym.envs import *
from AuxiliaryClasses.NeuralController import NeuralController, get_weights, set_weights
from Evolve.fixed_controller_GP import generate_fully_connected_graph, graph_to_matrix


class Genome:
    def __init__(self, scenario: str, steps: int = 500, seed=271828):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.scenario = scenario
        self.steps = steps
        self.structure = None
        self.controller = None
        self.input = []
        self.input = []
        self.initialize_structure()
        self.initialize_controller()

    def initialize_structure(self):
        self.structure = generate_fully_connected_graph()

    def initialize_controller(self):
        robot = np.array([
            [1, 3, 1, 0, 0],
            [4, 1, 3, 2, 2],
            [3, 4, 4, 4, 4],
            [3, 0, 0, 3, 2],
            [0, 0, 0, 0, 2]
        ])
        connections = evogym.get_full_connectivity(robot)
        env = gym.make(self.scenario, max_episode_steps=self.steps, body=robot, connections=connections)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        env.close()
        self.input_size = input_size
        self.output_size = output_size
        self.controller = NeuralController(input_size, output_size)

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
        genome.initialize_controller()
        set_weights(genome.controller, data['weights'])
        set_weights(genome.controller, data['weights'])
        return genome

    @staticmethod
    def save_genomes(genomes, path):
        """One-liner with basic safety"""
        torch.save([g.to_dict() for g in genomes], path)

    @staticmethod
    def load_genomes(path):
        return [Genome.from_dict(d) for d in torch.load(path)]



