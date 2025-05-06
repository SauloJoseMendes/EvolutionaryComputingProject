import networkx as nx
from ..evolve_structure import fixed_controller as fc
from ..evolve_controller import neural_controller
from evogym.envs import *


class Genome:
    def __init__(self, scenario: str, steps: int = 500):
        self.structure = None
        self.controller = None
        self.scenario = scenario
        self.steps = steps

    def initialize_structure(self):
        self.structure = fc.generate_fully_connected_graph()

    def get_structure(self) -> nx.DiGraph:
        return self.structure

    def set_structure(self, structure: nx.DiGraph):
        self.structure = structure

    def initialize_controller(self):
        robot, connections = fc.graph_to_matrix(self.get_structure())
        env = gym.make(self.scenario, max_episode_steps=self.steps, body=robot, connections=connections)
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        env.close()

        self.structure = neural_controller.NeuralController(input_size, output_size)
        
    def get_controller(self):
        return self.controller.get_weights()

    def set_controller(self, weights):
        self.controller.set_weights(weights)
