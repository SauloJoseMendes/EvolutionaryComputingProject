import copy
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
import evogym
import pandas as pd
import torch
from evogym.envs import *
import fixed_structure_hybrid as fs_h
import fixed_controller_GP as es
import fixed_structure_hybrid as ec
from AuxiliaryClasses.Genome import Genome
import os
import time
import numpy as np
import random
from typing import List, Tuple

SCENARIOS = ['GapJumper-v0', 'CaveCrawler-v0']
SEEDS = [42, 0, 123, 987, 314159, 271828, 2 ** 32 - 1]

BATCH_SIZE = 1
NUM_GENERATIONS_BOTH = 100
NUM_GENERATIONS_SINGLE = 150
NUM_GENERATIONS = NUM_GENERATIONS_BOTH + NUM_GENERATIONS_SINGLE
POPULATION_SIZE = 100
TOURNAMENT_SIZE = 3
MUTATION_RATE = 0.4
ELITISM_SIZE = 2
STEPS = 500

def save(data_csv, scenario):
    best_genomes = data_csv.pop("Best Genomes")
    df = pd.DataFrame(data_csv)
    run_path = f"../../evolve_both/static/data/runs/{scenario}/{NUM_GENERATIONS}/"
    genomes_path = f"../../evolve_both/static/data/genomes/{scenario}/{NUM_GENERATIONS}/"
    os.makedirs(run_path, exist_ok=True)
    run_filename = run_path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".csv"
    df.to_csv(run_filename, index=False)
    os.makedirs(genomes_path, exist_ok=True)
    genomes_filename = genomes_path + time.strftime("%Y_%m_%d_at_%H_%M_%S") + ".pt"
    Genome.save_genomes(best_genomes, genomes_filename)

def initialize_population(scenario) -> List[Genome]:
    genomes = []
    for _ in range(POPULATION_SIZE):
        genomes.append(Genome(scenario=scenario))
    return genomes

def evaluate(individual: Genome, scenario: str, view: bool = False) -> Tuple[float, float]:
    try:
        controller = individual.get_controller()
        individual_structure = individual.get_structure()
        robot, connections = es.graph_to_matrix(individual_structure)
        env = gym.make(scenario, max_episode_steps=STEPS, body=robot, connections=connections)
        state = env.reset()
        w = env.observation_space.shape[0]
        z = env.action_space.shape[0]
        isCompatible, penalty = individual.get_controller().check_compatibility(w, z)
        if isCompatible is False:
            return penalty, 0
        if isinstance(state, tuple):
            state = state[0]
        current_sim = env.sim
        current_viewer = evogym.EvoViewer(current_sim)
        current_viewer.track_objects(('robot',))
        initial_pos = current_sim.object_pos_at_time(current_sim.get_time(), 'robot').mean()
        t = 0
        total_reward = 0.0
        speed_accum = 0.0
        final_pos = initial_pos
        for t in range(STEPS):
            if view:
                current_viewer.render('screen')
            obs_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = controller(obs_tensor)
            action = action_tensor.detach().cpu().numpy().squeeze()
            state, reward, terminated, truncated, info = env.step(action)
            if isinstance(state, tuple):
                state = state[0]
            total_reward += reward
            speed_accum += current_sim.object_vel_at_time(current_sim.get_time(), 'robot').mean()
            if terminated or truncated:
                final_pos = current_sim.object_pos_at_time(current_sim.get_time(), 'robot').mean()
                break
        distance = final_pos - initial_pos
        avg_speed = speed_accum / (t + 1) if t >= 0 else 0.0
        final_fitness = (distance * 5.0) + (avg_speed * 2.0)
        current_viewer.close()
        env.close()
        return final_fitness, total_reward
    except ValueError as error_fitness:
        return -np.inf, -np.inf

def select_parents(population, fitness_scores, num_winners=1):
    winners: List[Genome] = []
    for _ in range(num_winners):
        indices = np.random.choice(POPULATION_SIZE, size=TOURNAMENT_SIZE, replace=False)
        best_idx = max(indices, key=lambda idx: fitness_scores[idx])
        winners.append(copy.deepcopy(population[best_idx]))
    return winners

def crossover(parent1: Genome, parent2: Genome, both: bool = True) -> Genome:
    parent1_controller = parent1.get_controller()
    parent2_controller = parent2.get_controller()
    offspring = copy.deepcopy(parent1)
    offspring_controller = ec.crossover(parent1_controller, parent2_controller)
    offspring.set_controller(offspring_controller)
    if both:
        parent1_graph = parent1.get_structure()
        parent2_graph = parent2.get_structure()
        offspring_graph = es.grafting_crossover(parent1_graph, parent2_graph)
        offspring.set_structure(offspring_graph)
    return offspring

def mutate(individual: Genome, both: bool = True) -> Genome:
    if both:
        individual_graph = individual.get_structure()
        resulting_graph = es.mutate_structure(individual_graph, mutation_rate=MUTATION_RATE)
        individual.set_structure(resulting_graph)
    individual_weights = individual.get_weights()
    resulting_weights = ec.mutate(weights=individual_weights, mutation_strength=MUTATION_RATE)
    individual.set_weights(resulting_weights)
    return individual

def elitism(population: List[Genome], fitness_scores: List[float]) -> List[Genome]:
    sorted_idx = np.argsort(fitness_scores)[::-1][:ELITISM_SIZE]
    return [copy.deepcopy(population[i]) for i in sorted_idx]

def safe_evaluate(individual, scenario):
    try:
        return evaluate(individual, scenario=scenario)
    except Exception as e:
        print(f"Evaluation crashed for individual: {e}")
        return (-np.inf, -np.inf)

def evolve(scenario: str, debug=True):
    def filter_results(results: List):
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

    best_genomes = np.empty(NUM_GENERATIONS, dtype=object)
    best_fitness_scores = np.empty(NUM_GENERATIONS)
    best_rewards = np.empty(NUM_GENERATIONS)
    avg_fitness = np.full(NUM_GENERATIONS, np.nan)
    avg_rewards = np.full(NUM_GENERATIONS, np.nan)
    np.random.seed(None)
    random.seed(None)
    torch.manual_seed(int(time.time()))
    population = initialize_population(scenario)
    both = True
    with (ProcessPoolExecutor(max_workers=4) as executor):
        for generation in range(NUM_GENERATIONS):
            if generation > NUM_GENERATIONS_BOTH:
                both = False
            try:
                safe_evaluator = partial(safe_evaluate, scenario=scenario)
                results = list(executor.map(safe_evaluator, population))
                fitness_scores, rewards = filter_results(results)
            except BrokenProcessPool:
                print("Warning: Process pool crashed completely")
                fitness_scores = np.full(len(population), -np.inf)
                rewards = np.full(len(population), -np.inf)
            except Exception as e:
                print(f"Unexpected error during parallel evaluation: {e}")
                fitness_scores = np.full(len(population), -np.inf)
                rewards = np.full(len(population), -np.inf)

            current_best_fitness_idx = np.argmax(fitness_scores)
            current_best_reward_idx = np.argmax(rewards)
            best_fitness_scores[generation] = fitness_scores[current_best_fitness_idx]
            best_genomes[generation] = population[current_best_fitness_idx]
            best_rewards[generation] = rewards[current_best_reward_idx]
            fitness_scores = np.array(fitness_scores)
            rewards = np.array(rewards)
            fitness_scores_c = fitness_scores[np.isfinite(fitness_scores)]
            rewards_c = rewards[np.isfinite(rewards)]
            if rewards_c.shape[0] == 0:
                avg_rewards[generation] = np.nan
            else:
                avg_rewards[generation] = np.nanmean(rewards_c)
            if fitness_scores_c.shape[0] == 0:
                avg_fitness[generation] = np.nan
            else:
                avg_fitness[generation] = np.nanmean(fitness_scores_c)
            elites = elitism(population, list(fitness_scores))
            offspring: List[Genome] = []
            while len(offspring) < POPULATION_SIZE - len(elites):
                parents = select_parents(population=population, fitness_scores=list(fitness_scores), num_winners=2)
                child = crossover(parents[0], parents[1], both=both)
                mutate(child, both=both)
                offspring.append(child)
            population = elites + offspring
            if debug:
                print(f"Generation {generation + 1} best fitness: {best_fitness_scores[generation]:.3f}, "
                      f"best reward : {best_rewards[generation]}")

    return best_genomes, best_fitness_scores, best_rewards, avg_fitness, avg_rewards

def run(scenario, batches=1):
    for iteration in range(batches):
        best_genomes, best_fitnesses, best_rewards, avg_fitness, avg_reward = evolve(scenario=scenario)
        print(f"===== Iteration {iteration} =====")
        print(f"Best Fitness Achieved: {best_fitnesses[-1]}")
        print(f"Best Reward Achieved: {best_rewards[-1]}")
        data = {
            "Average Fitness": avg_fitness,
            "Average Reward": avg_reward,
            "Best Fitness": best_fitnesses,
            "Best Reward": best_rewards,
            "Best Genomes": best_genomes,
        }
        save(data, scenario)
        print("============================")

if __name__ == '__main__':
    for _scenario in ['CaveCrawler-v0']:
        run(batches=5, scenario=_scenario)
    print("\n\n\n\n VOU CORRER AGORA F_S\n\n\n")
    for _scenario in ['DownStepper-v0', 'ObstacleTraverser-v0-v0']:
        fs_h.run(batches=5, scenario=_scenario)
