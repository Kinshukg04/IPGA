import numpy as np
import random
from typing import List, Tuple

# Input data from Tables 1-4
levelized_costs = np.array([3.00, 4.50, 3.30, 2.90, 3.30, 3.30, 3.80, 4.70, 17.00, 15.00])
investment_risks = np.array([0.09, 0.06, 0.54, 0.10, 0.17, 0.15, 0.62, 0.85, 0.98, 0.79])
fuel_risks = np.array([0.06, 0.10, 0.46, 0.00, 0.00, 0.00, 0.10, 0.10, 0.00, 0.00])
om_risks = np.array([0.05, 0.04, 0.40, 0.05, 0.12, 0.09, 0.81, 1.02, 0.20, 0.20])
carbon_costs = np.array([0.15, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
barrier_indices = np.array([0.1066, 0.1040, 0.2362, 0.1012, 0.0752, 0.0838, 0.1026, 0.1120, 0.0330, 0.0456])

# Technology labels
technologies = [
    "Coal", "Natural gas", "Nuclear", "Hydro",
    "Wind", "Small hydro", "Biomass", "Waste to energy",
    "Solar thermal", "Solar PV"
]

# Combine risks into a single matrix for easier computation
all_risks = np.array([investment_risks, fuel_risks, om_risks, carbon_costs])

def initialize_population(population_size: int, num_assets: int) -> np.ndarray:
    """Initialize a population of random portfolios."""
    population = np.random.rand(population_size, num_assets)
    return population / population.sum(axis=1)[:, np.newaxis]

def fitness_function(portfolio: np.ndarray) -> Tuple[float, float]:
    """Calculate fitness scores (cost and CRBI) for a portfolio."""
    cost = np.sum(portfolio * levelized_costs) + np.sum(portfolio * carbon_costs)  # Adjusted carbon cost
    combined_risk = np.sqrt(np.sum((portfolio @ all_risks.T) ** 2))
    crbi = np.sum(portfolio * barrier_indices) * combined_risk
    return cost, crbi

def apply_constraints(portfolio: np.ndarray) -> bool:
    """Apply constraints to ensure feasibility of the portfolio."""
    # Example constraint: Ensure minimum contribution from renewable sources
    non_renewables = portfolio[3:10].sum()  # Assuming renewable indices are from Hydro onwards
    if portfolio[1]>0.03 and portfolio[0]>0.2 and non_renewables >= 0.1 and all(portfolio)>0:
        flag = True
    else:
        flag = False
    return flag 

def non_dominated_sorting(population: np.ndarray, fitness_scores: List[Tuple[float, float]]) -> List[List[int]]:
    """Perform non-dominated sorting to identify Pareto fronts."""
    n = len(population)
    dominated = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(fitness_scores[i], fitness_scores[j]):
                    dominated[i].append(j)
                elif dominates(fitness_scores[j], fitness_scores[i]):
                    domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominated[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        fronts.append(next_front)

    return [f for f in fronts if f]

def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Check if solution a dominates solution b."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def crowding_distance_assignment(front: List[int], fitness_scores: List[Tuple[float, float]]) -> List[float]:
    """Calculate crowding distance for solutions in a front."""
    if len(front) <= 2:
        return [float('inf')] * len(front)

    distances = [0.0] * len(front)
    front_size = len(front)

    for objective in range(2):
        objective_values = [(idx, fitness_scores[front[idx]][objective]) for idx in range(front_size)]
        objective_values.sort(key=lambda x: x[1])

        distances[objective_values[0][0]] = float('inf')
        distances[objective_values[-1][0]] = float('inf')

        objective_range = objective_values[-1][1] - objective_values[0][1]
        if objective_range == 0:
            continue

        for i in range(1, front_size - 1):
            distances[objective_values[i][0]] += (
                objective_values[i + 1][1] - objective_values[i - 1][1]
            ) / objective_range

    return distances

def tournament_selection(population: np.ndarray, fitness_scores: List[Tuple[float, float]]) -> np.ndarray:
    """Select a parent using tournament selection."""
    i, j = random.sample(range(len(population)), 2)
    if dominates(fitness_scores[i], fitness_scores[j]):
        return population[i]
    elif dominates(fitness_scores[j], fitness_scores[i]):
        return population[j]
    else:
        return population[random.choice([i, j])]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    """Perform crossover between two parents."""
    alpha = np.random.rand()
    child = alpha * parent1 + (1 - alpha) * parent2
    return child / np.sum(child)

def mutate(portfolio: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Apply mutation to a portfolio."""
    mutated = portfolio.copy()
    for i in range(len(portfolio)):
        if random.random() < mutation_rate:
            mutated[i] += np.random.normal(0, 0.1)
    mutated = np.clip(mutated, 0, 1)
    return mutated / np.sum(mutated)

def ipga(population_size: int, generations: int, elite_size: int, 
         mutation_rate: float) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
    """Main function for Intelligent Pareto-search Genetic Algorithm (IPGA)."""
    population = initialize_population(population_size, len(levelized_costs))
    elite_population = []
    
    # Track the absolute best solution across all generations
    best_overall_solution = None
    best_overall_cost = float('inf')
    best_overall_crbi = float('inf')

    for generation in range(generations):
        # Calculate fitness and apply constraints
        fitness_scores = [fitness_function(p) for p in population]
        population = [p for p, f in zip(population, fitness_scores) if apply_constraints(p)]
        fitness_scores = [f for p, f in zip(population, fitness_scores) if apply_constraints(p)]

        # Perform non-dominated sorting
        fronts = non_dominated_sorting(population, fitness_scores)
        if(len(fronts)==0):
            continue
        pareto_front = fronts[0]

        # Calculate crowding distances
        crowding_distances = crowding_distance_assignment(pareto_front, fitness_scores)
        
        # Select elite solutions based on crowding distance
        elite_indices = sorted(range(len(crowding_distances)), 
                               key=lambda k: crowding_distances[k], 
                               reverse=True)[:elite_size]

        # Store elite solutions
        
        sorted_elites = [(population[pareto_front[i]], fitness_scores[pareto_front[i]]) for i in elite_indices]
        elite_population.extend(sorted_elites)

        # Find the best solution in this generation
        generation_best_solution = min(zip(population, fitness_scores), key=lambda x: (x[1][0], x[1][1]))
        best_portfolio, (best_cost, best_crbi) = generation_best_solution

        # Update overall best solution
        if best_cost < best_overall_cost and best_crbi < best_overall_crbi:
            best_overall_solution = generation_best_solution
            best_overall_cost = best_cost
            best_overall_crbi = best_crbi

        # Generate new population
        new_population = []
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            if apply_constraints(child):
                new_population.append(child)
        
        population = np.array(new_population)

        # Print generation's best solution
        generation_mix = best_portfolio * 100  # Convert to percentages
        mix_dict = {tech: float(mix) for tech, mix in zip(technologies, generation_mix)}
        # print(f"Generation {generation}: Best Cost = {best_cost:.2f}, "
        #       f"Best CRBI = {best_crbi:.2f}, Optimal Mix: {mix_dict}")

    # Print the absolute best solution across all generations
    if best_overall_solution:
        best_portfolio, (best_cost, best_crbi) = best_overall_solution
        generation_mix = best_portfolio * 100
        mix_dict = {tech: float(mix) for tech, mix in zip(technologies, generation_mix)}
        
        print("\n--- Best Overall Solution Across Generations ---")
        print(f"Total Generation Mix: {sum(generation_mix):.2f}%")
        print("Technology Mix (%):")
        for tech, mix in mix_dict.items():
            print(f"{tech}: {mix:.2f}")
        print(f"\nBest Total Cost: {best_cost:.2f}")
        print(f"Best Comprehensive Risk Barrier Index (CRBI): {best_crbi:.4f}")

    return elite_population

# Example usage
if __name__ == "__main__":
    POPULATION_SIZE = 1000
    GENERATIONS = 100
    ELITE_SIZE = 50
    MUTATION_RATE = 0.1

    optimal_solutions = ipga(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_size=ELITE_SIZE,
        mutation_rate=MUTATION_RATE
    )
    print("Number of optimal solutions found:", len(optimal_solutions))
