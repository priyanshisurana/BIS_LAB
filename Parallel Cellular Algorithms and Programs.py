import numpy as np
import random
from multiprocessing import Pool


# Define the objective function to optimize (e.g., Rastrigin function)
def objective_function(solution):
    return sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10 for x in solution)


# Initialize the grid population randomly
def initialize_population(grid_size, solution_dim, bounds):
    return [
        [np.random.uniform(bounds[0], bounds[1], solution_dim) for _ in range(grid_size)]
        for _ in range(grid_size)
    ]


# Perform crossover operation
def crossover(parent1, parent2):
    alpha = np.random.rand()  # Crossover ratio
    child = alpha * parent1 + (1 - alpha) * parent2
    return np.clip(child, -5.12, 5.12)  # Ensure child stays within bounds


# Perform mutation operation
def mutate(solution, mutation_rate, bounds):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] += np.random.uniform(bounds[0] * 0.1, bounds[1] * 0.1)
    return np.clip(solution, bounds[0], bounds[1])


# Get neighbors for a specific cell
def get_neighbors(grid, x, y):
    neighbors = []
    grid_size = len(grid)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = (x + dx) % grid_size, (y + dy) % grid_size
            neighbors.append(grid[nx][ny])
    return neighbors


# Evaluate fitness and evolve cell
def evolve_cell(args):
    x, y, grid, mutation_rate, bounds = args
    solution = grid[x][y]
    neighbors = get_neighbors(grid, x, y)

    # Select the best neighbor
    best_neighbor = min(neighbors, key=objective_function)

    # Crossover and mutation
    child = crossover(solution, best_neighbor)
    child = mutate(child, mutation_rate, bounds)

    return child


# Main parallel optimization routine
def parallel_cellular_optimization(grid_size, solution_dim, bounds, generations, mutation_rate, processes=4):
    # Initialize the population grid
    grid = initialize_population(grid_size, solution_dim, bounds)

    # Run generations
    for generation in range(generations):
        with Pool(processes=processes) as pool:
            args = [(x, y, grid, mutation_rate, bounds) for x in range(grid_size) for y in range(grid_size)]
            new_population = pool.map(evolve_cell, args)

        # Update the grid with the new solutions
        grid = [
            new_population[row * grid_size:(row + 1) * grid_size]
            for row in range(grid_size)
        ]

        # Logging
        best_solution = min((solution for row in grid for solution in row), key=objective_function)
        best_fitness = objective_function(best_solution)
        print(f"Generation {generation + 1}/{generations}, Best Fitness: {best_fitness:.4f}")

    # Final best solution
    best_solution = min((solution for row in grid for solution in row), key=objective_function)
    return best_solution


if __name__ == "__main__":
    # Parameters
    GRID_SIZE = 10
    SOLUTION_DIM = 5
    BOUNDS = (-5.12, 5.12)  # Range of Rastrigin function
    GENERATIONS = 50
    MUTATION_RATE = 0.1

    # Run optimization
    best = parallel_cellular_optimization(
        GRID_SIZE, SOLUTION_DIM, BOUNDS, GENERATIONS, MUTATION_RATE, processes=4
    )
    print("Best Solution Found:", best)
    print("Best Fitness:", objective_function(best))

