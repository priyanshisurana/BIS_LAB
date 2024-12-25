import random
import numpy as np

# Step 1: Define the objective function (e.g., for minimization)
def objective_function(solution):
    # Example: Minimize the sum of squares of the solution
    return sum([x ** 2 for x in solution])

# Step 2: Initialize Parameters
def initialize_parameters(population_size, num_genes, mutation_rate, crossover_rate):
    return population_size, num_genes, mutation_rate, crossover_rate

# Step 3: Initialize Population with random genetic sequences
def initialize_population(population_size, num_genes):
    return np.random.uniform(-5, 5, (population_size, num_genes))  # Initialize with random floats in a given range

# Step 4: Evaluate fitness for the population
def evaluate_fitness(population):
    return np.array([objective_function(ind) for ind in population])

# Step 5: Selection (roulette-wheel selection)
def selection(population, fitness):
    fitness_sum = np.sum(fitness)
    selection_probs = (fitness_sum - fitness) / fitness_sum  # Minimize fitness
    selected_parents_indices = np.random.choice(range(len(population)), size=len(population), p=selection_probs / selection_probs.sum())
    return population[selected_parents_indices]

# Step 6: Crossover (1-point crossover)
def crossover(parents, crossover_rate):
    population_size, num_genes = parents.shape
    offspring = np.copy(parents)
    
    for i in range(0, population_size, 2):
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, num_genes - 1)  # Random crossover point
            offspring[i, crossover_point:], offspring[i+1, crossover_point:] = offspring[i+1, crossover_point:], offspring[i, crossover_point:]
    
    return offspring

# Step 7: Mutation
def mutation(offspring, mutation_rate):
    population_size, num_genes = offspring.shape
    for i in range(population_size):
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, num_genes - 1)
            offspring[i, mutation_point] = random.uniform(-5, 5)  # Apply random mutation
    return offspring

# Step 8: Gene Expression (Convert the genetic sequence to a real solution)
def gene_expression(offspring):
    return offspring  # In this basic example, genetic code is the solution itself

# Step 9: Main loop for GEA
def gene_expression_algorithm(population_size, num_genes, mutation_rate, crossover_rate, max_generations):
    # Initialize population
    population = initialize_population(population_size, num_genes)
    
    # Evaluate fitness
    fitness = evaluate_fitness(population)
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    # Iterate for a number of generations
    for generation in range(max_generations):
        # Selection
        selected_parents = selection(population, fitness)
        
        # Crossover
        offspring = crossover(selected_parents, crossover_rate)
        
        # Mutation
        mutated_offspring = mutation(offspring, mutation_rate)
        
        # Gene Expression (convert to real solutions)
        new_population = gene_expression(mutated_offspring)
        
        # Evaluate fitness of new population
        fitness = evaluate_fitness(new_population)
        
        # Track the best solution
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = new_population[np.argmin(fitness)]
        
        # Update population for next generation
        population = new_population

    return best_solution, best_fitness

# Run the GEA
population_size = 50
num_genes = 5
mutation_rate = 0.1
crossover_rate = 0.7
max_generations = 100

best_solution, best_fitness = gene_expression_algorithm(population_size, num_genes, mutation_rate, crossover_rate, max_generations)

# Output the best solution and its fitness
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

