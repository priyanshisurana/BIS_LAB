import numpy as np

# Objective function (example: Sphere function)
def objective_function(x):
    return np.sum(x**2)

# Grey Wolf Optimizer
class GreyWolfOptimizer:
    def __init__(self, obj_function, dim, bounds, num_wolves=30, max_iter=100):
        self.obj_function = obj_function
        self.dim = dim  # Dimension of the solution
        self.bounds = bounds  # Bounds of the search space
        self.num_wolves = num_wolves  # Population size
        self.max_iter = max_iter  # Maximum iterations
        
        # Initialize positions of wolves
        self.positions = np.random.uniform(
            low=bounds[0], high=bounds[1], size=(num_wolves, dim)
        )
        self.fitness = np.apply_along_axis(self.obj_function, 1, self.positions)
        
        # Initialize alpha, beta, delta wolves
        self.update_alpha_beta_delta()
    
    def update_alpha_beta_delta(self):
        sorted_indices = np.argsort(self.fitness)
        self.alpha = self.positions[sorted_indices[0]]
        self.beta = self.positions[sorted_indices[1]]
        self.delta = self.positions[sorted_indices[2]]
        self.alpha_fitness = self.fitness[sorted_indices[0]]
    
    def update_positions(self, a):
        for i in range(self.num_wolves):
            # Update based on alpha, beta, and delta
            for leader, position in zip([self.alpha, self.beta, self.delta], [self.alpha, self.beta, self.delta]):
                r1, r2 = np.random.random(size=(self.dim,)), np.random.random(size=(self.dim,))
                A = 2 * a * r1 - a
                C = 2 * r2
                D = abs(C * leader - self.positions[i])
                self.positions[i] = position - A * D
            
            # Clip position to bounds
            self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])
    
    def optimize(self):
        for iter in range(self.max_iter):
            # Decrease parameter a
            a = 2 - (2 * iter / self.max_iter)
            
            # Update positions
            self.update_positions(a)
            
            # Recalculate fitness
            self.fitness = np.apply_along_axis(self.obj_function, 1, self.positions)
            
            # Update alpha, beta, delta
            self.update_alpha_beta_delta()
        
        # Return the best solution
        return self.alpha, self.alpha_fitness

# Example usage
if __name__ == "__main__":
    # Problem setup
    dimension = 5  # Number of variables
    bounds = (-10, 10)  # Search space bounds
    max_iterations = 100
    num_wolves = 30
    
    gwo = GreyWolfOptimizer(
        obj_function=objective_function,
        dim=dimension,
        bounds=bounds,
        num_wolves=num_wolves,
        max_iter=max_iterations
    )
    
    best_position, best_fitness = gwo.optimize()
    print("Best Position:", best_position)
    print("Best Fitness:", best_fitness)



































