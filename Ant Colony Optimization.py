import random
import numpy as np
import matplotlib.pyplot as plt

# Parameters for ACO
num_ants = 50
alpha = 1.0  # Pheromone importance
beta = 2.0   # Distance priority
rho = 0.1    # Evaporation rate
q = 100      # Pheromone intensity
iterations = 1000

# Initialize cities (cities as random points in a 2D plane)
num_cities = 10
cities = np.random.rand(num_cities, 2)

# Calculate the distance between two points
def euclidean_distance(city1, city2):
    return np.sqrt(np.sum((city1 - city2) ** 2))

# Create distance matrix
distances = np.array([[euclidean_distance(cities[i], cities[j]) for j in range(num_cities)] for i in range(num_cities)])

# Initialize pheromone matrix
pheromone = np.ones((num_cities, num_cities)) / num_cities

# Ant Colony Optimization main loop
def aco(pheromone, distances):
    best_route = None
    best_distance = float('inf')

    for iteration in range(iterations):
        all_routes = []
        all_distances = []

        # Step 1: Each ant constructs a solution
        for _ in range(num_ants):
            route = [random.randint(0, num_cities - 1)]  # Start at a random city
            visited = set(route)
            for _ in range(num_cities - 1):
                current_city = route[-1]
                probabilities = []
                for next_city in range(num_cities):
                    if next_city not in visited:
                        pheromone_val = pheromone[current_city][next_city] ** alpha
                        distance_val = (1.0 / distances[current_city][next_city]) ** beta
                        probabilities.append(pheromone_val * distance_val)
                    else:
                        probabilities.append(0)
                # Normalize the probabilities
                total = sum(probabilities)
                if total > 0:
                    probabilities = [p / total for p in probabilities]
                else:
                    probabilities = [1.0 / num_cities] * num_cities
                # Select the next city based on probabilities
                next_city = np.random.choice(range(num_cities), p=probabilities)
                route.append(next_city)
                visited.add(next_city)
            all_routes.append(route)
            # Calculate route distance
            route_distance = sum(distances[route[i]][route[i + 1]] for i in range(num_cities - 1))
            route_distance += distances[route[-1]][route[0]]  # Return to the start city
            all_distances.append(route_distance)

            # Step 2: Update the best solution
            if route_distance < best_distance:
                best_route = route
                best_distance = route_distance

        # Step 3: Update pheromone matrix
        pheromone *= (1 - rho)  # Evaporation
        for ant_idx, route in enumerate(all_routes):
            pheromone_contribution = q / all_distances[ant_idx]
            for i in range(num_cities - 1):
                pheromone[route[i]][route[i + 1]] += pheromone_contribution
            pheromone[route[-1]][route[0]] += pheromone_contribution

    return best_route, best_distance

# Run ACO to solve TSP
best_route, best_distance = aco(pheromone, distances)

# Print and visualize the result
print(f"Best route: {best_route}")
print(f"Best distance: {best_distance}")

# Plot the best route
best_cities = cities[best_route]
best_cities = np.vstack((best_cities, best_cities[0]))  # Connect back to the start city
plt.figure(figsize=(8, 8))
plt.plot(best_cities[:, 0], best_cities[:, 1], 'o-', markerfacecolor='red')
plt.title(f"Best TSP Route (Distance: {best_distance:.2f})")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
