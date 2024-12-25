import random

# Define the fitness function for job scheduling
def fitness_function(schedule, processing_times):
    """
    The fitness function calculates the total makespan of the given schedule.
    :param schedule: List representing the order of jobs.
    :param processing_times: List representing the processing times of jobs.
    :return: Total makespan (sum of processing times).
    """
    makespan = sum(processing_times[job] for job in schedule)
    return makespan

# Initialize parameters
num_particles = 10  # Number of particles in the swarm
num_jobs = 5  # Number of jobs to schedule
processing_times = [4, 2, 7, 1, 3]  # Processing times of each job
max_iterations = 50  # Maximum number of iterations

# Initialize swarm
swarm = []
for _ in range(num_particles):
    particle = {
        "position": random.sample(range(num_jobs), num_jobs),  # Random job sequence
        "velocity": [],
        "pbest": None,
        "pbest_fitness": float("inf"),
    }
    swarm.append(particle)

gbest = None
gbest_fitness = float("inf")

# Main PSO loop
for iteration in range(max_iterations):
    for particle in swarm:
        # Calculate fitness of the particle's position
        current_fitness = fitness_function(particle["position"], processing_times)

        # Update personal best (pbest)
        if current_fitness < particle["pbest_fitness"]:
            particle["pbest"] = particle["position"][:]
            particle["pbest_fitness"] = current_fitness

        # Update global best (gbest)
        if current_fitness < gbest_fitness:
            gbest = particle["position"][:]
            gbest_fitness = current_fitness

    # Update velocity and position of each particle
    for particle in swarm:
        new_velocity = []

        # Generate velocity based on gbest and pbest
        for i in range(num_jobs):
            if random.random() < 0.5:  # With some probability, follow gbest
                if gbest[i] not in particle["position"]:
                    new_velocity.append(gbest[i])
            elif random.random() < 0.5:  # With some probability, follow pbest
                if particle["pbest"][i] not in particle["position"]:
                    new_velocity.append(particle["pbest"][i])

        # Update particle velocity and position
        particle["velocity"] = new_velocity
        particle["position"] = list(particle["position"][:num_jobs - len(new_velocity)]) + new_velocity

# Print results
print("Best schedule found:", gbest)
print("Makespan:", gbest_fitness)
