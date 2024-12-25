import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Target labels
num_features = X.shape[1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fitness function: Evaluate accuracy with selected features
def evaluate_fitness(features):
    selected_features = [i for i in range(num_features) if features[i] == 1]
    if len(selected_features) == 0:  # No features selected
        return 0
    X_train_fs = X_train[:, selected_features]
    X_test_fs = X_test[:, selected_features]
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train_fs, y_train)
    y_pred = model.predict(X_test_fs)
    return accuracy_score(y_test, y_pred)

# Cuckoo Search parameters
num_nests = 15  # Number of nests
num_iterations = 50  # Maximum iterations
pa = 0.25  # Probability of abandoning nests
alpha = 0.01  # Lévy flight scale

# Initialize nests (binary arrays representing selected features)
nests = [np.random.randint(2, size=num_features) for _ in range(num_nests)]
fitness = [evaluate_fitness(nest) for nest in nests]

# Lévy flight implementation
def levy_flight(lam):
    sigma = (np.math.gamma(1 + lam) * np.sin(np.pi * lam / 2)) / \
            (np.math.gamma((1 + lam) / 2) * lam * 2**((lam - 1) / 2)) ** (1 / lam)
    u = np.random.normal(0, sigma, size=num_features)
    v = np.random.normal(0, 1, size=num_features)
    step = u / np.abs(v)**(1 / lam)
    return step

# Main loop
for iteration in range(num_iterations):
    for i in range(num_nests):
        # Generate a new solution
        new_nest = nests[i] + alpha * levy_flight(1.5)
        new_nest = np.clip(new_nest, 0, 1)  # Keep within [0, 1]
        new_nest = np.random.randint(2, size=num_features)  # Convert to binary
        new_fitness = evaluate_fitness(new_nest)
        
        # Replace if the new solution is better
        if new_fitness > fitness[i]:
            nests[i] = new_nest
            fitness[i] = new_fitness

    # Abandon worst nests
    num_abandon = int(pa * num_nests)
    worst_indices = np.argsort(fitness)[:num_abandon]
    for idx in worst_indices:
        nests[idx] = np.random.randint(2, size=num_features)
        fitness[idx] = evaluate_fitness(nests[idx])

# Get the best solution
best_index = np.argmax(fitness)
best_features = nests[best_index]
best_accuracy = fitness[best_index]

# Output results
selected_features = [i for i in range(num_features) if best_features[i] == 1]
print("Selected Features:", selected_features)
print("Best Accuracy:", best_accuracy)

