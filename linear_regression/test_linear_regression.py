import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression


# Generate a Synthetic Dataset

np.random.seed(42)  # For reproducibility

# Number of samples
m = 1000

# Generate random feature values and create a linear relationship: y = 4 + 3*x + noise
X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X[:, 0] + np.random.randn(m)


# Initialize and Fit the Model

model = LinearRegression(learning_rate=0.1, iterations=1000)
model.fit(X, y)

# Output the learned parameters (theta)
print("Learned parameters (theta):", model.theta)


# Plot the Cost History

plt.plot(range(len(model.cost_history)), model.cost_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History over Iterations')
plt.show()


