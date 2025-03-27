import jax.numpy as jnp
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fig = plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, alpha=0.5)
plt.show()

# Initialize model
model = LinearRegression(learning_rate=0.05, n_iterations=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

def mse_loss(y_test, y_pred):
    return jnp.mean((y_test - y_pred) ** 2)

mse = mse_loss(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

y_pred_plot = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(10, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.5), label='Train Data', alpha=0.5)
m2 = plt.scatter(X_test, y_test, color=cmap(0.8), label='Test Data', alpha=0.5)
plt.plot(X, y_pred_plot, color='red', label='Model Prediction')
plt.show()
