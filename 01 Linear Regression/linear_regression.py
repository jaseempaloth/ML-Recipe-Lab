import jax.numpy as jnp
from jax import grad, jit

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = jnp.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = jnp.dot(X, self.weights) + self.bias

            dw = jnp.dot(X.T, (y_pred - y)) / n_samples
            db = jnp.sum(y_pred - y) / n_samples
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return jnp.dot(X, self.weights) + self.bias