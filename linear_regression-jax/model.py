import jax 
import jax.numpy as jnp
from jax import grad, jit

class LinearRegression:
    def __init__(self, n_features):
        self.params = {
            'w': jnp.zeros(n_features),
            'b': 0.0
        }
        self.grad_fn = jax.grad(self._compute_loss)
        
    def _forward(self, X):
        return jnp.dot(X, self.params['w']) + self.params['b']
    
    def _compute_loss(self, params, X, y):
        predictions = jnp.dot(X, params['w']) + params['b']
        return jnp.mean((predictions - y) ** 2)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, learning_rate=0.01, n_iterations=100):
        history = []
        for i in range(n_iterations):
            if X_val is not None and y_val is not None:
                val_loss = self._compute_loss(self.params, X_val, y_val)
                history.append(val_loss)
                print(f"Iteration {i}, Validation Loss: {val_loss}")
            
            grads = self.grad_fn(self.params, X_train, y_train)
            self.params = jax.tree_map(
                lambda p, g: p - learning_rate * g, self.params, grads
                )
        
        return history

    def predict(self, X):
        return self._forward(X)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return jnp.mean((predictions - y) ** 2)  # MSE