import jax 
import jax.numpy as jnp
from jax import random

class LinearRegression:
    def __init__(self, n_features):
        """
        Initialize the linear regression model.
        
        Args:
            n_features (int): Number of input features
        """
        self.n_features = n_features
        self.params = self._init_params()
    
    def _init_params(self):
        """Initialize model parameters."""
        key = random.PRNGKey(0)
        params = {
            'w': random.normal(key, (self.n_features,)) * 0.01,
            'b': 0.0
        }
        return params

    def forward(self, params, X):
        """
        Forward pass of the linear regression model.
        
        Args:
            params (dict): Model parameters
            X (jnp.ndarray): Input features
            
        Returns:
            jnp.ndarray: Model predictions
        """
        return jnp.dot(X, params['w']) + params['b']