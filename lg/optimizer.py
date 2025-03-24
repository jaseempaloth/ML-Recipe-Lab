from jax import jit
import jax.numpy as jnp

class GradientDescent:
    def __init__(self, learning_rate=0.01):
        """
        Initialize the Gradient Descent optimizer.
        
        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate

    @staticmethod
    @jit
    def update(params, grads, learning_rate):
        """
        Update the model parameters using gradient descent.
        
        Args:
            params (dict): Model parameters.
            grads (dict): Gradients of the loss with respect to the parameters.
            learning_rate (float): Learning rate for the optimizer.
            
        Returns:
            dict: Updated model parameters.
        """
        return { k: p - learning_rate * g for k, (p, g) in 
                zip(params.keys(), zip(params.values(), grads.values()))}       
