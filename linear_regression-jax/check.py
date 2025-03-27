import jax
import jax.numpy as jnp
from jax import grad, jit, random
from typing import Dict, Tuple, Optional, Union, Callable
import time

class LinearRegression:
    def __init__(self, seed: int = 0):
        self.params = None
        self.key = random.PRNGKey(seed)
    
    def initialize(self, n_features: int) -> Dict:
        # Generate a fresh key for initialization
        self.key, subkey = random.split(self.key)
        # Initialize weights and bias with improved scaling
        self.params = {
            'weights': random.normal(subkey, (n_features,)) * 0.1,  # Scale for better convergence
            'bias': 0.0
        }
        return self.params
    
    @jit  # JIT compile for faster execution
    def forward(self, params: Dict, X: jnp.ndarray) -> jnp.ndarray:
        # Linear regression prediction
        return jnp.dot(X, params['weights']) + params['bias']
    
    @jit  # JIT compile loss function
    def compute_loss(self, params: Dict, X: jnp.ndarray, y: jnp.ndarray, 
                    l1_lambda: float = 0.0, l2_lambda: float = 0.0) -> float:
        # Compute the loss with optional regularization
        predictions = self.forward(params, X)
        mse_loss = jnp.mean(jnp.square(predictions - y))
        
        # Add regularization if specified
        if l1_lambda > 0:
            l1_reg = l1_lambda * jnp.sum(jnp.abs(params['weights']))
            mse_loss += l1_reg
        
        if l2_lambda > 0:
            l2_reg = l2_lambda * jnp.sum(jnp.square(params['weights']))
            mse_loss += l2_reg
            
        return mse_loss
    
    @jit  # JIT compile the update step
    def _update_params(self, params: Dict, grads: Dict, learning_rate: float) -> Dict:
        # Vectorized parameter updates
        return {
            'weights': params['weights'] - learning_rate * grads['weights'],
            'bias': params['bias'] - learning_rate * grads['bias']
        }
    
    def fit(self, X: jnp.ndarray, y: jnp.ndarray, learning_rate: float = 0.01, 
           epochs: int = 100, batch_size: Optional[int] = None, 
           early_stopping: bool = False, patience: int = 10, 
           l1_lambda: float = 0.0, l2_lambda: float = 0.0,
           verbose: bool = True) -> Dict:
        # Fit the model with optimized training loop
        if self.params is None:
            self.params = self.initialize(X.shape[1])
        
        # Create jitted gradient function
        grad_fn = jit(grad(lambda p, x, y: self.compute_loss(p, x, y, l1_lambda, l2_lambda)))
        
        n_samples = X.shape[0]
        best_loss = float('inf')
        patience_counter = 0
        start_time = time.time()
        
        # Prepare for mini-batch training if batch_size is specified
        if batch_size is None:
            batch_size = n_samples
        
        for epoch in range(epochs):
            # Shuffle data for stochastic gradient descent
            if batch_size < n_samples:
                self.key, subkey = random.split(self.key)
                indices = random.permutation(subkey, n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled, y_shuffled = X, y
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:min(i + batch_size, n_samples)]
                y_batch = y_shuffled[i:min(i + batch_size, n_samples)]
                
                # Compute gradients
                grads = grad_fn(self.params, X_batch, y_batch)
                
                # Update parameters
                self.params = self._update_params(self.params, grads, learning_rate)
            
            # Compute loss for monitoring
            if verbose and epoch % 10 == 0:
                current_loss = self.compute_loss(self.params, X, y, l1_lambda, l2_lambda)
                elapsed_time = time.time() - start_time
                print(f'Epoch {epoch}, Loss: {current_loss:.6f}, Time: {elapsed_time:.2f}s')
            
            # Early stopping logic
            if early_stopping:
                current_loss = self.compute_loss(self.params, X, y, l1_lambda, l2_lambda)
                if current_loss < best_loss - 1e-4:  # Improvement threshold
                    best_loss = current_loss
                    patience_counter = 0
                    best_params = self.params.copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    self.params = best_params  # Restore best parameters
                    break
        
        if verbose:
            final_loss = self.compute_loss(self.params, X, y, l1_lambda, l2_lambda)
            total_time = time.time() - start_time
            print(f'Training completed - Final Loss: {final_loss:.6f}, Total time: {total_time:.2f}s')
        
        return self.params
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        # Make predictions (using compiled forward function)
        return self.forward(self.params, X)



