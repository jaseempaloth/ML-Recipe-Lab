from jax import jit, vmap
import jax.numpy as jnp
from jax import random
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def generate_data(n_samples=100, n_features=1, noise=0.1, test_size=0.2, random_state=42):
    """
    Generate synthetic regression data with a bias term.
    
    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features.
        noise (float): Standard deviation of the Gaussian noise.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        
    Returns:
        tuple: Training and testing data (X_train, y_train, X_test, y_test).
    """
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=random_state)
    # Ensure y is 1D
    y = y.reshape(-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Verify shapes before continuing
    assert X_train.shape[0] == y_train.shape[0], f"Sample mismatch: X_train has {X_train.shape[0]} samples, y_train has {y_train.shape[0]}"
    
    # Add a bias term to the feature matrix
    X_train = jnp.concatenate([jnp.ones((X_train.shape[0], 1)), X_train], axis=1)
    X_test = jnp.concatenate([jnp.ones((X_test.shape[0], 1)), X_test], axis=1)

    # Convert to JAX arrays
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    return X_train, y_train, X_test, y_test


class DataLoader:
    def __init__(self, X, y, batch_size=32, shuffle=True):
        """"
        "Initialize the DataLoader.
        Args:
            X (jnp.ndarray): Input features
            y (jnp.ndarray): Target values
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the data
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = X.shape[0]
        self.n_batches = (self.n_samples - 1) // batch_size + 1
        self.key = random.PRNGKey(0)

    @staticmethod
    @jit
    def get_batch(X, y, indices):
        """Get batch using pre-computed indices."""
        return X[indices], y[indices]
    
    def __iter__(self):
        """Iterator for batches."""
        # Create shuffled indices
        indices = jnp.arange(self.n_samples)
        if self.shuffle:
            self.key, subkey = random.split(self.key)
            indices = random.permutation(subkey, indices)

        # Yield batches
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield self.get_batch(self.X, self.y, batch_indices)









