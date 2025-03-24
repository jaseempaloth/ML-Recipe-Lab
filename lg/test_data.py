import jax.numpy as jnp
from jax import random
from data import DataLoader  # Import DataLoader from your file (adjust the path if needed)

# Generate some dummy data
key = random.PRNGKey(42)
X = random.normal(key, (100, 5))  # 100 samples, 5 features
y = random.normal(key, (100,))    # 100 targets

# Create DataLoader
batch_size = 10
dataloader = DataLoader(X, y, batch_size=batch_size, shuffle=True)

# Test DataLoader
print(f"Total Samples: {X.shape[0]}, Batch Size: {batch_size}")

for i, (X_batch, y_batch) in enumerate(dataloader):
    print(f"Batch {i + 1}: X shape = {X_batch.shape}, y shape = {y_batch.shape}")

# Optional: Check number of batches
n_batches = (X.shape[0] - 1) // batch_size + 1
assert i + 1 == n_batches, f"Expected {n_batches} batches but got {i + 1}"
print("Test Passed!")