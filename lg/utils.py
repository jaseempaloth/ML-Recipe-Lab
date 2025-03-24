from jax import jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt

def plot_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot time per epoch
    plt.subplot(1, 2, 2)
    plt.plot(history['time_per_epoch'])
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.title('Time per Epoch')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

@jit
def r2_score(y_true, y_pred):
    """Calculate R² score (coefficient of determination)."""
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / ss_tot

# Vectorized version for batch processing
batch_r2_score = vmap(r2_score, in_axes=(0, 0))
