from jax import jit, vmap
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import time

from model import LinearRegression
from loss import mse_loss, mae_loss, rmse_loss
from optimizer import GradientDescent
from data import generate_data
from train import train
from utils import plot_history, r2_score

def main():
    # Generate data
    X_train, y_train, X_test, y_test = generate_data(
        n_samples=1000, n_features=5, noise=0.5)
    
    print(f"Data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Check dimension of y_train and y_test
    if len(y_train.shape) > 1:
        print(f"Warning: y_train has shape {y_train.shape}, expected 1D array")
        if y_train.shape[1] > 1:
            y_train = y_train[:, 0]  # Take first column if multi-dimensional
            print(f"Reshaped y_train to: {y_train.shape}")
    
    if len(y_test.shape) > 1:
        print(f"Warning: y_test has shape {y_test.shape}, expected 1D array")
        if y_test.shape[1] > 1:
            y_test = y_test[:, 0]  # Take first column if multi-dimensional
            print(f"Reshaped y_test to: {y_test.shape}")
    
    # Initialize model
    model = LinearRegression(n_features=X_train.shape[1])
    
    # Train model with performance measurement
    start_time = time.time()
    history = train(
        model=model,
        loss_fn=mse_loss,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=0.01,
        epochs=100,
        batch_size=32,
        verbose=True
    )
    train_time = time.time() - start_time
    
    # Create vectorized prediction function
    predicts = model.forward(model.params, X_test)
    r2 = r2_score(y_test, predicts)
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Test R² score: {float(r2):.4f}")
    
    # Plot history
    plot_history(history)
    
    # Print final parameters
    print("\nFinal parameters:")
    print(f"Weights: {model.params['w']}")
    if 'b' in model.params:
        print(f"Bias: {model.params['b']}")
    
    # Compare prediction vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predicts, alpha=0.5)
    plt.plot([float(y_test.min()), float(y_test.max())], 
             [float(y_test.min()), float(y_test.max())], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Prediction vs Actual')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
