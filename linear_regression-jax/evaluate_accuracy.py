import numpy as np
import jax.numpy as jnp
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from model import LinearRegression

def evaluate_model(X_train, X_test, y_train, y_test, model_name="Model"):
    """Evaluate model performance using multiple metrics."""
    
    # Initialize and train model
    model = LinearRegression(n_features=X_train.shape[1])
    print(f"\nTraining {model_name}...")
    history = model.fit(X_train, y_train, X_val=X_test, y_val=y_test, 
                        learning_rate=0.01, n_iterations=300)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Convert to numpy arrays if needed
    if hasattr(y_pred, 'device_buffer'):
        y_pred = np.array(y_pred)
    if hasattr(y_test, 'device_buffer'):
        y_test = np.array(y_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print(f"\n===== {model_name} Performance =====")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot history and results
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training history
    axs[0].plot(history)
    axs[0].set_title(f'{model_name} - Validation Loss Over Training')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('MSE Loss')
    axs[0].grid(True)
    
    # Predictions vs actual
    axs[1].scatter(y_test, y_pred, alpha=0.5)
    axs[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    axs[1].set_xlabel('Actual')
    axs[1].set_ylabel('Predicted')
    axs[1].set_title(f'{model_name} - Predicted vs Actual (R² = {r2:.4f})')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'history': history
    }

if __name__ == "__main__":
    # Test on synthetic data
    print("Evaluating on synthetic data...")
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    synthetic_results = evaluate_model(X_train, X_test, y_train, y_test, "Synthetic Data Model")
    
    # Test on California Housing dataset
    print("\nEvaluating on California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Preprocess: standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    california_results = evaluate_model(X_train, X_test, y_train, y_test, "California Housing Model")
    
    # Compare models
    print("\n===== Model Comparison =====")
    print(f"Synthetic Data R²: {synthetic_results['r2']:.4f}")
    print(f"California Housing R²: {california_results['r2']:.4f}")
