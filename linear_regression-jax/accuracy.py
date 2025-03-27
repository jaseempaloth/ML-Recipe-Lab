import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Parameters:
        y_true: Ground truth target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing metrics
    """
    # Convert JAX arrays to numpy if needed
    if hasattr(y_true, 'device_buffer'):
        y_true = np.array(y_true)
    if hasattr(y_pred, 'device_buffer'):
        y_pred = np.array(y_pred)
        
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse, 
        'mae': mae,
        'r2': r2
    }

def print_metrics(metrics, model_name="Model"):
    """Print formatted metrics."""
    print(f"\n===== {model_name} Performance =====")
    print(f"Mean Squared Error (MSE): {metrics['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")

def plot_history(history, title="Validation Loss Over Training"):
    """Plot training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.show()

def plot_predictions(y_true, y_pred, title="Predicted vs Actual Values"):
    """Plot predictions against actual values."""
    # Convert JAX arrays to numpy if needed
    if hasattr(y_true, 'device_buffer'):
        y_true = np.array(y_true)
    if hasattr(y_pred, 'device_buffer'):
        y_pred = np.array(y_pred)
        
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f"{title} (R² = {r2_score(y_true, y_pred):.4f})")
    plt.grid(True)
    plt.show()

def evaluate_regression_model(y_true, y_pred, history=None, model_name="Model"):
    """
    Comprehensive evaluation of regression model with metrics and plots.
    
    Parameters:
        y_true: Ground truth target values
        y_pred: Predicted target values
        history: Optional training history for loss curve
        model_name: Name of the model for display
    """
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics, model_name)
    
    if history is not None:
        plot_history(history, f"{model_name} - Training History")
    
    plot_predictions(y_true, y_pred, f"{model_name} - Predicted vs Actual")
    
    return metrics
