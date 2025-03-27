import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from model import LinearRegression
import time

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12})

# Display JAX information
print("\n=== JAX Configuration ===")
import jax
print(f"JAX Backend: {jax.default_backend()}")
print(f"Available Devices: {jax.devices()}")

# Load a dataset from sklearn
print("\n=== Loading Dataset ===")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Standardize features (important for regularized models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to jax arrays
X_train_jax = jnp.array(X_train)
y_train_jax = jnp.array(y_train)
X_test_jax = jnp.array(X_test)
y_test_jax = jnp.array(y_test)

print(f"Dataset info:")
print(f"  - Features: {X_train.shape[1]}")
print(f"  - Training samples: {X_train.shape[0]}")
print(f"  - Test samples: {X_test.shape[0]}")

# Define model comparison configurations
configs = [
    {
        "name": "Basic Model",
        "learning_rate": 0.01,
        "epochs": 300,
        "batch_size": None,  # Use full batch
        "l1_lambda": 0.0,
        "l2_lambda": 0.0,
        "early_stopping": False
    },
    {
        "name": "Mini-Batch Training",
        "learning_rate": 0.01,
        "epochs": 300,
        "batch_size": 32,
        "l1_lambda": 0.0,
        "l2_lambda": 0.0,
        "early_stopping": False
    },
    {
        "name": "L2 Regularization",
        "learning_rate": 0.01,
        "epochs": 300,
        "batch_size": 32,
        "l1_lambda": 0.0,
        "l2_lambda": 0.01,
        "early_stopping": False
    },
    {
        "name": "L1 Regularization",
        "learning_rate": 0.01,
        "epochs": 300,
        "batch_size": 32,
        "l1_lambda": 0.01,
        "l2_lambda": 0.0,
        "early_stopping": False
    },
    {
        "name": "Early Stopping",
        "learning_rate": 0.01,
        "epochs": 500,
        "batch_size": 32,
        "l1_lambda": 0.0,
        "l2_lambda": 0.0,
        "early_stopping": True,
        "patience": 20
    }
]

results = {}

# Train and evaluate each model configuration
for config in configs:
    print(f"\n=== Training: {config['name']} ===")
    model_name = config["name"]
    
    # Create and train model with specified configuration
    start_time = time.time()
    model = LinearRegression(seed=42)
    model.initialize(X_train.shape[1])
    
    # Extract early stopping parameters if present
    early_stopping_args = {}
    if "early_stopping" in config:
        early_stopping_args["early_stopping"] = config["early_stopping"]
    if "patience" in config:
        early_stopping_args["patience"] = config["patience"]
    
    # Train the model
    model.fit(
        X_train_jax, 
        y_train_jax, 
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        l1_lambda=config["l1_lambda"],
        l2_lambda=config["l2_lambda"],
        **early_stopping_args
    )
    
    # Calculate training time
    train_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test_jax)
    y_pred_np = np.array(y_pred)  # Convert to numpy for sklearn metrics
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred_np)
    r2 = r2_score(y_test, y_pred_np)
    
    # Store results
    results[model_name] = {
        "model": model,
        "params": model.params,
        "mse": mse,
        "r2": r2,
        "time": train_time
    }
    
    print(f"  - Training time: {train_time:.2f} seconds")
    print(f"  - Test MSE: {mse:.2f}")
    print(f"  - Test R²: {r2:.4f}")

# Compare model performance
print("\n=== Model Performance Comparison ===")
print(f"{'Model':<20} {'MSE':<10} {'R²':<10} {'Time (s)':<10}")
print("-" * 50)
for name, result in results.items():
    print(f"{name:<20} {result['mse']:<10.2f} {result['r2']:<10.4f} {result['time']:<10.2f}")

# Plot model comparison results
plt.figure(figsize=(14, 10))

# Plot 1: MSE comparison
plt.subplot(2, 2, 1)
model_names = list(results.keys())
mse_values = [results[name]["mse"] for name in model_names]
plt.bar(model_names, mse_values, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Mean Squared Error (lower is better)')
plt.tight_layout()

# Plot 2: R² comparison
plt.subplot(2, 2, 2)
r2_values = [results[name]["r2"] for name in model_names]
plt.bar(model_names, r2_values, color='lightgreen')
plt.xticks(rotation=45, ha='right')
plt.title('R² Score (higher is better)')
plt.tight_layout()

# Plot 3: Training time comparison
plt.subplot(2, 2, 3)
time_values = [results[name]["time"] for name in model_names]
plt.bar(model_names, time_values, color='salmon')
plt.xticks(rotation=45, ha='right')
plt.title('Training Time (seconds)')
plt.tight_layout()

# Plot 4: Actual vs Predicted for best model
plt.subplot(2, 2, 4)
best_model_name = model_names[np.argmax(r2_values)]
best_model = results[best_model_name]["model"]
y_pred_best = np.array(best_model.predict(X_test_jax))

plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Actual vs Predicted ({best_model_name})')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.tight_layout()

plt.suptitle('Model Performance Comparison', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Create sample predictions
print("\n=== Sample Predictions from Best Model ===")
best_model = results[best_model_name]["model"]
sample_indices = np.random.choice(len(y_test), size=5, replace=False)

print(f"Using best model: {best_model_name}")
print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
print("-" * 30)
for idx in sample_indices:
    actual = y_test[idx]
    predicted = float(best_model.predict(X_test_jax[idx].reshape(1, -1)))
    error = actual - predicted
    print(f"{actual:<10.2f} {predicted:<10.2f} {error:<10.2f}")
