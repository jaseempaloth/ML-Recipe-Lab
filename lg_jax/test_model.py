import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time

# Add JAX imports to check device information
import jax

# Print JAX device information
print("=" * 40)
print("JAX Device Information:")
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")
print(f"Device count: {jax.device_count()}")
print("=" * 40)

# Import our JAX-based LinearRegression implementation
from model import LinearRegression

# Generate synthetic data with known coefficients
np.random.seed(42)
n_samples, n_features = 1000, 5

# Generate feature matrix with correlated features
X = np.random.randn(n_samples, n_features)
# Add correlation between features
X[:, 1] = X[:, 0] * 0.5 + X[:, 1] * 0.5
X[:, 3] = X[:, 2] * 0.7 + X[:, 3] * 0.3

# True coefficients
true_coef = np.array([3.5, -2.0, 1.5, 0.5, -1.0])
true_intercept = 2.0

# Generate target
y = np.dot(X, true_coef) + true_intercept + np.random.normal(0, 1, size=n_samples)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and compare model variants
models = {
    'JAX-GD': LinearRegression(solver='gd', max_iter=1000, learning_rate=0.01),
    'JAX-Closed': LinearRegression(solver='closed'),
    'JAX-GD-Regularized': LinearRegression(solver='gd', alpha=0.1, max_iter=1000, learning_rate=0.01),
    'JAX-Closed-Regularized': LinearRegression(solver='closed', alpha=0.1)
}

results = {}

print("Model Comparison:")
print("-" * 80)
print(f"{'Model':<25} {'Fit Time (s)':<15} {'MSE':<15} {'R²':<15} {'Intercept':<10}")
print("-" * 80)

for name, model in models.items():
    # Time the fitting process
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    fit_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'coef': model.coef_,
        'intercept': model.intercept_,
        'mse': mse,
        'r2': r2,
        'fit_time': fit_time
    }
    
    print(f"{name:<25} {fit_time:<15.4f} {mse:<15.4f} {r2:<15.4f} {model.intercept_:<10.4f}")

# Plot the coefficients
plt.figure(figsize=(10, 6))
x = np.arange(len(true_coef))
width = 0.15

plt.bar(x - 0.3, true_coef, width, label='True Coef', color='black')
plt.bar(x - 0.15, results['JAX-GD']['coef'], width, label='JAX-GD', alpha=0.7)
plt.bar(x, results['JAX-Closed']['coef'], width, label='JAX-Closed', alpha=0.7)
plt.bar(x + 0.15, results['JAX-GD-Regularized']['coef'], width, label='JAX-GD-Reg', alpha=0.7)
plt.bar(x + 0.3, results['JAX-Closed-Regularized']['coef'], width, label='JAX-Closed-Reg', alpha=0.7)

plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Comparison')
plt.xticks(x, [f'X{i}' for i in range(len(true_coef))])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('coefficient_comparison.png')
plt.show()

# Plot training time
plt.figure(figsize=(8, 5))
models_names = list(results.keys())
fit_times = [results[name]['fit_time'] for name in models_names]

plt.barh(models_names, fit_times, color='skyblue')
plt.xlabel('Fit Time (seconds)')
plt.title('Model Fitting Time Comparison')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('fit_time_comparison.png')
plt.show()

# Plot performance metrics
plt.figure(figsize=(10, 5))
x = np.arange(len(models_names))
width = 0.35

plt.bar(x - width/2, [results[name]['mse'] for name in models_names], width, label='MSE', color='salmon')
plt.bar(x + width/2, [1 - results[name]['r2'] for name in models_names], width, label='1 - R²', color='lightgreen')

plt.xlabel('Model')
plt.ylabel('Error Metric')
plt.title('Model Performance Comparison (lower is better)')
plt.xticks(x, models_names, rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()

print("\nCoefficient details:")
print("-" * 80)
print(f"{'Feature':<10} {'True':<10}", end="")
for name in models_names:
    print(f" {name:<15}", end="")
print("\n" + "-" * 80)

for i in range(len(true_coef)):
    print(f"X{i:<9} {true_coef[i]:<10.4f}", end="")
    for name in models_names:
        print(f" {results[name]['coef'][i]:<15.4f}", end="")
    print()

print("-" * 80)
print(f"{'Intercept':<10} {true_intercept:<10.4f}", end="")
for name in models_names:
    print(f" {results[name]['intercept']:<15.4f}", end="")
print()