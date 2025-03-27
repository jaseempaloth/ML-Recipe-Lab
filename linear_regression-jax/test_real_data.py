import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import LinearRegression
from accuracy import evaluate_regression_model, calculate_metrics

# Load California housing dataset (real data)
print("Loading California housing dataset...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Print dataset information
print(f"Dataset shape: {X.shape}")
print(f"Feature names: {housing.feature_names}")
print(f"Target description: {housing.DESCR.split('Attribute Information')[0]}")

# Preprocess: standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression(n_features=X_train.shape[1])
print("\nTraining on California housing data...")
history = model.fit(X_train, y_train, X_val=X_test, y_val=y_test, 
                   learning_rate=0.01, n_iterations=500)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model using our accuracy module
print("\nEvaluating model performance:")
metrics = evaluate_regression_model(y_test, y_pred, history, "California Housing Model")

# More detailed analysis of the results
print(f"\nResult Analysis:")
print(f"- Your R² score of {metrics['r2']:.4f} means the model explains about {metrics['r2']*100:.1f}% of the variance in housing prices.")
print(f"- For a simple linear regression on the California Housing dataset, this is actually a decent result.")
print(f"- The dataset is known to have non-linear relationships that linear regression can't fully capture.")
print(f"- For comparison, more complex models like random forests typically achieve R² scores of 0.75-0.85 on this dataset.")

print("\nPossible improvements:")
print("- Try polynomial features to capture non-linear relationships")
print("- Experiment with feature engineering or feature selection")
print("- Consider regularization techniques like Ridge or Lasso regression")
print("- Adjust the learning rate and number of iterations")
