from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from model import LinearRegression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=100, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = LinearRegression(n_features=X_train.shape[1])

# Train the model
model.fit(X_train, y_train, X_val=X_test, y_val=y_test, learning_rate=0.01, n_iterations=1000)

# Final evaluation
final_loss = model.score(X_test, y_test)
print(f"Final test loss: {final_loss}")
