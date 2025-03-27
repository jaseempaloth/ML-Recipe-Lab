import jax.numpy as jnp
from jax import grad, jit
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    # Define the mean squared error loss function
    @staticmethod
    def loss(params, X, y):
        w, b = params
        y_pred = jnp.dot(X, w) + b
        return jnp.mean((y_pred - y) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = jnp.zeros(n_features)
        self.bias = 0.0
        params = (self.weights, self.bias)

        # JIT-compile the gradient of the loss function
        grad_loss = jit(grad(self.loss))

        # Gradient descent loop
        for _ in range(self.n_iterations):
            w_grad, b_grad = grad_loss(params, X, y)
            params = (
                params[0] - self.learning_rate * w_grad,
                params[1] - self.learning_rate * b_grad
            )

        self.weights, self.bias = params

    def predict(self, X):
        return jnp.dot(X, self.weights) + self.bias

def mse(y_test, y_pred):
    return jnp.mean((y_test - y_pred) ** 2)

def main():
    # Generate a regression dataset using sklearn's make_regression
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert data to JAX arrays
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    X_test = jnp.array(X_test)
    y_test = jnp.array(y_test)

    # Create and train the linear regression model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    mse_loss = mse(y_test, predictions)
    print(f"Mean Squared Error: {mse_loss:.4f}")
    
    y_pred_plot = model.predict(X)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(10, 6))
    
    plt.scatter(X_train, y_train, color=cmap(0.5), label='Train Data', alpha=0.5)
    plt.scatter(X_test, y_test, color=cmap(0.8), label='Test Data', alpha=0.5)
    plt.plot(X, y_pred_plot, color='red', label='Model Prediction')
    
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Linear Regression with JAX and Automatic Differentiation")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
