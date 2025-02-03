import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []

    def sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """Compute the cost using binary cross-entropy loss."""
        m = len(y)
        predictions = self.sigmoid(X.dot(self.theta))
        # Adding a small epsilon to avoid log(0) errors
        epsilon = 1e-5
        cost = -(1/m) * np.sum(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
        return cost

    def compute_gradient(self, X, y):
        """Compute the gradient of the cost function."""
        m = len(y)
        predictions = self.sigmoid(X.dot(self.theta))
        gradient = (1/m) * X.T.dot(predictions - y)
        return gradient

    def fit(self, X, y):
        """Fit the model to the data using gradient descent."""
        m, n = X.shape
        self.theta = np.zeros(n)
        self.cost_history = []
        
        for i in range(self.iterations):
            grad = self.compute_gradient(X, y)
            self.theta -= self.learning_rate * grad
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
        
        return self

    def predict_proba(self, X):
        """Predict probability estimates for input data."""
        return self.sigmoid(X.dot(self.theta))

    def predict(self, X, threshold=0.5):
        """Predict binary labels for input data."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
