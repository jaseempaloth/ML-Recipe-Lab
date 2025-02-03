import numpy as np
from gradient_descent import gradient_descent

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Initialize the LinearRegressionGD model.
        
        Parameters:
            learning_rate (float): The learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = None

    def fit(self, X, y):
        """
        Fit the model to the data using gradient descent.
        
        Parameters:
            X (ndarray): Feature matrix of shape (m, n). If you don't include a bias (intercept) term,
                         one will be added automatically.
            y (ndarray): Target values of shape (m, ).
        """
        # Ensure X and y are numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Check if the first column is already a bias term (all ones).
        # If not, add a column of ones to X.
        if not np.all(X[:, 0] == 1):
            X = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Initialize theta
        self.theta = np.zeros(X.shape[1])

        # Perform gradient descent
        self.theta, self.cost_history = gradient_descent(
            X, y, self.theta, self.learning_rate, self.iterations
        )

    def predict(self, X):
        """
        Predict using the learned linear model.
        
        Parameters:
            X (ndarray): Feature matrix of shape (m, n). If the bias term is not included,
                         one will be added automatically.
        
        Returns:
            ndarray: Predicted values.
        """
        X = np.array(X)
        # Add bias term if necessary
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if not np.all(X[:, 0] == 1):
            X = np.c_[np.ones((X.shape[0], 1)), X]
        return X.dot(self.theta)
