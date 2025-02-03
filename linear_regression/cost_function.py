import numpy as np

def compute_cost(X, y, theta):
    """
    Compute the cost for linear regression using mean squared error.
    
    Parameters:
        X (ndarray): Feature matrix of shape (m, n) where m is the number of samples.
        y (ndarray): Target values of shape (m, ).
        theta (ndarray): Parameter vector of shape (n, ).       
    
    Returns:
        float: The cost (mean squared error).
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    squared_errors = errors**2
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    return cost


def compute_gradient(X, y, theta):
    """
    Compute the gradient of the cost function for linear regression.
    
    Parameters:
        X (ndarray): Feature matrix of shape (m, n) where m is the number of samples.
        y (ndarray): Target values of shape (m, ).
        theta (ndarray): Parameter vector of shape (n, ).       
    
    Returns:
        ndarray: The gradient vector of shape (n, ).
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    gradient = (1 / m) * X.T.dot(errors)
    return gradient