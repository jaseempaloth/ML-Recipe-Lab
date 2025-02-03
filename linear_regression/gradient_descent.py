import numpy as np
from cost_function import compute_cost, compute_gradient

def gradient_descent(X, y, theta, learning_rate, iterations):
    """
    Perform gradient descent to learn theta.
    
    Parameters:
        X (ndarray): Feature matrix of shape (m, n).
        y (ndarray): Target values of shape (m, ).
        theta (ndarray): Initial parameter vector of shape (n, ).
        learning_rate (float): Learning rate for gradient descent.
        iterations (int): Number of iterations.
    
    Returns:
        tuple: (theta, cost_history) where theta is the final parameter vector and 
               cost_history is a list containing the cost at each iteration.
    """
    cost_history = []

    for i in range(iterations):
        gradient = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history