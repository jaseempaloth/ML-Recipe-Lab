import numpy as np

def gradient_descent(x, y):
    m, b = 0, 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.01
    cost_history = []  
    for i in range(iterations):
        y_pred = m * x + b
        cost = (1 / n) * np.sum((y - y_pred)**2)
        m_gradient = -(2 / n) * np.sum(x * (y - y_pred))
        b_gradient = -(2 / n) * np.sum(y - y_pred)
        m = m - learning_rate * m_gradient
        b = b - learning_rate * b_gradient
        cost_history.append(cost)  
        print(f'm: {m}, b: {b}, cost: {cost}, iteration: {i}')
    return m, b, cost_history

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

m, b, cost_history = gradient_descent(x, y)

# plot the loss history
import matplotlib.pyplot as plt
plt.plot(range(len(cost_history)), cost_history, 'b-')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History over Iterations')
plt.show()