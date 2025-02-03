import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import our custom Logistic Regression model
from logistic_regression import LogisticRegression

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
# Add an intercept term (bias) to X
X = np.hstack([np.ones((X.shape[0], 1)), X])
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(learning_rate=0.01, iterations=1000)
model.fit(X_train, y_train)

# Print the final cost after training
print('Final cost:', model.cost_history[-1])

# Plot the cost history
plt.plot(range(len(model.cost_history)), model.cost_history, 'b-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History over Iterations')
plt.show()

# Predict on test set and evaluate accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print('Test Accuracy:', accuracy)
