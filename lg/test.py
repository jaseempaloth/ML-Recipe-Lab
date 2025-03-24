from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split



model = LinearRegression()
model.fit(X_train, y_train)