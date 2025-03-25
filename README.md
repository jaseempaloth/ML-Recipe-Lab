# Machine Learning from Scratch

This project provides implementations of basic machine learning algorithms from scratch, focusing on linear regression and logistic regression. It is designed as an educational tool to understand the inner workings of these models and the optimization techniques like gradient descent.

## Project Structure

```
ML Recipe Lab/
├── linear_regression/
│   ├── cost_function.py      # Cost function and gradient for linear regression
│   ├── gradient_descent.py     # Gradient descent implementation (if applicable)
│   ├── linear_regression.py    # Linear regression model implementation
│   ├── test_linear_regression.py           # Test file using synthetic data
│   ├── test_linear_regression_sklearn.py     # Test file using sklearn's diabetes dataset
│   └── linear_regression_demo.ipynb          # Jupyter Notebook demo for linear regression
├── logistic_regression/
│   ├── logistic_regression.py  # Logistic regression model implementation
│   └── test_logistic_regression.py           # Test file using sklearn's breast cancer dataset
└── README.md
```

## Dependencies

- Python 3.x
- numpy
- matplotlib
- scikit-learn

You can install the required packages using:

```bash
pip install numpy matplotlib scikit-learn
```

## Linear Regression

The linear regression implementation includes:

- **Cost function:** Mean Squared Error (MSE) with a 1/(2*m) factor (for gradient simplicity).
- **Gradient descent:** Used to iteratively minimize the cost function.
- **Prediction:** After training, the model can predict outputs given new inputs.

You can test the linear regression model using:

- `test_linear_regression.py` for synthetic data.
- `test_linear_regression_sklearn.py` for real data (diabetes dataset).

To run a test, execute:

```bash
python3 linear_regression/test_linear_regression_sklearn.py
```

## Logistic Regression

The logistic regression implementation from scratch includes:

- **Sigmoid function:** To map the linear combination of inputs to probabilities.
- **Binary cross-entropy cost function:** For measuring the loss in classification tasks.
- **Gradient descent:** To optimize the model parameters.
- **Prediction:** Functions to predict probability and binary labels (0/1) using a threshold.

You can test the logistic regression model using the breast cancer dataset from scikit-learn. Run:

```bash
python3 logistic_regression/test_logistic_regression.py
```

## Jupyter Notebook Demo

A Jupyter Notebook demo for linear regression is provided in:

- `linear_regression/linear_regression_demo.ipynb`

This notebook demonstrates the complete workflow:

1. Loading real data from scikit-learn.
2. Training the linear regression model.
3. Plotting the cost history over iterations.
4. Making predictions.

To run the notebook, execute:

```bash
jupyter notebook
```

and open the `linear_regression_demo.ipynb` file.

## License

This project is provided for educational purposes.
