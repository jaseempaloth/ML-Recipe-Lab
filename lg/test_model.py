import unittest
import jax
import jax.numpy as jnp
from jax import random
from model import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def test_initialization(self):
        """Test if the LinearRegression model initializes correctly."""
        n_features = 5
        model = LinearRegression(n_features)
        
        # Print model parameters for debugging
        print(f"\nInitialized model with {n_features} features")
        print(f"Weights shape: {model.params['w'].shape}")
        print(f"Bias value: {model.params['b']}")
        
        # Check if model attributes are set correctly
        self.assertEqual(model.n_features, n_features)
        self.assertIn('w', model.params)
        self.assertIn('b', model.params)
        
    def test_param_shapes(self):
        """Test if the parameters have the correct shapes."""
        n_features = 3
        model = LinearRegression(n_features)
        
        # Check parameter shapes
        self.assertEqual(model.params['w'].shape, (n_features,))
        self.assertEqual(model.params['b'].shape, ())
        
    def test_forward(self):
        """Test the forward pass of the model."""
        n_features = 2
        model = LinearRegression(n_features)
        
        # Create known parameters for testing
        test_params = {
            'w': jnp.array([1.0, 2.0]),
            'b': 0.5
        }
        
        # Create sample input
        X = jnp.array([[1.0, 1.0], [2.0, 3.0]])
        
        # Calculate expected output manually
        expected_output = jnp.array([3.5, 8.5])  # (1*1 + 1*2 + 0.5) and (2*1 + 3*2 + 0.5)
        
        # Get model output
        output = model.forward(test_params, X)
        
        # Print inputs and outputs for debugging
        print("\nForward pass test:")
        print(f"Input X:\n{X}")
        print(f"Parameters:\nw: {test_params['w']}\nb: {test_params['b']}")
        print(f"Model output: {output}")
        print(f"Expected output: {expected_output}")
        
        # Check if output matches expected output
        if not jnp.allclose(output, expected_output, rtol=1e-5):
            raise AssertionError("Arrays are not equal within the specified tolerance")

# Add a demonstration section that runs outside of tests
def demonstrate_model():
    print("\n----- LinearRegression Model Demonstration -----")
    # Create model with 3 features
    model = LinearRegression(3)
    print(f"Model parameters after initialization: {model.params}")
    
    # Create sample data
    X_sample = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    # Make prediction
    predictions = model.forward(model.params, X_sample)
    print(f"Sample input:\n{X_sample}")
    print(f"Model predictions: {predictions}")

if __name__ == '__main__':
    unittest.main()
    # Note: The following line won't run during unittest execution
    # To run the demonstration, comment out unittest.main() and uncomment:
    # demonstrate_model()
