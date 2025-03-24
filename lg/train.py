from jax import jit, grad, value_and_grad
import jax.numpy as jnp
import time

def train_step(params, X_batch, y_batch, model, loss_fn, learning_rate):
    """Single training step without JIT for clarity."""
    # Compute the loss and gradients
    loss_value_fn = lambda p: loss_fn(p, X_batch, y_batch, model)
    loss, grads = value_and_grad(loss_value_fn)(params)
    
    # Update the parameters using the optimizer
    params = {k: p - learning_rate * g for k, (p, g) in
              zip(params.keys(), zip(params.values(), grads.values()))}
    return params, loss

def train(model, loss_fn, X_train, X_test, y_train, y_test, learning_rate=0.01, epochs=100, batch_size=32, verbose=True):
    """Train the linear regression model."""
    from data import DataLoader

    # Verify input shapes
    assert X_train.shape[0] == y_train.shape[0], f"Sample mismatch: X_train has {X_train.shape[0]} samples, y_train has {y_train.shape[0]}"
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        print(f"Warning: y_train has shape {y_train.shape}, reshaping to 1D array")
        y_train = y_train[:, 0]  # Take first column if multi-dimensional
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        print(f"Warning: y_test has shape {y_test.shape}, reshaping to 1D array")
        y_test = y_test[:, 0]  # Take first column if multi-dimensional

    # Initialize history
    history = {
        'train_loss': [],
        'test_loss': [],
        'time_per_epoch': []
    }

    # Initialize the parameters
    params = model.params

    # Create a DataLoader 
    data_loader = DataLoader(X_train, y_train, batch_size=batch_size)

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = []

        # Batch training
        for batch_X, batch_y in data_loader:
            params, loss = train_step(
                params, batch_X, batch_y, model, loss_fn, learning_rate
            )
            epoch_loss.append(loss)

        # Calculate average loss for the epoch
        train_loss = jnp.mean(jnp.array(epoch_loss))

        # Measure epoch time
        epoch_time = time.time() - start_time

        # Calculate test loss
        test_loss = loss_fn(params, X_test, y_test, model)

        # Record history
        history['train_loss'].append(float(train_loss))
        history['test_loss'].append(float(test_loss))
        history['time_per_epoch'].append(epoch_time)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {float(train_loss):.4f} - "
                  f"Test Loss: {float(test_loss):.4f} - "
                  f"Time: {epoch_time:.2f}s")
    
    # Update model parameters
    model.params = params
    
    return history




