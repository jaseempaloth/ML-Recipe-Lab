from jax import jit, grad, vmap
import jax.numpy as jnp

def mse_loss(params, X, y, model):
    """
    Compute the Mean Squared Error (MSE) loss.
    MSE = (1/n) * sum((preds - y)^2)

    Args:
        params (dict): Model parameters
        X (jnp.ndarray): Input features
        y (jnp.ndarray): Target values
        model (callable): The model function to compute predictions
        
    Returns:
        jnp.ndarray: Computed MSE loss
    """
    preds = model.forward(params, X)
    return jnp.mean(jnp.square(preds - y))

def mae_loss(params, X, y, model):
    """
    Compute the Mean Absolute Error (MAE) loss.
    MAE = (1/n) * sum(|preds - y|)

    Args:
        params (dict): Model parameters
        X (jnp.ndarray): Input features
        y (jnp.ndarray): Target values
        model (callable): The model function to compute predictions
        
    Returns:
        jnp.ndarray: Computed MAE loss
    """
    preds = model.forward(params, X)
    return jnp.mean(jnp.abs(preds - y))

def rmse_loss(params, X, y, model):
    """
    Compute the Root Mean Squared Error (RMSE) loss.
    RMSE = sqrt((1/n) * sum((preds - y)^2))

    Args:
        params (dict): Model parameters
        X (jnp.ndarray): Input features
        y (jnp.ndarray): Target values
        model (callable): The model function to compute predictions
        
    Returns:
        jnp.ndarray: Computed MSE loss
    """
    preds = model.forward(params, X)
    return jnp.sqrt(jnp.mean(jnp.square(preds - y)))

# Compute the gradient of the loss function with respect to the parameters
mse_grad = jit(grad(mse_loss, argnums=0))
mae_grad = jit(grad(mae_loss, argnums=0))
rmse_grad = jit(grad(rmse_loss, argnums=0))

# Vectorized versions for batch processing
batch_mse_loss = vmap(mse_loss, in_axes=(None, 0, 0, None), out_axes=0)






