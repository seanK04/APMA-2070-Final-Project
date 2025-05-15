# train.py - Fixed JAX Implementation for Training

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Callable, Any
from functools import partial

# Import fixed model functions
from .model import compute_pde_residual, compute_ic_residual


def generate_domain_points(n_points, x_range=(-10, 10), t_range=(0, 5)):
    """
    Generate collocation points in the domain for training
    """
    key = jax.random.PRNGKey(0)
    x_key, t_key = jax.random.split(key)
    
    x = jax.random.uniform(x_key, (n_points, 1), minval=x_range[0], maxval=x_range[1])
    t = jax.random.uniform(t_key, (n_points, 1), minval=t_range[0], maxval=t_range[1])
    
    return jnp.concatenate([x, t], axis=1)


def generate_initial_points(n_points, x_range=(-10, 10)):
    """
    Generate collocation points at t=0 for initial conditions
    """
    key = jax.random.PRNGKey(1)
    x = jax.random.uniform(key, (n_points, 1), minval=x_range[0], maxval=x_range[1])
    t = jnp.zeros((n_points, 1))
    
    return jnp.concatenate([x, t], axis=1)


def loss_fn(apply_fn, params, domain_points, ic_points, lambda_pde=1.0, lambda_ic=10.0):
    """
    Compute the total loss function for the PDE
    """
    # Compute PDE residual loss
    pde_residual = compute_pde_residual(apply_fn, params, domain_points)
    pde_loss = jnp.mean(jnp.square(pde_residual))
    
    # Compute initial condition residual loss
    ic_residual_u, ic_residual_u_t = compute_ic_residual(apply_fn, params, ic_points)
    ic_loss = jnp.mean(jnp.square(ic_residual_u)) + jnp.mean(jnp.square(ic_residual_u_t))
    
    # Combine losses
    total_loss = lambda_pde * pde_loss + lambda_ic * ic_loss
    
    return total_loss, (pde_loss, ic_loss)


# Adjusted to match the expected signature from the main script
def train_model_jax(model, params, n_iterations, n_domain_points, n_ic_points, 
                   learning_rate=1e-3, lambda_pde=1.0, lambda_ic=10.0):
    """
    Train the PINN model to solve the Boussinesq equation using JAX
    
    Parameters:
        model: Haiku transformed model with init and apply methods
        params: Initial model parameters
        n_iterations: Number of training iterations
        n_domain_points: Number of collocation points in the domain
        n_ic_points: Number of initial condition points
        learning_rate: Learning rate for optimizer
        lambda_pde: Weight for PDE loss
        lambda_ic: Weight for initial condition loss
    
    Returns:
        history: Dictionary containing loss history
        params: Trained model parameters
    """
    # Get the apply function from the model
    apply_fn = model.apply
    
    # Create optimizer outside the JIT function
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Define the value and grad function outside of the JIT function
    def loss_with_params(p, domain_pts, ic_pts):
        return loss_fn(apply_fn, p, domain_pts, ic_pts, lambda_pde, lambda_ic)
    
    value_and_grad_fn = jax.value_and_grad(loss_with_params, has_aux=True)
    
    # Define update step without jitting yet
    def update_step(params, opt_state, domain_points, ic_points):
        # Compute gradients
        (loss, aux), grads = value_and_grad_fn(params, domain_points, ic_points)
        
        # Apply updates with optimizer
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss, aux
    
    # Now JIT the update step that doesn't contain the optimizer as an argument
    update_step_jit = jax.jit(update_step)
    
    # Initialize history
    history = {
        'total_loss': [],
        'pde_loss': [],
        'ic_loss': [],
        'time': []
    }
    
    # For measuring time
    start_time = time.time()
    
    # Training loop
    for iteration in tqdm(range(n_iterations)):
        # Generate new points for this iteration
        domain_points_batch = generate_domain_points(n_domain_points)
        ic_points_batch = generate_initial_points(n_ic_points)
        
        # Update step
        params, opt_state, total_loss, (pde_loss, ic_loss) = update_step_jit(
            params, opt_state, domain_points_batch, ic_points_batch
        )
        
        # Record losses
        history['total_loss'].append(float(total_loss))
        history['pde_loss'].append(float(pde_loss))
        history['ic_loss'].append(float(ic_loss))
        history['time'].append(time.time() - start_time)
        
        # Print losses every 100 iterations or at the end
        if (iteration + 1) % 100 == 0 or iteration == n_iterations - 1:
            print(f'Iteration {iteration + 1}/{n_iterations}, '
                  f'Total Loss: {float(total_loss):.6e}, '
                  f'PDE Loss: {float(pde_loss):.6e}, '
                  f'IC Loss: {float(ic_loss):.6e}')
    
    return history, params


def evaluate_model(model, params, x_range=(-10, 10), t_range=(0, 5), resolution=100, device=None):
    """
    Evaluate the trained model on a grid and return the solution
    
    This function is adjusted to match the signature expected by the main script
    """
    # Create a grid for evaluation
    x = np.linspace(x_range[0], x_range[1], resolution)
    t = np.linspace(t_range[0], t_range[1], resolution)
    X, T = np.meshgrid(x, t)
    
    # Reshape to a 2D tensor
    points = np.stack((X.flatten(), T.flatten()), axis=1)
    points_tensor = jnp.array(points)
    
    # Extract apply_fn from model
    apply_fn = model.apply
    
    # Predict u values (in batches to avoid memory issues)
    batch_size = 1000
    n_batches = (points_tensor.shape[0] + batch_size - 1) // batch_size
    u_pred = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, points_tensor.shape[0])
        batch = points_tensor[start_idx:end_idx]
        
        # Apply model - use is_training=False for evaluation
        u_batch = apply_fn(params, batch, is_training=False)
        u_pred.append(u_batch)
    
    # Concatenate batches
    u_pred = jnp.concatenate(u_pred, axis=0).flatten()
    
    # Reshape to grid
    u_grid = u_pred.reshape(resolution, resolution)
    
    return X, T, np.array(u_grid)


def plot_solution_2d(X, T, u_grid, title, save_path=None):
    """
    Generate a 2D plot of the solution
    """
    plt.figure(figsize=(10, 8))
    
    # Plot 2D heatmap
    plt.pcolormesh(X, T, u_grid, cmap='viridis', shading='auto')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.close()


def plot_solution_3d(X, T, u_grid, title, save_path=None):
    """
    Generate a 3D surface plot of the solution
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D surface
    surf = ax.plot_surface(X, T, u_grid, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='u')
    
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.close()


def plot_loss_history(history, title='Training Loss History', save_path=None):
    """
    Plot the loss history during training
    """
    plt.figure(figsize=(12, 6))
    
    plt.semilogy(history['total_loss'], label='Total Loss')
    plt.semilogy(history['pde_loss'], label='PDE Loss')
    plt.semilogy(history['ic_loss'], label='IC Loss')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.close()


def plot_comparison_jax(histories, labels, title='Loss Comparison', ylabel='Total Loss', save_path=None):
    """
    Create a comparison plot of training histories
    
    Parameters:
        histories: List of history dictionaries
        labels: List of labels for each history
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for i, history in enumerate(histories):
        if isinstance(history, dict) and 'total_loss' in history:
            plt.semilogy(history['total_loss'], label=labels[i])
        else:
            # If history is a list or doesn't have 'total_loss'
            plt.semilogy(history, label=labels[i])
    
    plt.xlabel('Iterations')
    plt.ylabel(ylabel + ' (log scale)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.close()