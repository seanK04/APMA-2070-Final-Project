import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from .model import AdaptiveActivation  # Changed to relative import

def generate_domain_points(n_points, x_range=(-10, 10), t_range=(0, 5)):
    """
    Generate collocation points in the domain for training
    """
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    t = np.random.uniform(t_range[0], t_range[1], n_points)
    domain_points = np.column_stack((x, t))
    return tf.convert_to_tensor(domain_points, dtype=tf.float32)

def generate_initial_points(n_points, x_range=(-10, 10)):
    """
    Generate collocation points at t=0 for initial conditions
    """
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    t = np.zeros(n_points)
    ic_points = np.column_stack((x, t))
    return tf.convert_to_tensor(ic_points, dtype=tf.float32)

def train_model_tf(model, optimizer, n_iterations, n_domain_points, n_ic_points, 
                  lambda_pde=1.0, lambda_ic=10.0, device=None):  # device parameter for compatibility
    """
    Train the PINN model to solve the Boussinesq equation
    
    Parameters:
        model: PINN model
        optimizer: TensorFlow optimizer
        n_iterations: Number of training iterations
        n_domain_points: Number of collocation points in the domain
        n_ic_points: Number of initial condition points
        lambda_pde: Weight for PDE loss
        lambda_ic: Weight for initial condition loss
        device: Not used in TensorFlow, kept for compatibility with PyTorch version
    
    Returns:
        history: Dictionary containing loss history
    """
    history = {
        'total_loss': [],
        'pde_loss': [],
        'ic_loss': [],
        'time': []
    }
    
    start_time = time.time()
    
    # Training loop
    for iteration in tqdm(range(n_iterations)):
        # Generate new collocation points for each iteration
        domain_points = generate_domain_points(n_domain_points)
        ic_points = generate_initial_points(n_ic_points)
        
        # Compute gradients and update weights
        with tf.GradientTape() as tape:
            # Compute PDE residual loss
            pde_residual = model.compute_pde_residual(domain_points)
            pde_loss = tf.reduce_mean(tf.square(pde_residual))
            
            # Compute initial condition residual loss
            ic_residual_u, ic_residual_u_t = model.compute_ic_residual(ic_points)
            ic_loss = tf.reduce_mean(tf.square(ic_residual_u)) + tf.reduce_mean(tf.square(ic_residual_u_t))
            
            # Combine losses
            total_loss = lambda_pde * pde_loss + lambda_ic * ic_loss
            
        # Get gradients
        gradients = tape.gradient(total_loss, model.trainable_variables)
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Record losses
        history['total_loss'].append(total_loss.numpy())
        history['pde_loss'].append(pde_loss.numpy())
        history['ic_loss'].append(ic_loss.numpy())
        history['time'].append(time.time() - start_time)
        
        # Print losses every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f'Iteration {iteration + 1}/{n_iterations}, '
                  f'Total Loss: {total_loss.numpy():.6e}, '
                  f'PDE Loss: {pde_loss.numpy():.6e}, '
                  f'IC Loss: {ic_loss.numpy():.6e}')
    
    return history

def evaluate_model(model, x_range=(-10, 10), t_range=(0, 5), resolution=100, device=None):  # device parameter for compatibility
    """
    Evaluate the trained model on a grid and visualize the solution
    """
    # Create a grid for evaluation
    x = np.linspace(x_range[0], x_range[1], resolution)
    t = np.linspace(t_range[0], t_range[1], resolution)
    X, T = np.meshgrid(x, t)
    
    # Reshape to a 2D tensor
    points = np.stack((X.flatten(), T.flatten()), axis=1)
    points_tensor = tf.convert_to_tensor(points, dtype=tf.float32)
    
    # Predict u values
    u_pred = model(points_tensor).numpy()
    
    # Reshape to grid
    u_grid = u_pred.reshape(resolution, resolution)
    
    return X, T, u_grid

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
    plt.show()

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
    plt.show()

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
    plt.show()

def plot_adaptive_parameters(model, title='Adaptive Activation Parameters', save_path=None):
    """
    Plot the values of adaptive parameters after training
    """
    if not model.adaptive:
        print("Model does not use adaptive activation functions.")
        return
    
    a_values = []
    for layer in model.layer_list:
        if isinstance(layer, AdaptiveActivation):
            a_values.append(layer.a.numpy())
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(a_values)), a_values)
    plt.xlabel('Layer')
    plt.ylabel('Parameter Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def plot_comparison_tf(histories, labels, title, ylabel='Total Loss', save_path=None):
    """
    Create a comparison plot of training histories
    
    Parameters:
        histories: List of history dictionaries or list of loss values
        labels: List of labels for each history
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for i, history in enumerate(histories):
        if isinstance(history, dict):
            # If history is a dictionary, assume it has 'total_loss' key
            plt.semilogy(history['total_loss'], label=labels[i])
        else:
            # If history is a list of values
            plt.semilogy(history, label=labels[i])
    
    plt.xlabel('Iterations')
    plt.ylabel(ylabel + ' (log scale)')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()