import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_domain_points(n_points, x_range=(-10, 10), t_range=(0, 5)):
    """
    Generate collocation points in the domain for training
    """
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    t = np.random.uniform(t_range[0], t_range[1], n_points)
    domain_points = np.column_stack((x, t))
    return torch.tensor(domain_points, dtype=torch.float32)

def generate_initial_points(n_points, x_range=(-10, 10)):
    """
    Generate collocation points at t=0 for initial conditions
    """
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    t = np.zeros(n_points)
    ic_points = np.column_stack((x, t))
    return torch.tensor(ic_points, dtype=torch.float32)

def train_model_pt(model, optimizer, n_iterations, n_domain_points, n_ic_points, lambda_pde=1.0, lambda_ic=10.0, device='cpu'):
    """
    Train the PINN model to solve the Boussinesq equation
    
    Parameters:
        model: PINN model
        optimizer: PyTorch optimizer
        n_iterations: Number of training iterations
        n_domain_points: Number of collocation points in the domain
        n_ic_points: Number of initial condition points
        lambda_pde: Weight for PDE loss
        lambda_ic: Weight for initial condition loss
        device: Device to run the model on ('cpu' or 'cuda')
    
    Returns:
        history: Dictionary containing loss history
    """
    history = {
        'total_loss': [],
        'pde_loss': [],
        'ic_loss': [],
        'time': []
    }
    
    model.train()
    start_time = time.time()
    
    # Training loop
    for iteration in tqdm(range(n_iterations)):
        # Generate new collocation points for each iteration
        domain_points = generate_domain_points(n_domain_points).to(device)
        ic_points = generate_initial_points(n_ic_points).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute PDE residual loss
        pde_residual = model.compute_pde_residual(domain_points)
        pde_loss = torch.mean(torch.square(pde_residual))
        
        # Compute initial condition residual loss
        ic_residual_u, ic_residual_u_t = model.compute_ic_residual(ic_points)
        ic_loss = torch.mean(torch.square(ic_residual_u)) + torch.mean(torch.square(ic_residual_u_t))
        
        # Combine losses
        total_loss = lambda_pde * pde_loss + lambda_ic * ic_loss
        
        # Backpropagation
        total_loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Record losses
        history['total_loss'].append(total_loss.item())
        history['pde_loss'].append(pde_loss.item())
        history['ic_loss'].append(ic_loss.item())
        history['time'].append(time.time() - start_time)
        
        # Print losses every 100 iterations
        if (iteration + 1) % 100 == 0:
            print(f'Iteration {iteration + 1}/{n_iterations}, '
                  f'Total Loss: {total_loss.item():.6e}, '
                  f'PDE Loss: {pde_loss.item():.6e}, '
                  f'IC Loss: {ic_loss.item():.6e}')
    
    return history

def evaluate_model(model, x_range=(-10, 10), t_range=(0, 5), resolution=100, device='cpu'):
    """
    Evaluate the trained model on a grid and visualize the solution
    """
    model.eval()
    
    # Create a grid for evaluation
    x = np.linspace(x_range[0], x_range[1], resolution)
    t = np.linspace(t_range[0], t_range[1], resolution)
    X, T = np.meshgrid(x, t)
    
    # Reshape to a 2D tensor
    points = np.stack((X.flatten(), T.flatten()), axis=1)
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    
    # Predict u values
    with torch.no_grad():
        u_pred = model(points_tensor).cpu().numpy()
    
    # Reshape to grid
    u_grid = u_pred.reshape(resolution, resolution)
    
    return X, T, u_grid

def calculate_error_metrics(u_pred, u_ref):
    """
    Calculate error metrics between predicted and reference solutions
    
    Args:
        u_pred (torch.Tensor): Predicted solution
        u_ref (torch.Tensor): Reference solution
        
    Returns:
        tuple: (l2_error, max_error, rmse) - Error metrics
    """
    # Ensure tensors are on the same device
    if u_pred.device != u_ref.device:
        u_pred = u_pred.to(u_ref.device)
    
    # L2 relative error
    l2_error = torch.sqrt(torch.mean((u_pred - u_ref)**2)) / torch.sqrt(torch.mean(u_ref**2))
    
    # Maximum pointwise error
    max_error = torch.max(torch.abs(u_pred - u_ref))
    
    # Root mean square error (RMSE)
    rmse = torch.sqrt(torch.mean((u_pred - u_ref)**2))
    
    return l2_error.item(), max_error.item(), rmse.item()

def compute_residual_analysis(model, domain_points, device='cpu'):
    """
    Compute PDE residual analysis for the Boussinesq equation
    
    Args:
        model: PINN model
        domain_points: Points in the domain for analysis
        device: Device to run the model on
        
    Returns:
        tuple: (terms, absolute_residuals, mean_residuals)
    """
    # Move to device and ensure we clone to not modify the original
    domain_points = domain_points.clone().to(device)
    
    # Ensure model is in eval mode
    model.eval()
    
    # Create tensor with requires_grad=True
    x = domain_points.clone().detach().requires_grad_(True)
    
    # Forward pass to get u
    u = model(x)
    
    # Dummy variable for grad computation
    dummy = u.sum()
    
    # Get first derivatives
    grad_u = torch.autograd.grad(dummy, x, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_t = grad_u[:, 1:2]
    
    # Compute second-order derivatives
    u_tt = torch.autograd.grad(u_t.sum(), x, create_graph=True)[0][:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
    
    # For the term ∂²/∂x²(3u²)
    u_squared = 3 * (u ** 2)
    u_squared_grad = torch.autograd.grad(u_squared.sum(), x, create_graph=True)[0]
    u_squared_x = u_squared_grad[:, 0:1]
    u_squared_xx = torch.autograd.grad(u_squared_x.sum(), x, create_graph=True)[0][:, 0:1]
    
    # For the fourth derivative
    u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0][:, 0:1]
    u_xxxx = torch.autograd.grad(u_xxx.sum(), x, create_graph=True)[0][:, 0:1]
    
    # Compute residual components
    residual = u_tt - u_xx - u_squared_xx + u_xxxx
    
    # Convert to numpy for plotting
    abs_u_tt = torch.abs(u_tt).detach().cpu().numpy()
    abs_u_xx = torch.abs(u_xx).detach().cpu().numpy()
    abs_u_squared_xx = torch.abs(u_squared_xx).detach().cpu().numpy()
    abs_u_xxxx = torch.abs(u_xxxx).detach().cpu().numpy()
    
    # Compute mean residuals for each term
    mean_u_tt = torch.mean(torch.abs(u_tt)).item()
    mean_u_xx = torch.mean(torch.abs(u_xx)).item()
    mean_u_squared_xx = torch.mean(torch.abs(u_squared_xx)).item()
    mean_u_xxxx = torch.mean(torch.abs(u_xxxx)).item()
    mean_residual = torch.mean(torch.abs(residual)).item()
    
    # Store terms for plotting
    terms = ['∂²ψ/∂τ²', '∂²ψ/∂ξ²', '∂²/∂ξ²(3ψ²)', '∂⁴ψ/∂ξ⁴', 'Total']
    absolute_residuals = [abs_u_tt, abs_u_xx, abs_u_squared_xx, abs_u_xxxx, residual.detach().cpu().numpy()]
    mean_residuals = [mean_u_tt, mean_u_xx, mean_u_squared_xx, mean_u_xxxx, mean_residual]
    
    return terms, absolute_residuals, mean_residuals

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

def plot_adaptive_parameters(model, title='Adaptive Activation Parameters', save_path=None):
    """
    Plot the values of adaptive parameters after training
    """
    if not model.adaptive:
        print("Model does not use adaptive activation functions.")
        return
    
    a_values = [af.a.item() for af in model.activation_functions]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(a_values)), a_values)
    plt.xlabel('Layer')
    plt.ylabel('Parameter Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.close()

def plot_comparison_pt(histories, labels, title, ylabel='Total Loss', save_path=None):
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
            plt.semilogy(history['total_loss'], label=labels[i])
        else:
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

def plot_residual_heatmap(X, T, residual, title, save_path=None):
    """Generate a heatmap of the residual distribution"""
    plt.figure(figsize=(12, 9))
    residual_grid = residual.reshape(X.shape)
    plt.pcolormesh(X, T, residual_grid, cmap='viridis', shading='auto')
    plt.colorbar(label='Residual')
    plt.xlabel('ξ')
    plt.ylabel('τ')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_residual_bar_chart(terms, mean_residuals, title, save_path=None):
    """Generate a bar chart of mean residual values for each term"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(terms)), mean_residuals, color='skyblue')
    plt.xticks(range(len(terms)), terms, rotation=45, ha='right')
    plt.xlabel('Terms in Boussinesq Equation')
    plt.ylabel('Mean Absolute Residual')
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add values on top of bars
    for i, v in enumerate(mean_residuals):
        plt.text(i, v + 0.01 * max(mean_residuals), f'{v:.2e}', ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_error_metrics_table(error_metrics, model_names, title, save_path=None):
    """Generate a table of error metrics"""
    fig = plt.figure(figsize=(12, 4))
    ax = plt.gca()
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table data
    table_data = []
    for i, model_name in enumerate(model_names):
        l2_error, max_error, rmse = error_metrics[i]
        table_data.append([model_name, f'{l2_error:.6e}', f'{max_error:.6e}', f'{rmse:.6e}'])
    
    # Create the table
    table = plt.table(
        cellText=table_data,
        colLabels=['Model', 'L2 Relative Error', 'Max Pointwise Error', 'RMSE'],
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title(title, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_solutions_plot(models, model_names, x_range, t_range, resolution, title, save_path=None):
    """Compare solutions from different models in a subplot grid"""
    n_models = len(models)
    rows = (n_models + 1) // 2  # +1 to account for reference solution
    
    fig, axes = plt.subplots(rows, 2, figsize=(16, 5*rows), 
                             subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        X, T, u_grid = evaluate_model(model, x_range, t_range, resolution)
        
        # Plot 3D surface
        surf = axes[i].plot_surface(X, T, u_grid, cmap='viridis', edgecolor='none')
        
        axes[i].set_xlabel('ξ')
        axes[i].set_ylabel('τ')
        axes[i].set_zlabel('ψ')
        axes[i].set_title(f"{name}")
    
    # Hide any unused subplots
    for i in range(n_models, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
