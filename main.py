import os
import torch
import tensorflow as tf
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# Add directories to Python path to make the imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import PyTorch modules
from pt.model import PINN_pt
from pt.train import (
    train_model_pt, evaluate_model as evaluate_model_pt, generate_domain_points,
    calculate_error_metrics, compute_residual_analysis, plot_residual_heatmap,
    plot_residual_bar_chart, create_error_metrics_table, compare_solutions_plot
)

# Import TensorFlow modules
from tf.model import PINN_tf
from tf.train import train_model_tf, evaluate_model as evaluate_model_tf

# Import JAX modules
from boussinesq_jax.model import create_pinn_model
from boussinesq_jax.train import train_model_jax, evaluate_model as evaluate_model_jax

def create_task1_visualizations(results, save_dir, device):
    """Create visualizations for Task 1 (Activation Function Comparison)"""
    # Filter PyTorch models for fixed activations
    pt_fixed_models = {k: v for k, v in results.items() 
                      if v['framework'] == 'pytorch' and 'Fixed' in k}
    
    # Filter PyTorch models for adaptive activations
    pt_adaptive_models = {k: v for k, v in results.items() 
                         if v['framework'] == 'pytorch' and 'Adaptive' in k}
    
    # Common parameters
    x_range = (-10, 10)
    t_range = (0, 5)
    resolution = 100
    n_points = 5000

    # 1. Solution Visualization Plot
    # Get models and names for fixed activations
    fixed_models = [v['model'] for k, v in pt_fixed_models.items()]
    fixed_names = [k.split()[-1] for k in pt_fixed_models.keys()]
    
    # Create solution comparison plot
    compare_solutions_plot(
        fixed_models, fixed_names, x_range, t_range, resolution,
        "Solution Comparison for Different Activation Functions",
        os.path.join(save_dir, "task1_solution_comparison.png")
    )
    
    # 2. PDE Residual Analysis
    for name, result in pt_fixed_models.items():
        model = result['model']
        activation = name.split()[-1]
        
        # Create evaluation grid for visualization
        X, T, _ = evaluate_model_pt(model, x_range, t_range, resolution, device)
        
        # For residual analysis, use a smaller set of points
        domain_points = generate_domain_points(n_points, x_range, t_range)
        
        # Compute residual analysis
        terms, absolute_residuals, mean_residuals = compute_residual_analysis(
            model, domain_points, device
        )
        
        # Plot residual bar chart
        plot_residual_bar_chart(
            terms, mean_residuals,
            f"Mean Residual Values for Each Term - {activation}",
            os.path.join(save_dir, f"task1_residual_barchart_{activation.lower()}.png")
        )
        
        # For heatmap, compute residuals on a smaller grid to avoid memory issues
        subsample_res = 50  # Lower resolution for residual computation
        x_sub = np.linspace(x_range[0], x_range[1], subsample_res)
        t_sub = np.linspace(t_range[0], t_range[1], subsample_res)
        X_sub, T_sub = np.meshgrid(x_sub, t_sub)
        
        # Create points for the subsampled grid
        points_sub = np.stack((X_sub.flatten(), T_sub.flatten()), axis=1)
        points_tensor = torch.tensor(points_sub, dtype=torch.float32).to(device)
        
        # Compute PDE residual on subsampled grid
        residual = model.compute_pde_residual(points_tensor).detach().cpu().numpy()
        
        # Plot residual heatmap
        plot_residual_heatmap(
            X_sub, T_sub, residual, 
            f"PDE Residual Distribution - {activation}",
            os.path.join(save_dir, f"task1_residual_heatmap_{activation.lower()}.png")
        )
    
    # 3. Error Metrics Table
    # Use the best model as reference
    best_model_name = min(pt_fixed_models.items(), key=lambda x: x[1]['final_total_loss'])[0]
    best_model = pt_fixed_models[best_model_name]['model']
    
    # Get reference solution
    X_ref, T_ref, u_ref_grid = evaluate_model_pt(best_model, x_range, t_range, resolution, device)
    u_ref = torch.tensor(u_ref_grid.flatten(), device=device)
    
    # Calculate error metrics for all models against reference
    error_metrics = []
    model_names = []
    
    for name, result in pt_fixed_models.items():
        if name != best_model_name:  # Skip reference model
            model = result['model']
            model_names.append(name.split()[-1])
            
            _, _, u_grid = evaluate_model_pt(model, x_range, t_range, resolution, device)
            u_pred = torch.tensor(u_grid.flatten(), device=device)
            
            metrics = calculate_error_metrics(u_pred, u_ref)
            error_metrics.append(metrics)
    
    # Create error metrics table
    create_error_metrics_table(
        error_metrics, model_names,
        "Error Metrics Comparison (Reference: Best Model)",
        os.path.join(save_dir, "task1_error_metrics_table.png")
    )
    
    # 4. Training Performance
    # Extract training metrics for fixed activation models
    fixed_histories = [v['history'] for k, v in pt_fixed_models.items()]
    fixed_labels = [k.split()[-1] for k in pt_fixed_models.keys()]
    
    # Plot training loss history
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(fixed_histories):
        plt.semilogy(history['total_loss'], label=fixed_labels[i])
    
    plt.xlabel('Iterations')
    plt.ylabel('Total Loss (log scale)')
    plt.title('Training Loss Convergence for Different Activation Functions')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "task1_training_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create table with final loss values and training times
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for i, name in enumerate(pt_fixed_models.keys()):
        result = pt_fixed_models[name]
        activation = name.split()[-1]
        
        final_loss = result['final_total_loss']
        training_time = result['training_time']
        
        table_data.append([activation, f'{final_loss:.6e}', f'{training_time:.2f}'])
    
    table = plt.table(
        cellText=table_data,
        colLabels=['Activation Function', 'Final Loss', 'Training Time (s)'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title("Training Performance Metrics", pad=20)
    plt.savefig(os.path.join(save_dir, "task1_training_metrics_table.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Fixed vs. Adaptive Activation Comparison
    # Group models by activation type
    activation_pairs = {}
    for name, result in {**pt_fixed_models, **pt_adaptive_models}.items():
        activation = name.split()[-1]
        type_prefix = 'Fixed' if 'Fixed' in name else 'Adaptive'
        
        if activation not in activation_pairs:
            activation_pairs[activation] = {}
        
        activation_pairs[activation][type_prefix] = result
    
    # Compare fixed vs. adaptive for each activation function
    for activation, pair in activation_pairs.items():
        if 'Fixed' in pair and 'Adaptive' in pair:
            # Error metrics comparison
            fixed_model = pair['Fixed']['model']
            adaptive_model = pair['Adaptive']['model']
            
            # Get solutions
            _, _, u_fixed_grid = evaluate_model_pt(fixed_model, x_range, t_range, resolution, device)
            u_fixed = torch.tensor(u_fixed_grid.flatten(), device=device)
            
            _, _, u_adaptive_grid = evaluate_model_pt(adaptive_model, x_range, t_range, resolution, device)
            u_adaptive = torch.tensor(u_adaptive_grid.flatten(), device=device)
            
            # Calculate error metrics between fixed and adaptive
            metrics = calculate_error_metrics(u_adaptive, u_fixed)
            
            # Create comparison table
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('tight')
            ax.axis('off')
            
            fixed_final_loss = pair['Fixed']['final_total_loss']
            adaptive_final_loss = pair['Adaptive']['final_total_loss']
            
            fixed_training_time = pair['Fixed']['training_time']
            adaptive_training_time = pair['Adaptive']['training_time']
            
            table_data = [
                ['Fixed', f'{fixed_final_loss:.6e}', f'{fixed_training_time:.2f}'],
                ['Adaptive', f'{adaptive_final_loss:.6e}', f'{adaptive_training_time:.2f}'],
                ['Difference (%)', f'{(1 - adaptive_final_loss/fixed_final_loss)*100:.2f}%', 
                 f'{(1 - adaptive_training_time/fixed_training_time)*100:.2f}%']
            ]
            
            table = plt.table(
                cellText=table_data,
                colLabels=['Type', 'Final Loss', 'Training Time (s)'],
                loc='center',
                cellLoc='center'
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            plt.title(f"Fixed vs. Adaptive {activation} Comparison", pad=20)
            plt.savefig(os.path.join(save_dir, f"task1_fixed_vs_adaptive_{activation.lower()}_table.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Comparative solution plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': '3d'})
            
            surf1 = ax1.plot_surface(X_ref, T_ref, u_fixed_grid, cmap='viridis', edgecolor='none')
            ax1.set_title(f"Fixed {activation}")
            ax1.set_xlabel('ξ')
            ax1.set_ylabel('τ')
            ax1.set_zlabel('ψ')
            
            surf2 = ax2.plot_surface(X_ref, T_ref, u_adaptive_grid, cmap='viridis', edgecolor='none')
            ax2.set_title(f"Adaptive {activation}")
            ax2.set_xlabel('ξ')
            ax2.set_ylabel('τ')
            ax2.set_zlabel('ψ')
            
            plt.suptitle(f"Solution Comparison: Fixed vs. Adaptive {activation}", fontsize=16)
            plt.savefig(os.path.join(save_dir, f"task1_fixed_vs_adaptive_{activation.lower()}_plot.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def create_task2_visualizations(results, save_dir):
    """Create visualizations for Task 2 (Framework Comparison)"""
    # 6. Training Performance Table
    frameworks = ['pytorch', 'tensorflow', 'jax']
    
    # Group results by framework
    framework_results = {fw: [] for fw in frameworks}
    for name, result in results.items():
        framework = result['framework']
        if framework in framework_results:
            framework_results[framework].append(result)
    
    # Create training performance table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for framework in frameworks:
        if framework_results[framework]:
            avg_loss = np.mean([r['final_total_loss'] for r in framework_results[framework]])
            avg_time = np.mean([r['training_time'] for r in framework_results[framework]])
            iters_per_sec = 1000 / avg_time  # assuming 1000 iterations
            
            table_data.append([
                framework.capitalize(), 
                f'{avg_loss:.6e}', 
                f'{avg_time:.2f}',
                f'{iters_per_sec:.2f}'
            ])
    
    table = plt.table(
        cellText=table_data,
        colLabels=['Framework', 'Avg Final Loss', 'Avg Training Time (s)', 'Iterations/second'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title("Framework Training Performance Comparison", pad=20)
    plt.savefig(os.path.join(save_dir, "task2_framework_performance_table.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Convergence Comparison Plot
    # Compare fixed tanh activation across frameworks as an example
    framework_histories = {}
    for framework in frameworks:
        for name, result in results.items():
            if result['framework'] == framework and 'Fixed Tanh' in name:
                framework_histories[framework] = result['history']['total_loss']
                break
    
    plt.figure(figsize=(12, 6))
    for framework, history in framework_histories.items():
        plt.semilogy(history, label=framework.capitalize())
    
    plt.xlabel('Iterations')
    plt.ylabel('Total Loss (log scale)')
    plt.title('Loss Convergence Comparison Across Frameworks (Fixed Tanh)')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "task2_convergence_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_overall_analysis(results, save_dir):
    """Create the overall analysis with ranking table"""
    # 8. Overall Summary Table
    # Create a list of all configurations with their metrics
    config_metrics = []
    
    for name, result in results.items():
        framework = result['framework']
        activation_type = 'Adaptive' if 'Adaptive' in name else 'Fixed'
        activation = name.split()[-1]
        
        config_metrics.append({
            'name': name,
            'framework': framework,
            'activation_type': activation_type,
            'activation': activation,
            'final_loss': result['final_total_loss'],
            'training_time': result['training_time']
        })
    
    # Sort by final loss (accuracy)
    config_metrics_by_loss = sorted(config_metrics, key=lambda x: x['final_loss'])
    
    # Sort by training time (efficiency)
    config_metrics_by_time = sorted(config_metrics, key=lambda x: x['training_time'])
    
    # Assign ranks
    for i, config in enumerate(config_metrics_by_loss):
        config['loss_rank'] = i + 1
    
    for i, config in enumerate(config_metrics_by_time):
        config['time_rank'] = i + 1
    
    # Sort by combined rank
    for config in config_metrics:
        config['combined_rank'] = config['loss_rank'] + config['time_rank']
    
    config_metrics_by_combined = sorted(config_metrics, key=lambda x: x['combined_rank'])
    
    # Create the summary table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for config in config_metrics_by_combined:
        table_data.append([
            config['combined_rank'],
            f"{config['framework'].capitalize()} {config['activation_type']} {config['activation']}",
            config['loss_rank'],
            f"{config['final_loss']:.6e}",
            config['time_rank'],
            f"{config['training_time']:.2f}"
        ])
    
    table = plt.table(
        cellText=table_data,
        colLabels=['Overall Rank', 'Configuration', 'Accuracy Rank', 'Final Loss', 
                  'Efficiency Rank', 'Training Time (s)'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.title("Overall Ranking of Activation Functions Across Frameworks", pad=20)
    plt.savefig(os.path.join(save_dir, "overall_summary_table.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hidden_layers = [50, 50, 50, 50]
    n_iterations = 1000
    learning_rate = 1e-3
    domain_points = 5000
    ic_points = 1000
    lambda_pde = 1.0
    lambda_ic = 10.0
    seed = 42
    save_dir = 'results'
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define configurations to train
    configs = [
        # PyTorch configurations
        {'framework': 'pytorch', 'activation': 'relu', 'adaptive': False, 'name': 'PyTorch Fixed ReLU'},
        {'framework': 'pytorch', 'activation': 'tanh', 'adaptive': False, 'name': 'PyTorch Fixed Tanh'},
        {'framework': 'pytorch', 'activation': 'sigmoid', 'adaptive': False, 'name': 'PyTorch Fixed Sigmoid'},
        {'framework': 'pytorch', 'activation': 'relu', 'adaptive': True, 'name': 'PyTorch Adaptive ReLU'},
        {'framework': 'pytorch', 'activation': 'tanh', 'adaptive': True, 'name': 'PyTorch Adaptive Tanh'},
        {'framework': 'pytorch', 'activation': 'sigmoid', 'adaptive': True, 'name': 'PyTorch Adaptive Sigmoid'},
        
        # TensorFlow configurations
        {'framework': 'tensorflow', 'activation': 'relu', 'adaptive': False, 'name': 'TensorFlow Fixed ReLU'},
        {'framework': 'tensorflow', 'activation': 'tanh', 'adaptive': False, 'name': 'TensorFlow Fixed Tanh'},
        {'framework': 'tensorflow', 'activation': 'sigmoid', 'adaptive': False, 'name': 'TensorFlow Fixed Sigmoid'},
        {'framework': 'tensorflow', 'activation': 'relu', 'adaptive': True, 'name': 'TensorFlow Adaptive ReLU'},
        {'framework': 'tensorflow', 'activation': 'tanh', 'adaptive': True, 'name': 'TensorFlow Adaptive Tanh'},
        {'framework': 'tensorflow', 'activation': 'sigmoid', 'adaptive': True, 'name': 'TensorFlow Adaptive Sigmoid'},
        
        # JAX configurations
        {'framework': 'jax', 'activation': 'relu', 'adaptive': False, 'name': 'JAX Fixed ReLU'},
        {'framework': 'jax', 'activation': 'tanh', 'adaptive': False, 'name': 'JAX Fixed Tanh'},
        {'framework': 'jax', 'activation': 'sigmoid', 'adaptive': False, 'name': 'JAX Fixed Sigmoid'},
        {'framework': 'jax', 'activation': 'relu', 'adaptive': True, 'name': 'JAX Adaptive ReLU'},
        {'framework': 'jax', 'activation': 'tanh', 'adaptive': True, 'name': 'JAX Adaptive Tanh'},
        {'framework': 'jax', 'activation': 'sigmoid', 'adaptive': True, 'name': 'JAX Adaptive Sigmoid'},
    ]
    
    # Store results for each configuration
    results = {}
    
    # Train each configuration
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Training: {config['name']}")
        print(f"{'='*50}")
        
        framework = config['framework']
        activation = config['activation']
        adaptive = config['adaptive']
        
        # Create and train model based on framework
        if framework == 'pytorch':
            model = PINN_pt(
                hidden_layers=hidden_layers,
                activation=activation,
                adaptive=adaptive,
                device=device
            ).to(device)
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Print model info
            print(f"Model: PyTorch PINN with {activation} activation ({'adaptive' if adaptive else 'fixed'})")
            print(f"Hidden layers: {hidden_layers}")
            print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            print(f"Running on: {device}")
            
            # Train model
            print(f"\nTraining for {n_iterations} iterations...")
            start_time = time.time()
            
            history = train_model_pt(
                model=model,
                optimizer=optimizer,
                n_iterations=n_iterations,
                n_domain_points=domain_points,
                n_ic_points=ic_points,
                lambda_pde=lambda_pde,
                lambda_ic=lambda_ic,
                device=device
            )
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Store model for later use
            model_obj = model

        elif framework == 'tensorflow':
            try:
                model = PINN_tf(
                    hidden_layers=hidden_layers,
                    activation=activation,
                    adaptive=adaptive
                )
                
                # Create optimizer
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                
                # Print model info
                print(f"Model: TensorFlow PINN with {activation} activation ({'adaptive' if adaptive else 'fixed'})")
                print(f"Hidden layers: {hidden_layers}")
                
                # Train model
                print(f"\nTraining for {n_iterations} iterations...")
                start_time = time.time()
                
                history = train_model_tf(
                    model=model,
                    optimizer=optimizer,
                    n_iterations=n_iterations,
                    n_domain_points=domain_points,
                    n_ic_points=ic_points,
                    lambda_pde=lambda_pde,
                    lambda_ic=lambda_ic
                )
                
                training_time = time.time() - start_time
                print(f"Training completed in {training_time:.2f} seconds")
                
                model_obj = model
            
            except Exception as e:
                print(f"Error training TensorFlow model: {e}")
                import traceback
                traceback.print_exc()
                # Create dummy history if training fails
                history = {
                    'total_loss': [1.0] * n_iterations,
                    'pde_loss': [1.0] * n_iterations,
                    'ic_loss': [1.0] * n_iterations,
                    'time': list(range(n_iterations))
                }
                training_time = 0.0
                model_obj = None
        
        else:  # JAX
            try:
                # Create the JAX model
                model = create_pinn_model(
                    hidden_layers=hidden_layers,
                    activation=activation,
                    adaptive=adaptive
                )
                
                # Initialize model with a dummy input
                rng_key = jax.random.PRNGKey(seed)
                dummy_input = jnp.ones((1, 2))
                params = model.init(rng_key, dummy_input)
                
                # Print model info
                print(f"Model: JAX PINN with {activation} activation ({'adaptive' if adaptive else 'fixed'})")
                print(f"Hidden layers: {hidden_layers}")
                
                # Train model
                print(f"\nTraining for {n_iterations} iterations...")
                start_time = time.time()
                
                history, trained_params = train_model_jax(
                    model=model,
                    params=params,
                    n_iterations=n_iterations,
                    n_domain_points=domain_points,
                    n_ic_points=ic_points,
                    learning_rate=learning_rate,
                    lambda_pde=lambda_pde,
                    lambda_ic=lambda_ic
                )
                
                training_time = time.time() - start_time
                print(f"Training completed in {training_time:.2f} seconds")
                
                # Store model and params
                model_obj = (model, trained_params)
                
            except Exception as e:
                print(f"Error training JAX model: {e}")
                import traceback
                traceback.print_exc()
                # Create dummy history if training fails
                history = {
                    'total_loss': [1.0] * n_iterations,
                    'pde_loss': [1.0] * n_iterations,
                    'ic_loss': [1.0] * n_iterations,
                    'time': list(range(n_iterations))
                }
                training_time = 0.0
                model_obj = None
        
        # Print final metrics
        final_pde_loss = history['pde_loss'][-1]
        final_ic_loss = history['ic_loss'][-1]
        final_total_loss = history['total_loss'][-1]
        
        print("\nFinal Metrics:")
        print(f"PDE Loss: {final_pde_loss:.6e}")
        print(f"IC Loss: {final_ic_loss:.6e}")
        print(f"Total Loss: {final_total_loss:.6e}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        # Store results
        results[config['name']] = {
            'model': model_obj,
            'history': history,
            'training_time': training_time,
            'final_pde_loss': final_pde_loss,
            'final_ic_loss': final_ic_loss,
            'final_total_loss': final_total_loss,
            'framework': framework
        }
    
    # Create visualizations for Task 1, 2, and overall analysis
    print("\nCreating visualizations...")
    create_task1_visualizations(results, save_dir, device)
    create_task2_visualizations(results, save_dir)
    create_overall_analysis(results, save_dir)
    
    # Print summary table
    print("\nTraining Summary:")
    print(f"{'Configuration':<25} {'Total Loss':<15} {'PDE Loss':<15} {'IC Loss':<15} {'Time (s)':<10}")
    print('-' * 80)
    for config_name, result in results.items():
        print(f"{config_name:<25} {result['final_total_loss']:.6e} {result['final_pde_loss']:.6e} {result['final_ic_loss']:.6e} {result['training_time']:<10.2f}")
    
    print("\nVisualizations saved to directory:", save_dir)

if __name__ == "__main__":
    main()