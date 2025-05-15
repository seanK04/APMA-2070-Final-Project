# model.py - Fixed JAX implementation

import jax
import jax.numpy as jnp
import haiku as hk
from typing import List, Callable

class AdaptiveActivation(hk.Module):
    """
    Adaptive activation function with trainable parameters for JAX
    """
    def __init__(self, activation_type='tanh', init_value=1.0, name=None):
        super().__init__(name=name)
        self.activation_type = activation_type
        self.init_value = init_value
        
    def __call__(self, x):
        a = hk.get_parameter("a", shape=[], init=hk.initializers.Constant(self.init_value))
        
        if self.activation_type == 'tanh':
            return jnp.tanh(a * x)
        elif self.activation_type == 'sigmoid':
            return 1.0 / (1.0 + jnp.exp(-a * x))
        elif self.activation_type == 'relu':
            return a * jax.nn.relu(x)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")


def get_activation(activation_type='tanh', adaptive=False):
    """Helper function to get activation function or activation module"""
    if adaptive:
        return lambda x, name=None: AdaptiveActivation(activation_type=activation_type, name=name)(x)
    else:
        if activation_type == 'tanh':
            return jnp.tanh
        elif activation_type == 'sigmoid':
            return lambda x: 1.0 / (1.0 + jnp.exp(-x))
        elif activation_type == 'relu':
            return jax.nn.relu
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")


def create_pinn_model(hidden_layers: List[int], activation: str, adaptive: bool):
    """
    Define the PINN network architecture using Haiku
    """
    def build_network():
        """Inner function to create the network structure - no arguments needed"""
        return hk.Sequential([
            hk.nets.MLP(
                output_sizes=hidden_layers + [1],
                activation=get_activation(activation, adaptive),
                activate_final=False
            )
        ])
    
    # Define a clean function signature for network_fn that works with Haiku transform
    def network_fn(x, is_training=True):
        """Function that will be transformed by Haiku with proper signature"""
        network = build_network()
        return network(x)
    
    # Return a transformed network
    return hk.without_apply_rng(hk.transform(network_fn))


def compute_pde_residual(apply_fn, params, x):
    """
    Compute the PDE residual for the Boussinesq equation using JAX automatic differentiation
    
    The normalized Boussinesq equation is:
    ∂²u/∂t² - ∂²u/∂x² - ∂²/∂x²(3u²) + ∂⁴u/∂x⁴ = 0
    """
    # Helper to compute u(x,t)
    def u_fn(x_input):
        return apply_fn(params, x_input, is_training=False)
    
    # Value function (compute u at a given point)
    def value_op(x_input):
        return jnp.squeeze(u_fn(x_input))
    
    # First-order derivatives
    def dx_op(x_input):
        return jax.grad(value_op, argnums=0)(x_input)[0]
    
    def dt_op(x_input):
        return jax.grad(value_op, argnums=0)(x_input)[1]
    
    # Second-order derivatives
    def dxx_op(x_input):
        return jax.grad(dx_op, argnums=0)(x_input)[0]
    
    def dtt_op(x_input):
        return jax.grad(dt_op, argnums=0)(x_input)[1]
    
    # Third and fourth derivatives
    def dxxx_op(x_input):
        return jax.grad(dxx_op, argnums=0)(x_input)[0]
    
    def dxxxx_op(x_input):
        return jax.grad(dxxx_op, argnums=0)(x_input)[0]
    
    # Squared term derivatives
    def u_squared_op(x_input):
        return 3.0 * value_op(x_input)**2
    
    def u_squared_xx_op(x_input):
        u_squared_x = jax.grad(u_squared_op, argnums=0)(x_input)[0]
        return jax.grad(lambda x: jax.grad(u_squared_op, argnums=0)(x)[0], argnums=0)(x_input)[0]
    
    # Vectorized operations over all points
    u_tt = jax.vmap(dtt_op)(x)
    u_xx = jax.vmap(dxx_op)(x)
    u_xxxx = jax.vmap(dxxxx_op)(x)
    u_squared_xx = jax.vmap(u_squared_xx_op)(x)
    
    # Compute PDE residual: ∂²u/∂t² - ∂²u/∂x² - ∂²/∂x²(3u²) + ∂⁴u/∂x⁴ = 0
    residual = u_tt - u_xx - u_squared_xx + u_xxxx
    
    return jnp.expand_dims(residual, axis=1)

def compute_ic_residual(apply_fn, params, x_ic):
    """
    Compute residuals for initial conditions using JAX automatic differentiation
    """
    # Spatial coordinate
    x_coord = x_ic[:, 0:1]
    
    # For the soliton initial condition
    A = 0.5  # Amplitude
    L = 1.0  # Width parameter
    
    # sech(x)^2 = 1/cosh(x)^2
    u_true = A * jnp.square(1.0 / jnp.cosh(x_coord / L))
    
    # Compute u(x,0)
    u_pred = apply_fn(params, x_ic, is_training=False)
    
    # Define a helper function for gradients
    def u_fn(xt):
        return jnp.squeeze(apply_fn(params, jnp.expand_dims(xt, 0), is_training=False))
    
    # Compute time derivative
    u_t = jax.vmap(lambda xt: jax.grad(u_fn, argnums=0)(xt)[1])(x_ic)
    
    # Residuals: u - f(x) and ∂u/∂t - g(x)
    residual_u = u_pred - u_true
    residual_u_t = jnp.expand_dims(u_t, axis=1)  # Since g(x) = 0
    
    return residual_u, residual_u_t