import torch
import torch.nn as nn
import numpy as np

class AdaptiveActivation(nn.Module):
    """
    Adaptive activation function with trainable parameters
    """
    def __init__(self, activation_type='tanh', init_value=1.0):
        super(AdaptiveActivation, self).__init__()
        self.activation_type = activation_type
        self.a = nn.Parameter(torch.tensor(init_value))
        
    def forward(self, x):
        if self.activation_type == 'tanh':
            return torch.tanh(self.a * x)
        elif self.activation_type == 'sigmoid':
            return 1.0 / (1.0 + torch.exp(-self.a * x))
        elif self.activation_type == 'relu':
            return self.a * torch.relu(x)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")


class PINN_pt(nn.Module):
    """
    Physics Informed Neural Network for solving the Boussinesq equation
    """
    def __init__(self, 
                 hidden_layers=[50, 50, 50, 50], 
                 activation='tanh', 
                 adaptive=False,
                 device='cpu'):
        super(PINN_pt, self).__init__()
        
        self.device = device
        self.activation = activation
        self.adaptive = adaptive
        
        # Input layer: (x, t) -> hidden layers
        self.input_layer = nn.Linear(2, hidden_layers[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers)-1):
            self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Output layer: final hidden layer -> u
        self.output_layer = nn.Linear(hidden_layers[-1], 1)
        
        # Initialize activation functions
        self.activation_functions = nn.ModuleList()
        for _ in range(len(hidden_layers)):
            if adaptive:
                self.activation_functions.append(AdaptiveActivation(activation_type=activation))
            else:
                if activation == 'tanh':
                    self.activation_functions.append(nn.Tanh())
                elif activation == 'sigmoid':
                    self.activation_functions.append(nn.Sigmoid())
                elif activation == 'relu':
                    self.activation_functions.append(nn.ReLU())
                else:
                    raise ValueError(f"Unsupported activation type: {activation}")
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x):
        """
        Forward pass through the network
        x: input tensor of shape [batch_size, 2] representing (x, t)
        """
        x = self.input_layer(x)
        x = self.activation_functions[0](x)
        
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation_functions[i+1](x)
            
        x = self.output_layer(x)
        return x
    
    def compute_pde_residual(self, x):
        """
        Compute the PDE residual for the Boussinesq equation
        
        The normalized Boussinesq equation is:
        ∂²u/∂t² - ∂²u/∂x² - ∂²/∂x²(3u²) + ∂⁴u/∂x⁴ = 0
        
        x: input tensor of shape [batch_size, 2] representing (x, t)
        """
        x = x.clone().detach().requires_grad_(True)
        
        # Get x and t
        x_coord = x[:, 0:1]
        t = x[:, 1:2]
        
        # Forward pass to get u
        u = self.forward(x)
        
        # Create a dummy variable that combines u, x, and t to ensure proper dependency
        dummy = u.sum()
        
        # Compute first-order derivatives
        grad_u = torch.autograd.grad(
            dummy, x, 
            create_graph=True
        )[0]
        
        u_x = grad_u[:, 0:1]
        u_t = grad_u[:, 1:2]
        
        # Compute second-order derivatives
        # For ∂²u/∂t²
        u_tt = torch.autograd.grad(
            u_t.sum(), x,
            create_graph=True
        )[0][:, 1:2]
        
        # For ∂²u/∂x²
        u_xx = torch.autograd.grad(
            u_x.sum(), x,
            create_graph=True
        )[0][:, 0:1]
        
        # For the term ∂²/∂x²(3u²)
        u_squared = 3 * (u ** 2)
        
        # Compute gradient of u_squared with respect to x
        u_squared_grad = torch.autograd.grad(
            u_squared.sum(), x,
            create_graph=True
        )[0]
        
        u_squared_x = u_squared_grad[:, 0:1]
        
        # Second derivative of u_squared with respect to x
        u_squared_xx = torch.autograd.grad(
            u_squared_x.sum(), x,
            create_graph=True
        )[0][:, 0:1]
        
        # For the term ∂⁴u/∂x⁴, we compute it step by step
        # Third derivative
        u_xxx = torch.autograd.grad(
            u_xx.sum(), x,
            create_graph=True
        )[0][:, 0:1]
        
        # Fourth derivative
        u_xxxx = torch.autograd.grad(
            u_xxx.sum(), x,
            create_graph=True
        )[0][:, 0:1]
        
        # Compute residual: ∂²u/∂t² - ∂²u/∂x² - ∂²/∂x²(3u²) + ∂⁴u/∂x⁴ = 0
        residual = u_tt - u_xx - u_squared_xx + u_xxxx
        
        return residual
        
    def compute_detailed_pde_residuals(self, x):
        """
        Compute the individual terms of the PDE residual for the Boussinesq equation
        
        Returns:
            tuple: (u_tt, u_xx, u_squared_xx, u_xxxx) - The individual terms
        """
        x = x.clone().detach().requires_grad_(True)
        
        # Get x and t
        x_coord = x[:, 0:1]
        t = x[:, 1:2]
        
        # Forward pass to get u
        u = self.forward(x)
        
        # Create a dummy variable that combines u, x, and t for proper dependency
        dummy = u.sum()
        
        # Compute first-order derivatives
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
        
        return u_tt, u_xx, u_squared_xx, u_xxxx
    
    def compute_ic_residual(self, x_ic):
        """
        Compute residuals for initial conditions
        Assuming initial conditions: u(x,0) = f(x) and ∂u/∂t(x,0) = g(x)
        where f and g are problem-specific functions
        
        For the Boussinesq soliton, we'll use:
        f(x) = A * sech²(x/L)
        g(x) = 0
        """
        x_ic = x_ic.clone().detach().requires_grad_(True)
        
        x_coord = x_ic[:, 0:1]
        t = x_ic[:, 1:2]  # should be all zeros
        
        # Predict u
        u_pred = self.forward(x_ic)
        
        # For the soliton initial condition
        A = 0.5  # Amplitude
        L = 1.0  # Width parameter
        u_true = A * (1 / torch.cosh(x_coord / L))**2
        
        # Compute derivative with respect to the input
        grad_u = torch.autograd.grad(
            u_pred.sum(), x_ic, 
            create_graph=True
        )[0]
        
        # Extract ∂u/∂t (derivative with respect to t)
        u_t = grad_u[:, 1:2]
        
        # Residuals: u - f(x) and ∂u/∂t - g(x)
        residual_u = u_pred - u_true
        residual_u_t = u_t  # Since g(x) = 0
        
        return residual_u, residual_u_t