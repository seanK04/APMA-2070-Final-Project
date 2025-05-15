import tensorflow as tf
import numpy as np

class AdaptiveActivation(tf.keras.layers.Layer):
    """
    Adaptive activation function with trainable parameters
    """
    def __init__(self, activation_type='tanh', init_value=1.0, **kwargs):
        super(AdaptiveActivation, self).__init__(**kwargs)
        self.activation_type = activation_type
        self.a = tf.Variable(init_value, dtype=tf.float32, trainable=True)
        
    def call(self, x):
        if self.activation_type == 'tanh':
            return tf.math.tanh(self.a * x)
        elif self.activation_type == 'sigmoid':
            return 1.0 / (1.0 + tf.math.exp(-self.a * x))
        elif self.activation_type == 'relu':
            return self.a * tf.nn.relu(x)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")
            
    def get_config(self):
        config = super(AdaptiveActivation, self).get_config()
        config.update({
            'activation_type': self.activation_type
        })
        return config


class PINN_tf(tf.keras.Model):
    """
    Physics Informed Neural Network for solving the Boussinesq equation
    TensorFlow implementation
    """
    def __init__(self, 
                 hidden_layers=[50, 50, 50, 50], 
                 activation='tanh', 
                 adaptive=False,
                 device=None):  # device parameter included for compatibility but not used in TF
        super(PINN_tf, self).__init__()
        
        self.activation = activation
        self.adaptive = adaptive
        
        # Define layers
        self.layer_list = []
        
        # Input layer: (x, t) -> hidden layers
        self.layer_list.append(tf.keras.layers.Dense(hidden_layers[0], input_shape=(2,)))
        
        # Build activation for first layer
        if adaptive:
            self.layer_list.append(AdaptiveActivation(activation_type=activation))
        else:
            if activation == 'tanh':
                self.layer_list.append(tf.keras.layers.Activation('tanh'))
            elif activation == 'sigmoid':
                self.layer_list.append(tf.keras.layers.Activation('sigmoid'))
            elif activation == 'relu':
                self.layer_list.append(tf.keras.layers.Activation('relu'))
            else:
                raise ValueError(f"Unsupported activation type: {activation}")
        
        # Hidden layers
        for i in range(len(hidden_layers)-1):
            self.layer_list.append(tf.keras.layers.Dense(hidden_layers[i+1]))
            
            # Add activation
            if adaptive:
                self.layer_list.append(AdaptiveActivation(activation_type=activation))
            else:
                if activation == 'tanh':
                    self.layer_list.append(tf.keras.layers.Activation('tanh'))
                elif activation == 'sigmoid':
                    self.layer_list.append(tf.keras.layers.Activation('sigmoid'))
                elif activation == 'relu':
                    self.layer_list.append(tf.keras.layers.Activation('relu'))
                else:
                    raise ValueError(f"Unsupported activation type: {activation}")
        
        # Output layer: final hidden layer -> u
        self.layer_list.append(tf.keras.layers.Dense(1))
        
        # Initialize weights using Xavier/Glorot initialization
        for layer in self.layer_list:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_initializer = tf.keras.initializers.GlorotNormal()
                layer.bias_initializer = tf.keras.initializers.Zeros()
                
    def call(self, x):
        """
        Forward pass through the network
        x: input tensor of shape [batch_size, 2] representing (x, t)
        """
        for layer in self.layer_list:
            x = layer(x)
        return x
    
    def compute_pde_residual(self, x):
        """
        Compute the PDE residual for the Boussinesq equation using automatic differentiation
        
        The normalized Boussinesq equation is:
        ∂²u/∂t² - ∂²u/∂x² - ∂²/∂x²(3u²) + ∂⁴u/∂x⁴ = 0
        """
        # Convert to tensor and ensure we track gradients
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        
        # Use nested gradient tapes for higher-order derivatives
        with tf.GradientTape() as tape_tt:
            tape_tt.watch(x_tensor)
            t = x_tensor[:, 1:2]
            
            with tf.GradientTape() as tape_t:
                tape_t.watch(x_tensor)
                # Forward pass to get u
                u = self(x_tensor)
                
            # First derivatives
            grad_u = tape_t.gradient(u, x_tensor)
            u_t = tf.expand_dims(grad_u[:, 1], axis=1)  # du/dt
            
        # Second time derivative
        grad_u_t = tape_tt.gradient(u_t, x_tensor)
        u_tt = tf.expand_dims(grad_u_t[:, 1], axis=1)  # d²u/dt²
        
        # For spatial derivatives
        with tf.GradientTape() as tape_xxxx:
            tape_xxxx.watch(x_tensor)
            with tf.GradientTape() as tape_xxx:
                tape_xxx.watch(x_tensor)
                with tf.GradientTape() as tape_xx:
                    tape_xx.watch(x_tensor)
                    with tf.GradientTape() as tape_x:
                        tape_x.watch(x_tensor)
                        # Forward pass to get u
                        u = self(x_tensor)
                    
                    # First derivative: du/dx
                    grad_u = tape_x.gradient(u, x_tensor)
                    u_x = tf.expand_dims(grad_u[:, 0], axis=1)
                
                # Second derivative: d²u/dx²
                grad_u_x = tape_xx.gradient(u_x, x_tensor)
                u_xx = tf.expand_dims(grad_u_x[:, 0], axis=1)
            
            # Third derivative: d³u/dx³
            grad_u_xx = tape_xxx.gradient(u_xx, x_tensor)
            u_xxx = tf.expand_dims(grad_u_xx[:, 0], axis=1)
        
        # Fourth derivative: d⁴u/dx⁴
        grad_u_xxx = tape_xxxx.gradient(u_xxx, x_tensor)
        u_xxxx = tf.expand_dims(grad_u_xxx[:, 0], axis=1)
        
        # Compute derivative of 3u²
        with tf.GradientTape() as tape_squared_xx:
            tape_squared_xx.watch(x_tensor)
            with tf.GradientTape() as tape_squared_x:
                tape_squared_x.watch(x_tensor)
                # Compute 3u²
                u = self(x_tensor)
                u_squared = 3.0 * tf.square(u)
            
            # First derivative of 3u²
            grad_u_squared = tape_squared_x.gradient(u_squared, x_tensor)
            u_squared_x = tf.expand_dims(grad_u_squared[:, 0], axis=1)
        
        # Second derivative of 3u²
        grad_u_squared_x = tape_squared_xx.gradient(u_squared_x, x_tensor)
        u_squared_xx = tf.expand_dims(grad_u_squared_x[:, 0], axis=1)
        
        # Compute PDE residual: ∂²u/∂t² - ∂²u/∂x² - ∂²/∂x²(3u²) + ∂⁴u/∂x⁴ = 0
        residual = u_tt - u_xx - u_squared_xx + u_xxxx
        
        return residual

    def compute_ic_residual(self, x_ic):
        """
        Compute residuals for initial conditions using automatic differentiation
        """
        x_ic_tensor = tf.convert_to_tensor(x_ic, dtype=tf.float32)
        
        # Spatial coordinate
        x_coord = x_ic_tensor[:, 0:1]
        
        # For the soliton initial condition
        A = 0.5  # Amplitude
        L = 1.0  # Width parameter
        
        # sech(x)^2 = 1/cosh(x)^2
        u_true = A * tf.square(1.0 / tf.cosh(x_coord / L))
        
        # Compute u(x,0)
        u_pred = self(x_ic_tensor)
        
        # Compute derivative with respect to time using gradient tape
        with tf.GradientTape() as tape:
            tape.watch(x_ic_tensor)
            u_pred_for_grad = self(x_ic_tensor)
        
        grad_u = tape.gradient(u_pred_for_grad, x_ic_tensor)
        u_t = tf.expand_dims(grad_u[:, 1], axis=1)  # Extract du/dt
        
        # Residuals: u - f(x) and ∂u/∂t - g(x)
        residual_u = u_pred - u_true
        residual_u_t = u_t  # Since g(x) = 0
        
        return residual_u, residual_u_t