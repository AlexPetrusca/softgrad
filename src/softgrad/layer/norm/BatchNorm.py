import mlx.core as mx
from softgrad.layer import TrainableLayer


class BatchNorm(TrainableLayer):
    def __init__(self, momentum=0.1, eps=1e-5):
        """
        Batch Normalization - normalizes over batch (and spatial) dimensions.

        Args:
            momentum: Running statistics momentum (default: 0.1)
            eps: Numerical stability constant (default: 1e-5)
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.training = True

    def _link(self):
        # Determine number of features (last dimension)
        if len(self.input_shape) == 1:
            # 1D: (features,)
            num_features = self.input_shape[0]
        elif len(self.input_shape) == 3:
            # 2D: (H, W, channels)
            num_features = self.input_shape[2]
        else:
            raise ValueError(f"BatchNorm expects 1D or 3D input, got {len(self.input_shape)}D")

        self.num_features = num_features
        self.output_shape = self.input_shape

        # Learnable parameters
        self.params["gamma"] = mx.ones((num_features,))
        self.params["beta"] = mx.zeros((num_features,))

        # Running statistics (not trainable)
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def _forward(self, x_in: mx.array) -> mx.array:
        """
        Forward pass.

        Input: (batch, ..., features)
        Output: (batch, ..., features)
        """
        if self.training:
            # Compute statistics over batch (and spatial) dimensions
            if len(self.input_shape) == 1:
                # 1D: normalize over batch
                axes = (0,)
            else:
                # 2D: normalize over batch and spatial
                axes = (0, 1, 2)

            # Compute batch statistics
            mean = mx.mean(x_in, axis=axes, keepdims=True)
            var = mx.var(x_in, axis=axes, keepdims=True)

            # Normalize
            std = mx.sqrt(var + self.eps)
            x_norm = (x_in - mean) / std

            # Scale and shift
            y = self.params["gamma"] * x_norm + self.params["beta"]

            # Update running statistics
            mean_scalar = mx.squeeze(mean)
            var_scalar = mx.squeeze(var)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_scalar
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_scalar
            # mx.eval(self.running_mean)  # todo: uncomment if running out of resources
            # mx.eval(self.running_var)  # todo: uncomment if running out of resources

            # Cache for backward
            self.ctx['x_norm'] = x_norm
            self.ctx['std'] = std
            self.ctx['axes'] = axes

        else:
            # Use running statistics
            x_norm = (x_in - self.running_mean) / mx.sqrt(self.running_var + self.eps)
            y = self.params["gamma"] * x_norm + self.params["beta"]

        return y

    def _backward(self, dy: mx.array) -> mx.array:
        """
        Backward pass through batch normalization.

        The key insight: mean and var depend on ALL samples in batch,
        so gradient must account for this coupling.
        """
        x_norm = self.ctx['x_norm']
        std = self.ctx['std']
        axes = self.ctx['axes']

        # Calculate N (number of elements being normalized per feature)
        batch_size = self.ctx.x_in.shape[0]
        if len(self.input_shape) == 1:
            N = float(batch_size)
        else:
            N = float(batch_size * self.input_shape[0] * self.input_shape[1])

        # Gradients for learnable parameters
        self.params["dgamma"] = mx.sum(dy * x_norm, axis=axes)
        self.params["dbeta"] = mx.sum(dy, axis=axes)

        # Gradient for input
        # Standard BatchNorm backward formula accounting for batch coupling
        dy_scaled = dy * self.params["gamma"]

        # Compute sums needed for gradient
        sum_dy = mx.sum(dy_scaled, axis=axes, keepdims=True)
        sum_dy_xnorm = mx.sum(dy_scaled * x_norm, axis=axes, keepdims=True)

        # Final gradient
        dx = (1.0 / N) / std * (N * dy_scaled - sum_dy - x_norm * sum_dy_xnorm)

        return dx

    def train(self):
        """Enable training mode."""
        self.training = True

    def eval(self):
        """Enable evaluation mode."""
        self.training = False