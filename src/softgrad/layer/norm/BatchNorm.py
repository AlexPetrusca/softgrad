import mlx.core as mx
from softgrad.layer import TrainableLayer


class BatchNorm(TrainableLayer):
    def __init__(self, momentum=0.1, eps=1e-5):
        """
        Batch Normalization layer.

        Normalizes inputs across the batch dimension, learning scale (gamma)
        and shift (beta) parameters.

        Args:
            momentum: Momentum for running mean/var updates (default: 0.1)
            eps: Small constant for numerical stability (default: 1e-5)
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps

    def _link(self):
        # For inputs of shape (H, W, C) or (C,), normalize over all dims except last
        # Number of features is the last dimension
        if len(self.input_shape) == 1:
            # 1D: (features,)
            num_features = self.input_shape[0]
        elif len(self.input_shape) == 3:
            # 2D: (H, W, channels)
            num_features = self.input_shape[2]
        else:
            raise ValueError(f"BatchNorm only supports 1D or 3D inputs, got {len(self.input_shape)}D")

        self.num_features = num_features
        self.output_shape = self.input_shape

        # Learnable parameters: scale (gamma) and shift (beta)
        self.params["gamma"] = mx.ones((num_features,))
        self.params["beta"] = mx.zeros((num_features,))

        # Running statistics (not learnable, updated during training)
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def _forward(self, x_in: mx.array) -> mx.array:
        if self.training:
            # Training mode: use batch statistics
            if len(self.input_shape) == 1:
                # 1D input: (batch, features)
                # Normalize over batch dimension
                axes = (0,)
            else:
                # 3D input: (batch, H, W, C)
                # Normalize over batch and spatial dimensions
                axes = (0, 1, 2)

            # Compute batch mean and variance
            mean = mx.mean(x_in, axis=axes, keepdims=True)
            var = mx.var(x_in, axis=axes, keepdims=True)

            # Normalize
            x_norm = (x_in - mean) / mx.sqrt(var + self.eps)

            # Scale and shift
            y = self.params["gamma"] * x_norm + self.params["beta"]

            # Update running statistics (exponential moving average)
            # Squeeze to remove singleton dimensions for storage
            mean_squeezed = mx.squeeze(mean)
            var_squeezed = mx.squeeze(var)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_squeezed
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_squeezed

            # Save for backward pass
            self.ctx['x_norm'] = x_norm
            self.ctx['mean'] = mean
            self.ctx['var'] = var
            self.ctx['std'] = mx.sqrt(var + self.eps)

        else:
            # Inference mode: use running statistics
            mean = self.running_mean
            var = self.running_var

            # Normalize using running stats
            x_norm = (x_in - mean) / mx.sqrt(var + self.eps)

            # Scale and shift
            y = self.params["gamma"] * x_norm + self.params["beta"]

        return y

    def _backward(self, dx_out: mx.array) -> mx.array:
        x_norm = self.ctx['x_norm']
        mean = self.ctx['mean']
        var = self.ctx['var']
        std = self.ctx['std']
        x_in = self.ctx.x_in

        batch_size = x_in.shape[0]

        # Determine normalization axes
        if len(self.input_shape) == 1:
            axes = (0,)
            N = batch_size
        else:
            axes = (0, 1, 2)
            N = batch_size * self.input_shape[0] * self.input_shape[1]

        # Gradient w.r.t. gamma
        self.params["dgamma"] = mx.sum(dx_out * x_norm, axis=axes)

        # Gradient w.r.t. beta
        self.params["dbeta"] = mx.sum(dx_out, axis=axes)

        # Gradient w.r.t. input (complex!)
        # Using the chain rule through the normalization

        # dx_norm = gradient w.r.t. normalized input
        dx_norm = dx_out * self.params["gamma"]

        # Gradient through normalization (batch-coupled)
        dx_centered = dx_norm / std

        dvar = mx.sum(dx_norm * (x_in - mean) * -0.5 * mx.power(var + self.eps, -1.5), axis=axes, keepdims=True)

        dmean = mx.sum(dx_norm * -1.0 / std, axis=axes, keepdims=True) + \
                dvar * mx.sum(-2.0 * (x_in - mean), axis=axes, keepdims=True) / N

        dx_in = dx_centered + dvar * 2.0 * (x_in - mean) / N + dmean / N

        return dx_in

    def train(self):
        """Set layer to training mode."""
        self.training = True

    def eval(self):
        """Set layer to evaluation mode."""
        self.training = False