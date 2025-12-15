import mlx.core as mx
from softgrad.layer import Layer, TrainableLayer


class LayerNorm(TrainableLayer):
    def __init__(self, eps=1e-5, elementwise_affine=True):
        """
        Layer Normalization layer.

        Normalizes inputs across the feature dimensions for each example independently.
        Unlike BatchNorm, LayerNorm:
        - Normalizes within each example (not across batch)
        - No running statistics (same behavior in train/eval)
        - Better for variable batch sizes and sequential data

        Args:
            eps: Small constant for numerical stability (default: 1e-5)
            elementwise_affine: If True, learns scale and shift params (default: True)
        """
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def _link(self):
        self.output_shape = self.input_shape

        if self.elementwise_affine:
            # Learnable parameters: scale (gamma) and shift (beta)
            # Same shape as the normalized dimensions
            self.params["gamma"] = mx.ones(self.input_shape)
            self.params["beta"] = mx.zeros(self.input_shape)
        else:
            # No learnable parameters
            self.trainable = False

    def _forward(self, x_in: mx.array) -> mx.array:
        # Normalize over all dimensions except batch
        # For input (batch, *), normalize over (1, 2, ..., ndim-1)

        if len(self.input_shape) == 1:
            # 1D: (batch, features) → normalize over features
            axes = (1,)
        elif len(self.input_shape) == 3:
            # 3D: (batch, H, W, C) → normalize over H, W, C
            axes = (1, 2, 3)
        else:
            # General case: normalize over all non-batch dimensions
            axes = tuple(range(1, x_in.ndim))

        # Compute mean and variance for each example
        mean = mx.mean(x_in, axis=axes, keepdims=True)
        var = mx.var(x_in, axis=axes, keepdims=True)

        # Normalize
        x_norm = (x_in - mean) / mx.sqrt(var + self.eps)

        # Scale and shift (if enabled)
        if self.elementwise_affine:
            y = self.params["gamma"] * x_norm + self.params["beta"]
        else:
            y = x_norm

        # Save for backward pass
        self.ctx['x_norm'] = x_norm
        self.ctx['mean'] = mean
        self.ctx['var'] = var
        self.ctx['std'] = mx.sqrt(var + self.eps)
        self.ctx['axes'] = axes

        return y

    def _backward(self, dx_out: mx.array) -> mx.array:
        x_norm = self.ctx['x_norm']
        mean = self.ctx['mean']
        var = self.ctx['var']
        std = self.ctx['std']
        axes = self.ctx['axes']
        x_in = self.ctx.x_in

        # Number of elements being normalized for each example
        N = 1
        for axis in axes:
            N *= x_in.shape[axis]

        if self.elementwise_affine:
            # Gradient w.r.t. gamma
            self.params["dgamma"] = mx.sum(dx_out * x_norm, axis=0)

            # Gradient w.r.t. beta
            self.params["dbeta"] = mx.sum(dx_out, axis=0)

            # dx_norm = gradient w.r.t. normalized input
            dx_norm = dx_out * self.params["gamma"]
        else:
            dx_norm = dx_out

        # Gradient w.r.t. input
        # This is simpler than BatchNorm because each example is normalized independently

        # Gradient through normalization
        dx_centered = dx_norm / std

        dvar = mx.sum(dx_norm * (x_in - mean) * -0.5 * mx.power(var + self.eps, -1.5),
                      axis=axes, keepdims=True)

        dmean = mx.sum(dx_norm * -1.0 / std, axis=axes, keepdims=True) + \
                dvar * mx.sum(-2.0 * (x_in - mean), axis=axes, keepdims=True) / N

        dx_in = dx_centered + dvar * 2.0 * (x_in - mean) / N + dmean / N

        return dx_in