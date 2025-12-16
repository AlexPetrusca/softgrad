import mlx.core as mx
from softgrad.layer import TrainableLayer

class RMSNorm(TrainableLayer):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def _link(self):
        self.output_shape = self.input_shape
        # Learnable scale parameter only (no shift)
        self.params["gamma"] = mx.ones(self.input_shape)

    def _forward(self, x_in: mx.array) -> mx.array:
        # Normalize over all dimensions except batch
        if len(self.input_shape) == 1:
            axes = (1,)
        elif len(self.input_shape) == 3:
            axes = (1, 2, 3)
        else:
            axes = tuple(range(1, x_in.ndim))

        # Compute RMS
        rms = mx.sqrt(mx.mean(x_in ** 2, axis=axes, keepdims=True) + self.eps)

        # Normalize
        x_norm = x_in / rms

        # Scale
        y = self.params["gamma"] * x_norm

        # Save for backward
        self.ctx['x_norm'] = x_norm
        self.ctx['rms'] = rms
        self.ctx['axes'] = axes

        return y

    def _backward(self, dx_out: mx.array) -> mx.array:
        x_norm = self.ctx['x_norm']
        rms = self.ctx['rms']
        axes = self.ctx['axes']
        x_in = self.ctx.x_in

        # Gradient w.r.t. gamma
        self.params["dgamma"] = mx.sum(dx_out * x_norm, axis=0)

        # Gradient w.r.t. input (simpler than LayerNorm)
        dx_norm = dx_out * self.params["gamma"]

        N = 1
        for axis in axes:
            N *= x_in.shape[axis]

        # RMSNorm gradient (no mean term)
        dx_in = dx_norm / rms - x_norm * mx.sum(dx_norm * x_norm, axis=axes, keepdims=True) / N

        return dx_in