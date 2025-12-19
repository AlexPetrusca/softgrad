import mlx.core as mx
from softgrad.layer import Layer, TrainableLayer


class LayerNorm(TrainableLayer):
    def __init__(self, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def _link(self):
        self.output_shape = self.input_shape

        if self.elementwise_affine:
            # learnable parameters
            self.params["gamma"] = mx.ones(self.input_shape)  # variance
            self.params["beta"] = mx.zeros(self.input_shape)  # mean
        else:
            self.trainable = False

    def _forward(self, x_in: mx.array) -> mx.array:
        axes = tuple(range(1, x_in.ndim))  # normalize over all non-batch dimensions

        # compute example statistics
        mean = mx.mean(x_in, axis=axes, keepdims=True)
        var = mx.var(x_in, axis=axes, keepdims=True)

        # normalize
        x_norm = (x_in - mean) / mx.sqrt(var + self.eps)

        # scale and shift
        if self.elementwise_affine:
            y = self.params["gamma"] * x_norm + self.params["beta"]
        else:
            y = x_norm

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

        N = 1
        for axis in axes:
            N *= x_in.shape[axis]  # normalize over all non-batch dimensions

        if self.elementwise_affine:
            # gradients for learnable parameters
            self.params["dgamma"] = mx.sum(dx_out * x_norm, axis=0)
            self.params["dbeta"] = mx.sum(dx_out, axis=0)

            # gradient w.r.t. normalized input
            dx_norm = dx_out * self.params["gamma"]
        else:
            dx_norm = dx_out

        # gradient for input
        dx_centered = dx_norm / std
        dvar = mx.sum(dx_norm * (x_in - mean) * -0.5 * mx.power(var + self.eps, -1.5), axis=axes, keepdims=True)
        dmean = mx.sum(dx_norm * -1.0 / std, axis=axes, keepdims=True) + \
                dvar * mx.sum(-2.0 * (x_in - mean), axis=axes, keepdims=True) / N
        dx_in = dx_centered + dvar * 2.0 * (x_in - mean) / N + dmean / N
        return dx_in