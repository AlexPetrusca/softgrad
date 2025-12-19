import mlx.core as mx
from softgrad.layer import TrainableLayer


class BatchNorm(TrainableLayer):
    def __init__(self, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

    def _link(self):
        if len(self.input_shape) == 1:
            num_features = self.input_shape[0]  # 1D: (features,)
        elif len(self.input_shape) == 3:
            num_features = self.input_shape[2]  # 2D: (H, W, channels)
        else:
            raise ValueError(f"BatchNorm expects 1D or 3D input, got {len(self.input_shape)}D")

        self.num_features = num_features
        self.output_shape = self.input_shape

        # learnable parameters
        self.params["gamma"] = mx.ones((num_features,))  # variance
        self.params["beta"] = mx.zeros((num_features,))  # mean

        # running statistics (updated via momentum)
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))

    def _forward(self, x_in: mx.array) -> mx.array:
        if self.trainable:
            if len(self.input_shape) == 1:
                axes = (0,)  # 1D: normalize over batch
            else:
                axes = (0, 1, 2)  # 2D: normalize over batch and spatial

            # compute batch statistics
            mean = mx.mean(x_in, axis=axes, keepdims=True)
            var = mx.var(x_in, axis=axes, keepdims=True)

            # normalize (make input unit gaussian)
            std = mx.sqrt(var + self.eps)
            x_norm = (x_in - mean) / std

            # scale and shift
            y = self.params["gamma"] * x_norm + self.params["beta"]

            # update running statistics
            mean_scalar = mx.squeeze(mean)
            var_scalar = mx.squeeze(var)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_scalar
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_scalar
            mx.eval([self.running_mean, self.running_var])  # evaluate to clean up computation graphs ; avoid memory leak

            self.ctx['x_norm'] = x_norm
            self.ctx['std'] = std
            self.ctx['axes'] = axes
        else:
            x_norm = (x_in - self.running_mean) / mx.sqrt(self.running_var + self.eps)
            y = self.params["gamma"] * x_norm + self.params["beta"]

        return y

    def _backward(self, dy: mx.array) -> mx.array:
        x_norm = self.ctx['x_norm']
        std = self.ctx['std']
        axes = self.ctx['axes']

        batch_size = self.ctx.x_in.shape[0]
        if len(self.input_shape) == 1:
            N = float(batch_size)  # 1D: normalize over batch
        else:
            N = float(batch_size * self.input_shape[0] * self.input_shape[1])  # 2D: normalize over batch and spatial

        # gradients for learnable parameters
        self.params["dgamma"] = mx.sum(dy * x_norm, axis=axes)
        self.params["dbeta"] = mx.sum(dy, axis=axes)

        # gradient for input
        dy_scaled = dy * self.params["gamma"]
        sum_dy = mx.sum(dy_scaled, axis=axes, keepdims=True)
        sum_dy_xnorm = mx.sum(dy_scaled * x_norm, axis=axes, keepdims=True)
        dx = (1.0 / N) / std * (N * dy_scaled - sum_dy - x_norm * sum_dy_xnorm)
        return dx