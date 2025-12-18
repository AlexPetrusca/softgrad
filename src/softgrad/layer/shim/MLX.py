from mlx import core as mx
from mlx import nn as nn
from softgrad.layer import TrainableLayer


class MLX(TrainableLayer):
    def __init__(self, layer: nn.Module, dtype = mx.float32):
        super().__init__()
        self.layer: nn.Module = layer
        self.dtype = dtype

        def loss_fn(x_in, dx_out):
            return mx.sum(self.layer(x_in) * dx_out)

        self.value_grad_fn = mx.grad(loss_fn, argnums=0)
        self.param_grad_fn = nn.value_and_grad(self.layer, loss_fn)

    # pull parameter from mlx layer into framework
    def pull_parameter(self, key: str, value: dict | mx.array):
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                self.pull_parameter(f'{key}.{subkey}', subvalue)
        else:
            self.params[key] = value

    # push parameter from framework into mlx layer
    def push_parameter(self, key: str, value: mx.array):
        path = key.split('.')

        cur = self.layer
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value

    def _link(self) -> None:
        x_in = mx.zeros((1, *self.input_shape), dtype=self.dtype)
        x_out = self.layer(x_in)
        self.output_shape = x_out.shape[1:]

        for name, value in self.layer.trainable_parameters().items():
            self.pull_parameter(name, value)

    def _forward(self, x_in: mx.array) -> mx.array:
        for name, param in self.params.items():
            self.push_parameter(name, param.value)
        return self.layer(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        if len(self.params) > 0:
            _, grads = self.param_grad_fn(self.ctx.x_in, dx_out)
            for name, value in grads.items():
                self.pull_parameter(f'd{name}', value)

        return self.value_grad_fn(self.ctx.x_in, dx_out)