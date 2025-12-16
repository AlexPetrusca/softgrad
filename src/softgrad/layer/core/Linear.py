import math
import mlx.core as mx
from softgrad.layer import TrainableLayer


# # todo: add Linear function?
# class Linear(TrainableLayer):
#     def __init__(self, output_dims: int, bias: bool = True):
#         super().__init__()
#         self.output_shape = (output_dims,)
#         self.bias = bias
#
#     def _link(self):
#         # Same initialization that PyTorch uses
#         scale = math.sqrt(1.0 / self.input_shape[0])
#         self.params["W"] = mx.random.uniform(-scale, scale, shape=(self.input_shape[0], self.output_shape[0]))
#         if self.bias:
#             self.params["b"] = mx.random.uniform(-scale, scale, shape=(self.output_shape[0],))
#
#     def _forward(self, x_in: mx.array) -> mx.array:
#         res = x_in @ self.params["W"]
#         if self.bias:
#             res += self.params["b"]
#         return res
#
#     def _backward(self, dx_out: mx.array) -> mx.array:
#         self.params["dW"] += self.ctx.x_in.T @ dx_out
#         if self.bias:
#             self.params["db"] += mx.sum(dx_out, axis=0)
#         return dx_out @ self.params["W"].T


import math
import mlx.core as mx
from softgrad.layer import TrainableLayer


class Linear(TrainableLayer):
    def __init__(self, output_dims: int, bias: bool = True):
        super().__init__()
        self.output_dims = output_dims  # Store for use in _link
        self.bias = bias

    def _link(self):
        # Output shape preserves all dims except last: (..., C) → (..., output_dims)
        self.output_shape = self.input_shape[:-1] + (self.output_dims,)

        # Initialize weights based on input dimension (last dim)
        input_dim = self.input_shape[-1]
        scale = math.sqrt(1.0 / input_dim)

        self.params["W"] = mx.random.uniform(
            -scale, scale,
            shape=(input_dim, self.output_dims)
        )

        if self.bias:
            self.params["b"] = mx.random.uniform(
                -scale, scale,
                shape=(self.output_dims,)
            )

    def _forward(self, x_in: mx.array) -> mx.array:
        # Matrix multiply on last dimension
        # (..., input_dim) @ (input_dim, output_dims) → (..., output_dims)
        res = x_in @ self.params["W"]
        if self.bias:
            res += self.params["b"]
        return res

    def _backward(self, dx_out: mx.array) -> mx.array:
        original_shape = self.ctx.x_in.shape
        input_dim = self.input_shape[-1]

        # Flatten all leading dimensions: (..., input_dim) → (N, input_dim)
        x_flat = self.ctx.x_in.reshape(-1, input_dim)
        dx_flat = dx_out.reshape(-1, self.output_dims)

        # Gradient for W: x^T @ dx
        self.params["dW"] += x_flat.T @ dx_flat

        # Gradient for b: sum over all non-feature dimensions
        if self.bias:
            self.params["db"] += mx.sum(dx_flat, axis=0)

        # Gradient for input: dx @ W^T
        dx_in_flat = dx_flat @ self.params["W"].T
        dx_in = dx_in_flat.reshape(original_shape)

        return dx_in