from typing import Iterable

import mlx.core as mx
from softgrad.layer import RecursiveLayer
from softgrad.layer.Layer import Layer


class Residual(RecursiveLayer):
    def __init__(self, layer: Layer, residual_dim: Iterable[int]):
        super().__init__()
        self.children.append(layer)
        self.input_shape = residual_dim
        self.output_shape = residual_dim

    def _link(self):
        pass

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in + self.children[0].forward(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return dx_out + self.children[0].backward(dx_out)

