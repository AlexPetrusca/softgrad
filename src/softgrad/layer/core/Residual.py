import mlx.core as mx
from softgrad.layer import RecursiveLayer, Layer


class Residual(RecursiveLayer):
    def __init__(self, layer: Layer):
        super().__init__()
        self.layers = [layer]

    def _link(self):
        self.layers[0].link(self.input_shape)
        self.output_shape = self.input_shape

    def _forward(self, x_in: mx.array) -> mx.array:
        return x_in + self.layers[0].forward(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        return dx_out + self.layers[0].backward(dx_out)

