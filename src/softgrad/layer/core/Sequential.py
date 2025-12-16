import mlx.core as mx
from softgrad.layer import RecursiveLayer
from softgrad.layer import Layer

class Sequential(RecursiveLayer):
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.children = layers

    def get_trainable_layers(self):
        trainable = []
        for layer in self.children:
            trainable.extend(layer.get_trainable_layers())
        return trainable

    def _link(self):
        current_shape = self.input_shape
        for layer in self.children:
            layer.link(current_shape)
            current_shape = layer.output_shape
        self.output_shape = current_shape

    def _forward(self, x_in):
        x = x_in
        for layer in self.children:
            x = layer.forward(x)
        return x

    def _backward(self, dx_out):
        dx = dx_out
        for layer in reversed(self.children):
            dx = layer.backward(dx)
        return dx