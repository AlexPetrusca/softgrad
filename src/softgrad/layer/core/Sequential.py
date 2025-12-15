import mlx.core as mx
from softgrad.layer.Layer import Layer

class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.trainable = False

    def get_trainable_layers(self):
        trainable = []
        for layer in self.layers:
            trainable.extend(layer.get_trainable_layers())
        return trainable

    def _link(self):
        current_shape = self.input_shape
        for layer in self.layers:
            layer.link(current_shape)
            current_shape = layer.output_shape
        self.output_shape = current_shape

    def _forward(self, x_in):
        x = x_in
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backward(self, dx_out):
        dx = dx_out
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
        return dx