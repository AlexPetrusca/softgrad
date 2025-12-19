import mlx.core as mx
from softgrad.layer import Layer, RecursiveLayer
from softgrad.layer.conv import Conv2d
from softgrad.layer.norm import BatchNorm
from softgrad.layer.core import Sequential

class ProjectionResidual(RecursiveLayer):
    def __init__(self, main_path: Layer):
        super().__init__()
        self.main_path = main_path
        self.projection = None
        self.layers = [self.main_path]

    def _link(self):
        self.main_path.link(self.input_shape)

        # if dimensions don't match, project residual path with a 1x1 convolution
        if self.main_path.output_shape != self.input_shape:
            h, w, c_in = self.input_shape
            h_out, w_out, c_out = self.main_path.output_shape

            self.projection = Sequential([
                Conv2d(c_in, c_out, kernel_size=1, padding=0),  # 1x1 conv
                BatchNorm()
            ])
            self.projection.link(self.input_shape)
            self.layers.append(self.projection)

        self.output_shape = self.main_path.output_shape

    def _forward(self, x_in: mx.array) -> mx.array:
        main_out = self.main_path.forward(x_in)
        if self.projection:
            shortcut = self.projection.forward(x_in)
        else:
            shortcut = x_in
        return shortcut + main_out

    def _backward(self, dx_out: mx.array) -> mx.array:
        # backprop through main path
        dx_main = self.main_path.backward(dx_out)

        # backprop through projection or pass through
        if self.projection:
            dx_shortcut = self.projection.backward(dx_out)
        else:
            dx_shortcut = dx_out

        return dx_main + dx_shortcut