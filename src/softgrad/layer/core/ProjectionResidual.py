import mlx.core as mx
from softgrad.layer import Layer, RecursiveLayer
from softgrad.layer.conv import Conv2d
from softgrad.layer.norm import BatchNorm
from softgrad.layer.core import Sequential

class ProjectionResidual(RecursiveLayer):
    """
    Residual connection with projection shortcut for dimension changes.
    Uses 1x1 convolution to match dimensions when needed.
    """

    def __init__(self, main_path: Layer):
        super().__init__()
        self.main_path = main_path
        self.projection = None
        self.layers = [self.main_path]

    def _link(self):
        # Link main path
        self.main_path.link(self.input_shape)

        # Check if dimensions match
        if self.main_path.output_shape != self.input_shape:
            h, w, c_in = self.input_shape
            h_out, w_out, c_out = self.main_path.output_shape

            # Create projection: 1x1 conv + BatchNorm to match dimensions
            self.projection = Sequential([
                Conv2d(c_in, c_out, kernel_size=1, padding=0),  # 1x1 conv
                BatchNorm()
            ])
            self.projection.link(self.input_shape)
            self.layers.append(self.projection)

            # Verify shapes match after projection
            if self.projection.output_shape != self.main_path.output_shape:
                raise ValueError(
                    f"Projection output {self.projection.output_shape} "
                    f"doesn't match main path output {self.main_path.output_shape}"
                )

        self.output_shape = self.main_path.output_shape

    def _forward(self, x_in: mx.array) -> mx.array:
        # Forward through main path
        main_out = self.main_path.forward(x_in)

        # Forward through projection if needed, else use identity
        if self.projection:
            shortcut = self.projection.forward(x_in)
        else:
            shortcut = x_in

        # Add residual connection
        return shortcut + main_out

    def _backward(self, dx_out: mx.array) -> mx.array:
        # Backward through main path
        dx_main = self.main_path.backward(dx_out)

        # Backward through projection if needed, else pass through
        if self.projection:
            dx_shortcut = self.projection.backward(dx_out)
        else:
            dx_shortcut = dx_out

        # Sum gradients
        return dx_main + dx_shortcut