import mlx.core as mx
from softgrad.layer import RecursiveLayer, Layer


class Residual(RecursiveLayer):
    """
    Residual connection: output = input + F(input)

    The wrapped layer F must preserve the input shape.
    """

    def __init__(self, layer: Layer):
        super().__init__()
        self.layers = [layer]

    def _link(self):
        # Link the child layer with the same input shape
        self.layers[0].link(self.input_shape)

        # Verify that output shape matches input shape (required for residual)
        if self.layers[0].output_shape != self.input_shape:
            raise ValueError(
                f"Residual connection requires input and output shapes to match. "
                f"Input: {self.input_shape}, Layer output: {self.layers[0].output_shape}"
            )

        # Output shape is same as input shape
        self.output_shape = self.input_shape

    def _forward(self, x_in: mx.array) -> mx.array:
        # Forward: y = x + F(x)
        return x_in + self.layers[0].forward(x_in)

    def _backward(self, dx_out: mx.array) -> mx.array:
        # Backward: dL/dx = dL/dy * (1 + dF/dx)
        #                 = dL/dy + dL/dy * dF/dx
        #                 = dL/dy + child.backward(dL/dy)
        return dx_out + self.layers[0].backward(dx_out)

