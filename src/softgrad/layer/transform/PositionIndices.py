import mlx.core as mx
from softgrad.layer import Layer


class PositionIndices(Layer):
    def __init__(self):
        super().__init__()

    def _link(self):
        if len(self.input_shape) != 1:
            raise ValueError(f"Expected 1D input shape (seq_length,), got {self.input_shape}")

        self.seq_length = self.input_shape[0]
        self.output_shape = self.input_shape  # Same shape as input

        # Pre-compute position indices
        self.positions = mx.arange(self.seq_length)

    def _forward(self, x_in: mx.array) -> mx.array:
        batch_size = x_in.shape[0]

        # Broadcast positions to batch dimension
        # (seq_length,) → (1, seq_length) → (batch, seq_length)
        positions = mx.expand_dims(self.positions, axis=0)
        positions = mx.broadcast_to(positions, (batch_size, self.seq_length))

        return positions

    def _backward(self, dx_out: mx.array) -> mx.array:
        # No gradient flows to input (positions are independent of input)
        return mx.zeros(dx_out.shape, dtype=mx.float32)