import math
from mlx import core as mx
from softgrad.layer import Layer


class MaxPool2d(Layer):
    def __init__(self, kernel_size: tuple | int):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

    def _link(self) -> None:
        if len(self.input_shape) != 3:
            raise ValueError("Input shape must be 3 dimensional (H, W, C).")

        h_in, w_in, c = self.input_shape
        h_out = h_in // self.kernel_size[0]
        w_out = w_in // self.kernel_size[1]
        self.output_shape = (h_out, w_out, c)

    def _forward(self, x_in: mx.array) -> mx.array:
        batch_size = x_in.shape[0]
        h_out, w_out, c = self.output_shape
        kh, kw = self.kernel_size

        x_out = mx.zeros((batch_size, h_out, w_out, c))
        max_indices = mx.zeros((batch_size, h_out * w_out, c), dtype=mx.int32)

        for y in range(h_out):
            for x in range(w_out):
                h_start = y * kh
                h_end = h_start + kh
                w_start = x * kw
                w_end = w_start + kw

                # Extract window and find max
                window = x_in[:, h_start:h_end, w_start:w_end, :]
                x_out[:, y, x, :] = mx.max(window, axis=(1, 2))

                # Store flattened indices of max values
                window_flat = window.reshape(batch_size, -1, c)
                max_idx = mx.argmax(window_flat, axis=1)
                max_indices[:, y * w_out + x, :] = max_idx

        self.ctx['max_indices'] = max_indices
        return x_out

    def _backward(self, dx_out: mx.array) -> mx.array:
        dx_in = mx.zeros(self.ctx.x_in.shape)
        max_indices = self.ctx['max_indices']

        batch_size, h_in, w_in, c = self.ctx.x_in.shape
        _, h_out, w_out, _ = dx_out.shape
        kh, kw = self.kernel_size

        for y in range(h_out):
            for x in range(w_out):
                h_start = y * kh
                h_end = h_start + kh
                w_start = x * kw
                w_end = w_start + kw

                # Get max indices for this output position
                indices = max_indices[:, y * w_out + x, :, mx.newaxis]

                # Create mask indicating where the max value was
                flat_mask = mx.where(indices == mx.arange(kh * kw), 1, 0)
                mask = flat_mask.reshape(batch_size, kh, kw, c)

                # Route gradient to max position
                grad = dx_out[:, y, mx.newaxis, x, mx.newaxis, :]
                dx_in[:, h_start:h_end, w_start:w_end, :] += grad * mask

        return dx_in