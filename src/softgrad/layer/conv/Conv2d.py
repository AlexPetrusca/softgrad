import math
from mlx import core as mx
from operator import mul
from itertools import accumulate
from softgrad.layer import TrainableLayer


def _sliding_windows(x, window_shape, window_strides):
    """Extract sliding windows from input tensor."""
    if x.ndim < 3:
        raise ValueError(
            f"To extract sliding windows at least 1 spatial dimension "
            f"(3 total) is needed but the input only has {x.ndim} dimensions."
        )

    spatial_dims = x.shape[1:-1]
    if not (len(spatial_dims) == len(window_shape) == len(window_strides)):
        raise ValueError(
            f"To extract sliding windows the window shapes and strides must have "
            f"the same number of spatial dimensions as the signal."
        )

    shape = x.shape
    strides = list(reversed(list(accumulate(reversed(shape + (1,)), mul))))[1:]

    # Compute output shape
    final_shape = [shape[0]]
    final_shape += [
        (size - window) // stride + 1
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ]
    final_shape += window_shape
    final_shape += [shape[-1]]

    # Compute output strides
    final_strides = strides[:1]
    final_strides += [
        og_stride * stride for og_stride, stride in zip(strides[1:-1], window_strides)
    ]
    final_strides += strides[1:-1]
    final_strides += strides[-1:]

    return mx.as_strided(x, final_shape, final_strides)


class Conv2d(TrainableLayer):
    # Added padding parameter with default 0
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple | int, padding: int | tuple = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        # Store padding as tuple
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.stride = (1, 1)

    def _link(self) -> None:
        if len(self.input_shape) != 3:
            raise ValueError("Input shape must be 3 dimensional (H, W, C).")

        h_in, w_in, c_in = self.input_shape
        if c_in != self.in_channels:
            raise ValueError(f"Input channels {c_in} doesn't match in_channels {self.in_channels}")

        kh, kw = self.kernel_size
        # Account for padding in output size calculation
        h_out = h_in - kh + 1 + 2 * self.padding[0]
        w_out = w_in - kw + 1 + 2 * self.padding[1]
        self.output_shape = (h_out, w_out, self.out_channels)

        # Initialize weights and biases
        fan_in = kh * kw * self.in_channels
        scale = math.sqrt(1.0 / fan_in)

        self.params["W"] = mx.random.uniform(
            -scale, scale,
            shape=(kh, kw, self.in_channels, self.out_channels)
        )
        self.params["b"] = mx.random.uniform(
            -scale, scale,
            shape=(self.out_channels,)
        )

    def _forward(self, x_in: mx.array) -> mx.array:
        # Apply padding before extracting windows
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_in = mx.pad(
                x_in,
                [(0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]), (0, 0)],
                constant_values=0
            )

        # Extract sliding windows: (batch, h_out, w_out, kh, kw, C_in)
        windows = _sliding_windows(x_in, self.kernel_size, self.stride)

        batch_size, h_out, w_out, kh, kw, c_in = windows.shape

        # Reshape windows for matrix multiplication
        windows_flat = windows.reshape(batch_size, h_out, w_out, -1)

        # Reshape weights: (kh, kw, C_in, C_out) -> (kh*kw*C_in, C_out)
        W_flat = self.params["W"].reshape(-1, self.out_channels)

        # Perform convolution as batched matrix multiplication
        x_out = windows_flat @ W_flat + self.params["b"]

        # Store for backward pass
        self.ctx['windows_flat'] = windows_flat

        return x_out

    def _backward(self, dx_out: mx.array) -> mx.array:
        windows_flat = self.ctx['windows_flat']

        batch_size, h_in, w_in, c_in = self.ctx.x_in.shape
        _, h_out, w_out, c_out = dx_out.shape
        kh, kw = self.kernel_size

        # Gradient for weights
        windows_reshaped = windows_flat.reshape(-1, kh * kw * c_in)
        dx_out_reshaped = dx_out.reshape(-1, c_out)

        # dW = windows^T @ dx_out
        dW_flat = windows_reshaped.T @ dx_out_reshaped
        self.params["dW"] += dW_flat.reshape(kh, kw, c_in, c_out)

        # Gradient for bias
        self.params["db"] += mx.sum(dx_out, axis=(0, 1, 2))

        # Gradient for input
        W_flat = self.params["W"].reshape(-1, self.out_channels)
        dx_windows_flat = dx_out @ W_flat.T
        dx_windows = dx_windows_flat.reshape(batch_size, h_out, w_out, kh, kw, c_in)

        # Create gradient array with padded dimensions if padding was used
        if self.padding[0] > 0 or self.padding[1] > 0:
            h_padded = h_in + 2 * self.padding[0]
            w_padded = w_in + 2 * self.padding[1]
            dx_in_padded = mx.zeros((batch_size, h_padded, w_padded, c_in))
        else:
            dx_in_padded = mx.zeros((batch_size, h_in, w_in, c_in))

        # Accumulate gradients back to input positions
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride[0]
                h_end = h_start + kh
                w_start = j * self.stride[1]
                w_end = w_start + kw

                dx_in_padded[:, h_start:h_end, w_start:w_end, :] += dx_windows[:, i, j, :, :, :]

        # Remove padding from gradient if padding was applied
        if self.padding[0] > 0 or self.padding[1] > 0:
            ph, pw = self.padding
            dx_in = dx_in_padded[:, ph:ph + h_in, pw:pw + w_in, :]
        else:
            dx_in = dx_in_padded

        return dx_in