import math
from operator import mul
from itertools import accumulate
from mlx import core as mx
from softgrad.layer import Layer


class MaxPool2d(Layer):
    def __init__(self, kernel_size: tuple | int, stride: tuple | int = None):
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

    def _link(self) -> None:
        if len(self.input_shape) != 3:
            raise ValueError("Input shape must be 3 dimensional (H, W, C).")

        h_in, w_in, c = self.input_shape
        kh, kw = self.kernel_size
        sh, sw = self.stride

        h_out = (h_in - kh) // sh + 1
        w_out = (w_in - kw) // sw + 1

        self.output_shape = (h_out, w_out, c)

    def _forward(self, x_in: mx.array) -> mx.array:
        windows = _sliding_windows(x_in, self.kernel_size, self.stride)  # (batch, h_out, w_out, kh, kw, C)

        # Find max over window dimensions
        x_out = mx.max(windows, axis=(3, 4))  # (batch, h_out, w_out, C)

        self.ctx['windows'] = windows
        self.ctx['max_vals'] = x_out

        return x_out

    def _backward(self, dx_out: mx.array) -> mx.array:
        windows = self.ctx['windows']  # (batch, h_out, w_out, kh, kw, C)
        max_vals = self.ctx['max_vals']  # (batch, h_out, w_out, C)
        x_in = self.ctx.x_in  # (batch, h_in, w_in, C)

        batch_size, h_in, w_in, c = x_in.shape
        h_out, w_out = dx_out.shape[1:3]
        kh, kw = self.kernel_size
        sh, sw = self.stride

        # create mask for max values
        max_vals_expanded = max_vals[:, :, :, mx.newaxis, mx.newaxis, :]
        mask = (windows == max_vals_expanded).astype(mx.float32)
        mask_sum = mx.sum(mask, axis=(3, 4), keepdims=True)
        mask = mask / mx.maximum(mask_sum, 1.0)

        # for each window, route gradient through the mask
        dx_out_expanded = dx_out[:, :, :, mx.newaxis, mx.newaxis, :]
        dx_windows = dx_out_expanded * mask  # (batch, h_out, w_out, kh, kw, C)
        dx_in = mx.zeros((batch_size, h_in, w_in, c))
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * sh
                w_start = j * sw
                h_end = h_start + kh
                w_end = w_start + kw
                dx_in[:, h_start:h_end, w_start:w_end, :] += dx_windows[:, i, j, :, :, :]

        return dx_in


# todo: this is used in Conv2d as well, refactor out
# sliding windows in shape: (batch, h_out, w_out, kh, kw, C_in)
#   - batch = batch dimension (per sample)
#   - h_out, w_out = window starting positions (index the window you want)
#   - kh, kw = window elements (index the element you want in the window
#   - C_in = channel dimension (per element)
def _sliding_windows(x, window_shape, window_strides):
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
    # For non-overlapping windows (stride == window_size), use reshape
    if all(
            window == stride and size % window == 0
            for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ):
        batch = shape[0]
        channels = shape[-1]
        new_shape = [batch]
        for size, window in zip(spatial_dims, window_shape):
            new_shape.append(size // window)
            new_shape.append(window)
        new_shape.append(channels)

        reshaped = x.reshape(new_shape)  # (batch, h_out, w_out, kh, kw, C)
        # Move window dimensions after output dimensions
        return mx.transpose(reshaped, [0, 1, 3, 2, 4, 5]) # (batch, h_out, w_out, kh, kw, C)

    # For overlapping windows, use as_strided
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
