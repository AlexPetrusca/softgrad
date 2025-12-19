import math
from mlx import core as mx
from softgrad.layer import Layer


# todo: this is used in Conv2d as well, refactor out
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

        # Reshape and transpose to get (batch, h_out, w_out, kh, kw, C)
        reshaped = x.reshape(new_shape)
        # Move window dimensions after output dimensions
        # Current: (batch, h_out, kh, w_out, kw, C)
        # Want: (batch, h_out, w_out, kh, kw, C)
        axes = [0, 1, 3, 2, 4, 5]
        return mx.transpose(reshaped, axes)

    # For overlapping windows, use as_strided (more complex)
    from operator import mul
    from itertools import accumulate

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


# class MaxPool2d(Layer):
#     def __init__(self, kernel_size: tuple | int):
#         super().__init__()
#         if isinstance(kernel_size, int):
#             self.kernel_size = (kernel_size, kernel_size)
#         else:
#             self.kernel_size = kernel_size
#         self.stride = self.kernel_size  # Non-overlapping
#
#     def _link(self) -> None:
#         if len(self.input_shape) != 3:
#             raise ValueError("Input shape must be 3 dimensional (H, W, C).")
#
#         h_in, w_in, c = self.input_shape
#         h_out = h_in // self.kernel_size[0]
#         w_out = w_in // self.kernel_size[1]
#         self.output_shape = (h_out, w_out, c)
#
#     def _forward(self, x_in: mx.array) -> mx.array:
#         # Extract sliding windows: (batch, h_out, w_out, kh, kw, C)
#         windows = _sliding_windows(x_in, self.kernel_size, self.stride)
#
#         # Find max over window dimensions (axes 3, 4)
#         # Result: (batch, h_out, w_out, C)
#         x_out = mx.max(windows, axis=(3, 4))
#
#         # Store windows for backward pass
#         self.ctx['windows'] = windows
#         self.ctx['x_out'] = x_out
#
#         return x_out
#
#     def _backward(self, dx_out: mx.array) -> mx.array:
#         windows = self.ctx['windows']  # (batch, h_out, w_out, kh, kw, C)
#         x_out = self.ctx['x_out']  # (batch, h_out, w_out, C)
#
#         batch_size, h_in, w_in, c = self.ctx.x_in.shape
#         kh, kw = self.kernel_size
#         h_out, w_out = self.output_shape[0], self.output_shape[1]
#
#         # Expand output to match window dimensions
#         # (batch, h_out, w_out, C) -> (batch, h_out, w_out, 1, 1, C)
#         x_out_expanded = x_out[:, :, :, mx.newaxis, mx.newaxis, :]
#
#         # Create mask where window values equal the max
#         # (batch, h_out, w_out, kh, kw, C)
#         mask = (windows == x_out_expanded).astype(mx.float32)
#
#         # Normalize mask (handle ties by dividing gradient equally)
#         mask_sum = mx.sum(mask, axis=(3, 4), keepdims=True)
#         mask = mask / mx.maximum(mask_sum, 1.0)
#
#         # Expand gradient to match window dimensions
#         # (batch, h_out, w_out, C) -> (batch, h_out, w_out, 1, 1, C)
#         dx_out_expanded = dx_out[:, :, :, mx.newaxis, mx.newaxis, :]
#
#         # Apply gradient through mask
#         # (batch, h_out, w_out, kh, kw, C)
#         dx_windows = dx_out_expanded * mask
#
#         # Reshape back to the pooled input dimensions
#         # (batch, h_out, w_out, kh, kw, C) -> (batch, h_out, kh, w_out, kw, C)
#         axes = [0, 1, 3, 2, 4, 5]
#         dx_reshaped = mx.transpose(dx_windows, axes)
#
#         # Calculate the actual pooled dimensions (what was used in forward pass)
#         h_pooled = h_out * kh
#         w_pooled = w_out * kw
#
#         # Flatten to pooled shape
#         dx_pooled = dx_reshaped.reshape(batch_size, h_pooled, w_pooled, c)
#
#         # If input dimensions don't match pooled dimensions, pad with zeros
#         if h_pooled != h_in or w_pooled != w_in:
#             dx_in = mx.zeros((batch_size, h_in, w_in, c))
#             dx_in[:, :h_pooled, :w_pooled, :] = dx_pooled
#         else:
#             dx_in = dx_pooled
#
#         return dx_in


class MaxPool2d(Layer):
    def __init__(self, kernel_size: tuple | int, stride: tuple | int = None):
        """
        2D Max Pooling layer

        Args:
            kernel_size: Size of pooling window (int or tuple of 2 ints)
            stride: Stride of pooling window (int or tuple of 2 ints)
                   If None, defaults to kernel_size (non-overlapping)
        """
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size  # Default: non-overlapping
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

        # Calculate output dimensions with stride
        h_out = (h_in - kh) // sh + 1
        w_out = (w_in - kw) // sw + 1

        self.output_shape = (h_out, w_out, c)

    def _forward(self, x_in: mx.array) -> mx.array:
        # Extract sliding windows: (batch, h_out, w_out, kh, kw, C)
        windows = _sliding_windows(x_in, self.kernel_size, self.stride)

        # Find max over window dimensions (axes 3, 4)
        # Result: (batch, h_out, w_out, C)
        x_out = mx.max(windows, axis=(3, 4))

        # Store windows and max values for backward pass
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

        # Expand max values to match window dimensions
        max_vals_expanded = max_vals[:, :, :, mx.newaxis, mx.newaxis, :]

        # Create mask where window values equal the max
        mask = (windows == max_vals_expanded).astype(mx.float32)

        # Normalize mask (handle ties by dividing gradient equally)
        mask_sum = mx.sum(mask, axis=(3, 4), keepdims=True)
        mask = mask / mx.maximum(mask_sum, 1.0)

        # Expand gradient to match window dimensions
        dx_out_expanded = dx_out[:, :, :, mx.newaxis, mx.newaxis, :]

        # Apply gradient through mask
        dx_windows = dx_out_expanded * mask  # (batch, h_out, w_out, kh, kw, C)

        # Scatter gradients back to input positions
        dx_in = mx.zeros((batch_size, h_in, w_in, c))

        # For each output position, add gradient to corresponding input window
        # This handles overlapping windows by accumulating gradients
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * sh
                w_start = j * sw
                h_end = h_start + kh
                w_end = w_start + kw

                # Accumulate gradients (handles overlaps)
                dx_in[:, h_start:h_end, w_start:w_end, :] += dx_windows[:, i, j, :, :, :]

        return dx_in
