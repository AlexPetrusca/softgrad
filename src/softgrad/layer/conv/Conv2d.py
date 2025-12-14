import math
from mlx import core as mx
from softgrad.layer import Layer


class Conv2d(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple | int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

    def _link(self) -> None:
        if len(self.input_shape) != 3:
            raise ValueError("Input shape must be 3 dimensional (H, W, C).")

        h_in, w_in, c_in = self.input_shape
        if c_in != self.in_channels:
            raise ValueError(f"Input channels {c_in} doesn't match in_channels {self.in_channels}")

        kh, kw = self.kernel_size
        h_out = h_in - kh + 1
        w_out = w_in - kw + 1
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
        batch_size = x_in.shape[0]
        h_out, w_out, _ = self.output_shape
        kh, kw = self.kernel_size

        x_out = mx.zeros((batch_size, h_out, w_out, self.out_channels))

        # Reshape weights for matrix multiplication
        W_reshaped = self.params["W"].reshape(-1, self.out_channels)  # (kh*kw*C_in, C_out)

        for y in range(h_out):
            for x in range(w_out):
                # Extract window
                window = x_in[:, y:y + kh, x:x + kw, :]  # (batch, kh, kw, C_in)
                window_flat = window.reshape(batch_size, -1)  # (batch, kh*kw*C_in)

                # Convolve: window @ weights + bias
                x_out[:, y, x, :] = window_flat @ W_reshaped + self.params["b"]

        return x_out

    def _backward(self, dx_out: mx.array) -> mx.array:
        batch_size, h_in, w_in, c_in = self.ctx.x_in.shape
        _, h_out, w_out, _ = dx_out.shape
        kh, kw = self.kernel_size

        dx_in = mx.zeros(self.ctx.x_in.shape)
        W_reshaped = self.params["W"].reshape(-1, self.out_channels)  # (kh*kw*C_in, C_out)

        for y in range(h_out):
            for x in range(w_out):
                # Extract window from input
                window = self.ctx.x_in[:, y:y + kh, x:x + kw, :]  # (batch, kh, kw, C_in)
                window_flat = window.reshape(batch_size, -1)  # (batch, kh*kw*C_in)
                grad = dx_out[:, y, x, :]  # (batch, C_out)

                # Gradient for weights: window^T @ grad
                dW_flat = window_flat.T @ grad  # (kh*kw*C_in, batch) @ (batch, C_out) = (kh*kw*C_in, C_out)
                dW = dW_flat.reshape(kh, kw, c_in, self.out_channels)
                self.params["dW"] += dW

                # Gradient for bias: sum over batch
                self.params["db"] += mx.sum(grad, axis=0)

                # Gradient for input: grad @ weights^T
                dx_window = grad @ W_reshaped.T  # (batch, C_out) @ (C_out, kh*kw*C_in) = (batch, kh*kw*C_in)
                dx_window_reshaped = dx_window.reshape(batch_size, kh, kw, c_in)
                dx_in[:, y:y + kh, x:x + kw, :] += dx_window_reshaped

        return dx_in