import mlx.core as mx
from softgrad.layer import Layer
from softgrad.layer.core import Linear


class CausalSelfAttentionHead(Layer):
    """
    One head of causal self-attention.

    Input shape: (seq_length, n_embd)
    Output shape: (seq_length, head_size)
    """

    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        """
        Args:
            n_embd: Input embedding dimension
            head_size: Output dimension per head
            block_size: Maximum sequence length (for causal mask)
            dropout: Dropout probability (default: 0.0)
        """
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size
        self.dropout_p = dropout

        # Create Q, K, V projections (no bias)
        self.key = Linear(head_size, bias=False)
        self.query = Linear(head_size, bias=False)
        self.value = Linear(head_size, bias=False)

    def get_trainable_layers(self):
        """Collect trainable layers from Q, K, V projections."""
        trainable = []
        trainable.extend(self.key.get_trainable_layers())
        trainable.extend(self.query.get_trainable_layers())
        trainable.extend(self.value.get_trainable_layers())
        return trainable

    def _link(self):
        if len(self.input_shape) != 2:
            raise ValueError(f"Expected input shape (seq_length, n_embd), got {self.input_shape}")

        seq_length, n_embd = self.input_shape

        if n_embd != self.n_embd:
            raise ValueError(f"Input embedding dim {n_embd} doesn't match expected {self.n_embd}")

        self.output_shape = (seq_length, self.head_size)

        # Link Q, K, V projections
        self.key.link(self.input_shape)
        self.query.link(self.input_shape)
        self.value.link(self.input_shape)

        # Create causal mask (lower triangular)
        # This prevents attending to future positions
        self.tril = mx.tril(mx.ones((self.block_size, self.block_size)))

    def _forward(self, x_in: mx.array) -> mx.array:
        """
        Forward pass of causal self-attention.

        Args:
            x_in: (batch, seq_length, n_embd)
        Returns:
            out: (batch, seq_length, head_size)
        """
        B, T, C = x_in.shape

        # Compute Q, K, V
        # Each: (B, T, C) -> (B, T, head_size)
        k = self.key.forward(x_in)  # (B, T, head_size)
        q = self.query.forward(x_in)  # (B, T, head_size)
        v = self.value.forward(x_in)  # (B, T, head_size)

        # Compute attention scores (affinities)
        # Q @ K^T: (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        k_transposed = mx.transpose(k, (0, 2, 1))  # (B, head_size, T)
        wei_logits = q @ k_transposed * (self.head_size ** -0.5)  # (B, T, T)

        # Apply causal mask (prevent attending to future)
        # Use the mask for current sequence length T
        mask = self.tril[:T, :T]  # (T, T)
        wei_logits = mx.where(mask == 0, float('-inf'), wei_logits)

        # Softmax to get attention weights
        wei = self._softmax(wei_logits, axis=-1)  # (B, T, T)

        # Apply dropout
        if self.dropout_p > 0:
            wei = self._dropout(wei, self.dropout_p)

        # Weighted aggregation of values
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = wei @ v

        # Cache for backward pass
        self.ctx['k'] = k
        self.ctx['q'] = q
        self.ctx['v'] = v
        self.ctx['wei_logits'] = wei_logits
        self.ctx['wei'] = wei
        self.ctx['mask'] = mask
        self.ctx['T'] = T

        return out

    def _backward(self, dx_out: mx.array) -> mx.array:
        """
        Backward pass through causal self-attention.

        This is complex because we need to backprop through:
        1. Weighted sum (wei @ v)
        2. Dropout
        3. Softmax
        4. Masking
        5. Scaled dot product (q @ k^T / sqrt(d))
        6. Q, K, V projections
        """
        k = self.ctx['k']
        q = self.ctx['q']
        v = self.ctx['v']
        wei_logits = self.ctx['wei_logits']
        wei = self.ctx['wei']
        mask = self.ctx['mask']
        T = self.ctx['T']
        B = dx_out.shape[0]

        # Gradient through weighted sum: out = wei @ v
        # dL/dwei = dL/dout @ v^T
        # dL/dv = wei^T @ dL/dout
        dwei = dx_out @ mx.transpose(v, (0, 2, 1))  # (B, T, T)
        dv = mx.transpose(wei, (0, 2, 1)) @ dx_out  # (B, T, head_size)

        # Gradient through dropout (if applied)
        if self.dropout_p > 0 and hasattr(self.ctx, 'dropout_mask'):
            dwei = dwei * self.ctx['dropout_mask'] / (1 - self.dropout_p)

        # Gradient through softmax
        # softmax gradient: dL/dx = softmax * (dL/dy - sum(dL/dy * softmax))
        dwei_logits = self._softmax_backward(dwei, wei)  # (B, T, T)

        # Gradient through masking (masked positions have no gradient)
        dwei_logits = mx.where(mask == 0, 0.0, dwei_logits)

        # Gradient through scaled dot product: wei_logits = q @ k^T / sqrt(d)
        scale = self.head_size ** -0.5
        dwei_logits_scaled = dwei_logits * scale

        # dL/dq = dL/dwei_logits @ k
        # dL/dk = dL/dwei_logits^T @ q
        dq = dwei_logits_scaled @ k  # (B, T, head_size)
        dk = mx.transpose(dwei_logits_scaled, (0, 2, 1)) @ q  # (B, T, head_size)

        # Backprop through Q, K, V projections
        dx_from_q = self.query.backward(dq)
        dx_from_k = self.key.backward(dk)
        dx_from_v = self.value.backward(dv)

        # Sum gradients (all came from same input)
        dx_in = dx_from_q + dx_from_k + dx_from_v

        return dx_in

    def _softmax(self, x, axis=-1):
        """Numerically stable softmax."""
        # Subtract max for numerical stability
        x_max = mx.max(x, axis=axis, keepdims=True)
        exp_x = mx.exp(x - x_max)
        return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)

    def _softmax_backward(self, dy, y):
        """
        Gradient of softmax.
        dy: gradient from above
        y: softmax output (cached from forward)
        """
        # Softmax gradient: y * (dy - sum(dy * y))
        sum_term = mx.sum(dy * y, axis=-1, keepdims=True)
        return y * (dy - sum_term)

    def _dropout(self, x, p):
        """Apply dropout during training."""
        # For simplicity, always apply during forward
        # In production, you'd check for training mode
        keep_prob = 1 - p
        mask = mx.random.bernoulli(keep_prob, x.shape)
        self.ctx['dropout_mask'] = mask
        return x * mask / keep_prob