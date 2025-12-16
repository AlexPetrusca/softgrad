import mlx.core as mx
from softgrad.layer import Layer
from softgrad.layer.core import Linear

class MultiHeadAttention(Layer):
    """
    Multiple heads of causal self-attention in parallel.
    """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout=0.0):
        """
        Args:
            n_embd: Embedding dimension
            num_heads: Number of attention heads
            head_size: Dimension per head
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.n_embd = n_embd
        self.num_heads = num_heads
        self.head_size = head_size

        # Create multiple attention heads
        self.heads = [
            CausalSelfAttentionHead(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ]

        # Output projection
        self.proj = Linear(n_embd, bias=False)
        self.proj_dropout = dropout

    def get_trainable_layers(self):
        """Collect trainable layers from all heads and projection."""
        trainable = []
        for head in self.heads:
            trainable.extend(head.get_trainable_layers())
        trainable.extend(self.proj.get_trainable_layers())
        return trainable

    def _link(self):
        # Link all heads
        for head in self.heads:
            head.link(self.input_shape)

        # Output shape after concatenating heads
        seq_length = self.input_shape[0]
        concat_size = self.num_heads * self.head_size

        # Link output projection
        self.proj.link((seq_length, concat_size))
        self.output_shape = self.proj.output_shape

    def _forward(self, x_in: mx.array) -> mx.array:
        """
        Forward pass through multi-head attention.

        Args:
            x_in: (batch, seq_length, n_embd)
        Returns:
            out: (batch, seq_length, n_embd)
        """
        # Run all heads in parallel
        head_outputs = [head.forward(x_in) for head in self.heads]

        # Concatenate along last dimension
        # Each: (B, T, head_size) -> concat -> (B, T, num_heads * head_size)
        out = mx.concatenate(head_outputs, axis=-1)

        # Project back to embedding dimension
        out = self.proj.forward(out)

        # Apply dropout to projection
        if self.proj_dropout > 0:
            keep_prob = 1 - self.proj_dropout
            mask = mx.random.bernoulli(keep_prob, out.shape)
            self.ctx['proj_dropout_mask'] = mask
            out = out * mask / keep_prob

        self.ctx['head_outputs'] = head_outputs

        return out

    def _backward(self, dx_out: mx.array) -> mx.array:
        """Backward pass through multi-head attention."""

        # Gradient through projection dropout
        if self.proj_dropout > 0 and hasattr(self.ctx, 'proj_dropout_mask'):
            dx_out = dx_out * self.ctx['proj_dropout_mask'] / (1 - self.proj_dropout)

        # Backprop through output projection
        dx_concat = self.proj.backward(dx_out)

        # Split gradient for each head
        head_grads = mx.split(dx_concat, self.num_heads, axis=-1)

        # Backprop through each head
        dx_ins = []
        for head, dh in zip(self.heads, head_grads):
            dx_in = head.backward(dh)
            dx_ins.append(dx_in)

        # Sum gradients from all heads
        dx_total = dx_ins[0]
        for dx in dx_ins[1:]:
            dx_total = dx_total + dx

        return dx_total