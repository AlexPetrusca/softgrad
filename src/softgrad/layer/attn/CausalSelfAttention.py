import mlx.core as mx
from softgrad.layer import Layer
from softgrad.layer.core import Linear
from softgrad.function.activation import softmax


class CausalSelfAttention(Layer):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size
        self.block_size = block_size

        self.key = Linear(head_size, bias=False)
        self.query = Linear(head_size, bias=False)
        self.value = Linear(head_size, bias=False)

    def get_trainable_layers(self):
        trainable = []
        trainable.extend(self.key.get_trainable_layers())
        trainable.extend(self.query.get_trainable_layers())
        trainable.extend(self.value.get_trainable_layers())
        return trainable

    def _link(self):
        if len(self.input_shape) != 2:
            raise ValueError(f"Expected input shape (seq_length, n_embd), got {self.input_shape}")

        seq_length, n_embd = self.input_shape
        self.output_shape = (seq_length, self.head_size)

        self.key.link(self.input_shape)
        self.query.link(self.input_shape)
        self.value.link(self.input_shape)

        # causal mask (lower triangular)
        self.tril = mx.tril(mx.ones((self.block_size, self.block_size)))

    def _forward(self, x_in: mx.array) -> mx.array:
        B, T, C = x_in.shape  # (B, T, C)

        # compute Q, K, V
        k = self.key.forward(x_in)  # (B, T, head_size)
        q = self.query.forward(x_in)  # (B, T, head_size)
        v = self.value.forward(x_in)  # (B, T, head_size)

        # compute attention scores (affinities)
        k_transposed = mx.transpose(k, (0, 2, 1))  # (B, head_size, T)
        wei_logits = q @ k_transposed * (self.head_size ** -0.5)  # (B, T, T)

        # apply causal mask
        mask = self.tril[:T, :T]  # (T, T)
        wei_logits = mx.where(mask == 0, float('-inf'), wei_logits)  # (B, T, T)

        # softmax
        wei = self._softmax(wei_logits, axis=-1)  # (B, T, T)

        # compute values
        out = wei @ v  # (B, T, head_size)

        self.ctx['k'] = k
        self.ctx['q'] = q
        self.ctx['v'] = v
        self.ctx['wei_logits'] = wei_logits
        self.ctx['wei'] = wei
        self.ctx['mask'] = mask
        self.ctx['T'] = T

        return out

    def _backward(self, dx_out: mx.array) -> mx.array:
        k = self.ctx['k']
        q = self.ctx['q']
        v = self.ctx['v']
        wei_logits = self.ctx['wei_logits']
        wei = self.ctx['wei']
        mask = self.ctx['mask']
        T = self.ctx['T']
        B = dx_out.shape[0]

        # backprop through value computation: out = wei @ v
        dwei = dx_out @ mx.transpose(v, (0, 2, 1))  # (B, T, T)
        dv = mx.transpose(wei, (0, 2, 1)) @ dx_out  # (B, T, head_size)

        # backprop through softmax
        dwei_logits = self._softmax_backward(dwei, wei)  # (B, T, T)

        # backprop through masking (no gradient for masked)
        dwei_logits = mx.where(mask == 0, 0.0, dwei_logits)

        # backprop through affinity computation: q @ k_transposed * (self.head_size ** -0.5)
        scale = self.head_size ** -0.5
        dwei_logits_scaled = dwei_logits * scale

        # backprop through Q, K, V computation
        dq = dwei_logits_scaled @ k  # (B, T, head_size)
        dk = mx.transpose(dwei_logits_scaled, (0, 2, 1)) @ q  # (B, T, head_size)
        dx_from_q = self.query.backward(dq)
        dx_from_k = self.key.backward(dk)
        dx_from_v = self.value.backward(dv)

        # backprop through entire layer
        dx_in = dx_from_q + dx_from_k + dx_from_v

        return dx_in

    def _softmax(self, x, axis=-1):
        x_max = mx.max(x, axis=axis, keepdims=True)
        exp_x = mx.exp(x - x_max)
        return exp_x / mx.sum(exp_x, axis=axis, keepdims=True)

    def _softmax_backward(self, dy, y):
        sum_term = mx.sum(dy * y, axis=-1, keepdims=True)
        return y * (dy - sum_term)