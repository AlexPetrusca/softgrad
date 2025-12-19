import datetime
import math
import time

import mlx.core as mx
import matplotlib.pyplot as plt
import numpy as np
from mlx import nn

from softgrad import Network
from softgrad.function.activation import Relu, Softmax, softmax
from softgrad.function.core import Add, Concatenate
from softgrad.function.loss import CrossEntropyLoss, sequence_ce_loss
from softgrad.layer.attn import CausalSelfAttentionHead
from softgrad.layer.core import Parallel, Embedding, Sequential, Linear, Residual, Activation
from softgrad.layer.norm import LayerNorm
from softgrad.layer.shim import MLX
from softgrad.layer.transform.PositionIndices import PositionIndices
from softgrad.optim import SGD


class MLXCausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_heads = n_head
        self.n_embd = n_embd
        self.causal_mask = MLXCausalSelfAttention.create_additive_causal_mask(block_size, dtype=mx.bfloat16)

        self.query_proj = nn.Linear(self.n_embd, self.n_embd)
        self.key_proj = nn.Linear(self.n_embd, self.n_embd)
        self.value_proj = nn.Linear(self.n_embd, self.n_embd)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)

    def __call__(self, x):
        B, T, C = x.shape
        # calculate query, key, value for all heads
        q = self.query_proj(x) # (B, T, C) -> (B, T, C)
        k = self.key_proj(x) # (B, T, C) -> (B, T, C)
        v = self.value_proj(x) # (B, T, C) -> (B, T, C)

        # reshape query, key, value to batch over n_batches x n_heads
        #   - this way we can compute attention for all heads at once (i.e. multi-head attention) with a single matrix multiply
        #   - nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        q = mx.unflatten(q, -1, (self.n_heads, -1)).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = mx.unflatten(k, -1, (self.n_heads, -1)).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        v = mx.unflatten(v, -1, (self.n_heads, -1)).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)

        # causal flash attention
        scale = math.sqrt(1 / q.shape[-1])
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=self.causal_mask[:T, :T]) # 3x(B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side and project out
        output = output.transpose(0, 2, 1, 3).flatten(-2, -1) # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        return self.out_proj(output) # (B, T, C) -> (B, T, C)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype = mx.float32):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.astype(dtype) * mx.finfo(dtype).min
        return mask


class FeedForward(Sequential):
    def __init__(self, n_embd):
        super().__init__([
            Linear(4 * n_embd),
            Activation(Relu()),
            Linear(n_embd)
        ])


class MultiHeadAttention(Sequential):
    def __init__(self, num_heads, head_size):
        super().__init__([
            Parallel(
                [CausalSelfAttentionHead(n_embd, head_size, block_size) for _ in range(num_heads)]  # heads
            , Concatenate()),
            Linear(n_embd)  # projection
        ])


class TransformerBlock(Sequential):
    def __init__(self, n_embd, n_head):
        super().__init__([
            # communication
            Residual(Sequential([
                LayerNorm(),
                # MultiHeadAttention(n_head, n_embd // n_head)
                MLX(MLXCausalSelfAttention())
            ])),
            # computation
            Residual(Sequential([
                LayerNorm(),
                FeedForward(n_embd)
            ]))
        ])


mx.random.seed(1337)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
batch_size = 32
block_size = 128
max_iters = 2000        # Increased - transformers need more steps
eval_interval = 100     # Less frequent eval
learning_rate = 3e-3    # Lower LR for transformer (more stable)
eval_iters = 50
n_embd = 128
n_head = 4              # 4 heads of size 32 each
n_layer = 2             # 2 transformer blocks (minimal but effective)

# ============================================================================
# DATA LOADING
# ============================================================================
with open('rsc/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = mx.array(encode(text))
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data_split) - block_size, (batch_size,))
    x = mx.stack([data_split[int(i):int(i) + block_size] for i in ix])
    y = mx.stack([data_split[int(i) + 1:int(i) + block_size + 1] for i in ix])
    return x, y


def generate_text(network, start_text="", max_new_tokens=500, temperature=1.0, top_k=None):
    """
    Generate text from the model.

    Args:
        network: Your trained Network
        start_text: Starting prompt (empty string for unconditional generation)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k most likely tokens
    """
    # Encode the starting text
    if start_text:
        context = encode(start_text)
    else:
        context = [0]  # Start with a single token (could be any token)

    context = list(context)  # Make it mutable

    for _ in range(max_new_tokens):
        # Take the last block_size tokens (or pad if shorter)
        if len(context) < block_size:
            # Pad with zeros on the left
            context_padded = [0] * (block_size - len(context)) + context
        else:
            # Take last block_size tokens
            context_padded = context[-block_size:]

        # Convert to array with batch dimension
        context_array = mx.array(context_padded)[None, :]  # (1, block_size)

        # Get predictions
        logits = network.forward(context_array, save_ctx=False)  # (1, block_size, vocab_size)

        # Focus on the last time step (the position we're predicting)
        # If we padded, we want the position corresponding to our actual sequence length
        if len(context) < block_size:
            # Prediction is at position len(context) - 1 (due to padding)
            logits = logits[:, len(context) - 1, :]  # (1, vocab_size)
        else:
            # Prediction is at the last position
            logits = logits[:, -1, :]  # (1, vocab_size)

        # Apply temperature
        logits = logits / temperature

        # Optionally crop to top k tokens
        if top_k is not None:
            # Get top k values and indices
            top_values = mx.sort(logits[0])[-top_k:]
            threshold = top_values[0]

            # Mask out tokens below threshold
            logits_filtered = mx.where(logits[0] >= threshold, logits[0], float('-inf'))
            logits = logits_filtered[None, :]  # Add batch dim back

        # Convert to probabilities
        probs = mx.softmax(logits, axis=-1)  # (1, vocab_size)

        # Sample from the distribution
        idx_next = mx.random.categorical(mx.log(probs[0]), num_samples=1)  # (1,)

        # Append to sequence
        context.append(int(idx_next[0]))

    # Decode only the generated tokens (skip the initial context)
    if start_text:
        generated_tokens = context[len(encode(start_text)):]
    else:
        generated_tokens = context[1:]  # Skip the initial [0]

    return decode(generated_tokens)


def generate_with_prompt(network, prompt, num_samples=3, max_new_tokens=200, temperature=0.8):
    """Generate multiple samples from a prompt"""
    print(f"\nPrompt: '{prompt}'")
    print("=" * 80)

    for i in range(num_samples):
        print(f"\n--- Sample {i + 1} ---")
        generated = generate_text(
            network,
            start_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None  # Can set to 40 for more focused sampling
        )
        print(prompt + generated)
        print()


# ============================================================================
# SETUP NETWORK AND OPTIMIZER
# ============================================================================
print("Setting up network...")
network = Network(input_shape=(block_size,))
network.add_layer(Parallel([
    Embedding(vocab_size, n_embd),  # Semantic encoding
    Sequential([
        PositionIndices(),
        Embedding(block_size, n_embd)  # Positional encoding
    ])
], Add()))
network.add_layer(Sequential(
    [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]  # transformer blocks
))
network.add_layer(LayerNorm())
network.add_layer(Linear(vocab_size))  # LLM head

print("Setting up optimizer...")
optimizer = SGD(eta=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer.bind_loss_fn(sequence_ce_loss)
optimizer.bind_network(network)


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================
def estimate_loss():
    """Estimate loss on train and val sets"""
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split)

            # Forward pass
            logits = network.forward(X, save_ctx=False)

            # Compute loss
            loss_per_token = sequence_ce_loss.apply(logits, Y)
            mean_loss = mx.mean(loss_per_token)

            losses.append(mean_loss.item())

        out[split] = np.mean(losses)

    return out


# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\nTraining SimpleBigramModel with your framework...")
print("=" * 60)

for iter in range(max_iters):
    # Evaluate periodically
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Get batch
    xb, yb = get_batch('train')

    # Optimization step (forward + backward + update)
    optimizer.step(xb, yb)

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
losses = estimate_loss()
print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# ============================================================================
# GENERATION TESTS (add after training)
# ============================================================================

print("\n" + "=" * 80)
print("TEXT GENERATION")
print("=" * 80)

# 1. Unconditional generation (from scratch)
print("\n1. UNCONDITIONAL GENERATION (random start)")
print("-" * 80)
text = generate_text(network, start_text="", max_new_tokens=300, temperature=1.0)
print(text)

# 2. Conditional generation with prompts
print("\n2. CONDITIONAL GENERATION (with prompts)")
print("-" * 80)

prompts = [
    "ROMEO:",
    "To be or not to be",
    "First Citizen:\n",
    "The king",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    generated = generate_text(
        network,
        start_text=prompt,
        max_new_tokens=150,
        temperature=0.8,  # Slightly conservative
        top_k=40  # Only sample from top 40 tokens
    )
    print(prompt + generated)
    print()