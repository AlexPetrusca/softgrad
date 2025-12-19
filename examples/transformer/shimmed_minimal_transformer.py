# Test convergence with simplest possible transformer model

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from softgrad import Network
from softgrad.function.loss import sequence_ce_loss
from softgrad.layer.shim import MLX
from softgrad.optim import SGD

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


# ============================================================================
# MODEL - Returns ONLY logits (not loss)
# ============================================================================
# ============================================================================
# MINIMAL TRANSFORMER MODEL
# ============================================================================

class CausalSelfAttention(nn.Module):
    """Single head of causal self-attention"""

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Compute attention scores
        wei = (q @ k.transpose(0, 2, 1)) * (self.head_size ** -0.5)  # (B, T, T)

        # Causal mask
        mask = mx.tril(mx.ones((T, T)))
        wei = mx.where(mask == 0, float('-inf'), wei)
        wei = mx.softmax(wei, axis=-1)

        # Apply attention to values
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = [CausalSelfAttention(n_embd, head_size) for _ in range(n_head)]
        self.proj = nn.Linear(n_embd, n_embd)

    def __call__(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = [
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        ]

    def __call__(self, x):
        for layer in self.net:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def __call__(self, x):
        # Pre-norm architecture (more stable)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MinimalTransformer(nn.Module):
    """Minimal transformer for language modeling"""

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]

        # Output
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape

        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(mx.arange(T))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits  # Only return logits!


# ============================================================================
# SETUP NETWORK AND OPTIMIZER
# ============================================================================
print("Setting up network...")
network = Network(input_shape=(block_size,))
network.add_layer(MLX(MinimalTransformer(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size
), dtype=mx.int32))


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
# SANITY CHECK: Compare with random baseline
# ============================================================================
print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

# Random baseline: uniform distribution over vocab
random_loss = -mx.log(mx.array(1.0 / vocab_size)).item()
print(f"Random baseline loss: {random_loss:.4f}")

if losses['val'] < random_loss * 0.9:
    print("✓ Model is learning! (Val loss < 90% of random baseline)")
else:
    print("✗ Model may not be learning properly")

# ============================================================================
# SIMPLE GENERATION TEST
# ============================================================================
print("\n" + "=" * 60)
print("GENERATION TEST")
print("=" * 60)

# Generate a small sample to see if it makes sense
context = mx.zeros((1, block_size), dtype=mx.int32)
logits = network.forward(context, save_ctx=False)

# Get prediction for last position
last_logits = logits[0, -1, :]
probs = mx.softmax(last_logits, axis=-1)

# Show top-5 predicted characters
top_k = 5
top_indices = mx.argsort(probs)[-top_k:][::-1]
print(f"\nTop {top_k} predicted next characters (from empty context):")
for idx in top_indices:
    char = itos[int(idx)]
    prob = probs[int(idx)].item()
    print(f"  '{char}' : {prob:.4f}")

print("\n" + "=" * 60)
print("If train loss is decreasing, your framework is working!")
print("=" * 60)