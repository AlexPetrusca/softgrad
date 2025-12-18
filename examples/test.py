# MLX implementation of a GPT-style transformer
# Converted from PyTorch to MLX for Apple Silicon optimization
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

mx.random.seed(1337)

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4
eval_iters = 200
n_embd = 768
n_head = 6
n_transformer_blocks = 6
dropout = 0.2

# Load and prepare data
with open('rsc/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = mx.array(encode(text))
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data) - block_size, (batch_size,))
    x = mx.stack([data[int(i):int(i) + block_size] for i in ix])
    y = mx.stack([data[int(i) + 1:int(i) + block_size + 1] for i in ix])
    return x, y


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores
        wei_logits = (q @ k.transpose(0, 2, 1)) * (self.head_size ** -0.5)

        # Causal mask
        mask = mx.tril(mx.ones((T, T)))
        wei_logits = mx.where(mask[:T, :T] == 0, float('-inf'), wei_logits)

        wei = mx.softmax(wei_logits, axis=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """GPT-style language model"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_transformer_blocks)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(mx.arange(T))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits_flat = logits.reshape(B * T, C)
        targets_flat = targets.reshape(B * T)
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = mx.softmax(logits, axis=-1)

            # Sample from the distribution
            idx_next = mx.random.categorical(mx.log(probs), num_samples=1)
            idx = mx.concatenate([idx, idx_next], axis=1)

        return idx


def loss_fn(model, x, y):
    """Compute loss for a batch"""
    _, loss = model(x, y)
    return loss


def estimate_loss(model):
    """Estimate loss on train and val sets"""
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    return out


# Initialize model and optimizer
model = BigramLanguageModel()
# optimizer = optim.AdamW(learning_rate=learning_rate)
optimizer = optim.SGD(learning_rate=learning_rate, momentum=0.9, weight_decay=1e-4)

# Get loss and gradient function
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    print(f"{datetime.now()} - Batch {iter}")

    # Sample batch
    xb, yb = get_batch('train')

    # Compute loss and gradients
    loss, grads = loss_and_grad_fn(model, xb, yb)

    # Update parameters
    optimizer.update(model, grads)
    mx.eval(model.parameters())

# Final evaluation
losses = estimate_loss(model)
print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# Generate from the model
context = mx.zeros((1, 1), dtype=mx.int32)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))