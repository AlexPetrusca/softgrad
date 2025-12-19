# Test convergence with tiny transformer model

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from softgrad import Network
from softgrad.function.loss import sequence_ce_loss
from softgrad.layer.shim import MLX
from softgrad.optim import SGD

mx.random.seed(1337)

# ----------------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------------
batch_size = 32
block_size = 128
max_iters = 2000
eval_interval = 100
learning_rate = 3e-3
eval_iters = 50
n_embd = 128            # 128 per token
n_head = 4              # 4 heads (size 32 each)
n_layer = 2             # 2 transformer blocks

# ----------------------------------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------
# Setup Network
# ----------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
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

        # compute attention scores (affinities)
        wei = (q @ k.transpose(0, 2, 1)) * (self.head_size ** -0.5)  # (B, T, T)

        # causal mask
        mask = mx.tril(mx.ones((T, T)))
        wei = mx.where(mask == 0, float('-inf'), wei)
        wei = mx.softmax(wei, axis=-1)

        # compute values
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
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
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape

        # token + positional embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(mx.arange(T))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        # output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits


network = Network(input_shape=(block_size,))
network.add_layer(MLX(MinimalTransformer(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size
), dtype=mx.int32))


optimizer = SGD(eta=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer.bind_loss_fn(sequence_ce_loss)
optimizer.bind_network(network)


def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split)

            logits = network.forward(X, save_ctx=False)
            loss_per_token = sequence_ce_loss.apply(logits, Y)
            mean_loss = mx.mean(loss_per_token)

            losses.append(mean_loss.item())

        out[split] = np.mean(losses)

    return out


# ----------------------------------------------------------------------------------
# Train Network
# ----------------------------------------------------------------------------------
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    optimizer.step(xb, yb)

# ----------------------------------------------------------------------------------
# Final Evaluation
# ----------------------------------------------------------------------------------
losses = estimate_loss()
print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

context = mx.zeros((1, block_size), dtype=mx.int32)
logits = network.forward(context, save_ctx=False)

last_logits = logits[0, -1, :]
probs = mx.softmax(last_logits, axis=-1)

top_k = 5
top_indices = mx.argsort(probs)[-top_k:][::-1]
print(f"Top {top_k}:")
for idx in top_indices:
    char = itos[int(idx)]
    prob = probs[int(idx)].item()
    print(f"  '{char}' : {prob:.4f}")


# step    0: train loss 4.2917, val loss 4.2904
# step  100: train loss 3.5852, val loss 3.6003
# step  200: train loss 3.4603, val loss 3.4879
# step  300: train loss 3.4052, val loss 3.4410
# step  400: train loss 3.3587, val loss 3.3947
# step  500: train loss 3.3354, val loss 3.3690
# step  600: train loss 3.3089, val loss 3.3396
# step  700: train loss 3.2906, val loss 3.3241
# step  800: train loss 3.2688, val loss 3.3063
# step  900: train loss 3.2536, val loss 3.2867
# step 1000: train loss 3.2296, val loss 3.2727
# step 1100: train loss 3.2161, val loss 3.2603
# step 1200: train loss 3.1996, val loss 3.2333
# step 1300: train loss 3.1834, val loss 3.2101
# step 1400: train loss 3.1684, val loss 3.1948
# step 1500: train loss 3.1432, val loss 3.1762
# step 1600: train loss 3.1235, val loss 3.1649
# step 1700: train loss 3.1088, val loss 3.1388
# step 1800: train loss 3.0852, val loss 3.1223
# step 1900: train loss 3.0773, val loss 3.1072
# Final: train loss 3.0541, val loss 3.0869
# Top 5:
#   'i' : 0.0481
#   'T' : 0.0452
#   'C' : 0.0351
#   ' ' : 0.0340
#   'a' : 0.0339
