# Test convergence with simplest possible model in your framework
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from softgrad import Network
from softgrad.function import Function
from softgrad.function.loss import sequence_ce_loss
from softgrad.layer.shim import MLX
from softgrad.optim import SGD

mx.random.seed(1337)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
batch_size = 32
block_size = 128
max_iters = 1000
eval_interval = 50
learning_rate = 1e-2
eval_iters = 50
n_embd = 128

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
class SimpleBigramModel(nn.Module):
    """Simplest possible model - embed and average, then predict"""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape

        # Embed tokens
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)

        # Average over time (simplest aggregation)
        x = mx.mean(tok_emb, axis=1)  # (B, n_embd)

        # Predict logits
        logits = self.lm_head(x)  # (B, vocab_size)

        # Broadcast to all positions
        logits = mx.broadcast_to(logits[:, None, :], (B, T, vocab_size))

        return logits  # Only return logits!


# ============================================================================
# SETUP NETWORK AND OPTIMIZER
# ============================================================================
print("Setting up network...")
network = Network(input_shape=(block_size,))
network.add_layer(MLX(SimpleBigramModel(), dtype=mx.int32))

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