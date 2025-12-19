# Test convergence with simple bigram model

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from softgrad import Network
from softgrad.function import Function
from softgrad.function.loss import sequence_ce_loss
from softgrad.layer.shim import MLX
from softgrad.optim import SGD

mx.random.seed(1337)

# ----------------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------------
batch_size = 32
block_size = 128
max_iters = 1000
eval_interval = 50
learning_rate = 1e-2
eval_iters = 50
n_embd = 128

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

class SimpleBigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def __call__(self, idx):
        B, T = idx.shape

        # get embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        x = mx.mean(tok_emb, axis=1)  # (B, n_embd)

        # pass through mlp
        logits = self.lm_head(x)  # (B, vocab_size)
        logits = mx.broadcast_to(logits[:, None, :], (B, T, vocab_size))  # (B, T, vocab_size)

        return logits

network = Network(input_shape=(block_size,))
network.add_layer(MLX(SimpleBigramModel(), dtype=mx.int32))

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
print(f"\nTop {top_k}:")
for idx in top_indices:
    char = itos[int(idx)]
    prob = probs[int(idx)].item()
    print(f"  '{char}' : {prob:.4f}")


# step    0: train loss 4.1722, val loss 4.1716
# step   50: train loss 4.1667, val loss 4.1661
# step  100: train loss 4.1598, val loss 4.1596
# step  150: train loss 4.1530, val loss 4.1528
# step  200: train loss 4.1462, val loss 4.1466
# step  250: train loss 4.1395, val loss 4.1400
# step  300: train loss 4.1326, val loss 4.1335
# step  350: train loss 4.1261, val loss 4.1268
# step  400: train loss 4.1193, val loss 4.1204
# step  450: train loss 4.1127, val loss 4.1140
# step  500: train loss 4.1065, val loss 4.1076
# step  550: train loss 4.0997, val loss 4.1017
# step  600: train loss 4.0937, val loss 4.0954
# step  650: train loss 4.0869, val loss 4.0885
# step  700: train loss 4.0805, val loss 4.0834
# step  750: train loss 4.0744, val loss 4.0769
# step  800: train loss 4.0680, val loss 4.0702
# step  850: train loss 4.0614, val loss 4.0640
# step  900: train loss 4.0555, val loss 4.0589
# step  950: train loss 4.0498, val loss 4.0520
# Final: train loss 4.0441, val loss 4.0456
#
# Top 5:
#   ' ' : 0.0238
#   't' : 0.0188
#   'e' : 0.0186
#   'i' : 0.0180
#   'f' : 0.0174