import mlx.core as mx
import numpy as np

from softgrad import Network
from softgrad.function.activation import Relu
from softgrad.function.core import Add, Concatenate
from softgrad.function.loss import sequence_ce_loss
from softgrad.layer.attn import CausalSelfAttention
from softgrad.layer.core import Parallel, Embedding, Sequential, Linear, Residual, Activation
from softgrad.layer.norm import LayerNorm
from softgrad.layer.transform.PositionIndices import PositionIndices
from softgrad.optim import SGD


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
                [CausalSelfAttention(n_embd, head_size, block_size) for _ in range(num_heads)]  # heads
            , Concatenate()),
            Linear(n_embd)  # projection
        ])


class TransformerBlock(Sequential):
    def __init__(self, n_embd, n_head):
        super().__init__([
            # communication
            Residual(Sequential([
                LayerNorm(),
                MultiHeadAttention(n_head, n_embd // n_head)
            ])),
            # computation
            Residual(Sequential([
                LayerNorm(),
                FeedForward(n_embd)
            ]))
        ])


mx.random.seed(1337)

# ----------------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------------
batch_size = 32
block_size = 2
max_iters = 5000
eval_interval = 100
learning_rate = 3e-2
eval_iters = 50
n_embd = 128            # each token -> 128
n_head = 4              # 4 heads -> 32
n_layer = 2             # 2 transformer blocks

# ----------------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------------

max_num = 100
max_modulo = 20
vocab_size = max_num + 1

X_train = mx.random.randint(0, max_num + 1, shape=(10000,))
Y_train = mx.random.randint(1, max_modulo + 1, shape=(10000,))
Z_train = X_train % Y_train

X_val = mx.random.randint(0, max_num + 1, shape=(1000,))
Y_val = mx.random.randint(1, max_modulo + 1, shape=(1000,))
Z_val = X_val % Y_val

def get_batch(split):
    X_split = X_train if split == 'train' else X_val
    Y_split = Y_train if split == 'train' else Y_val
    Z_split = Z_train if split == 'train' else Z_val

    XY_split = mx.stack([X_split, Y_split], -1)
    n = mx.random.randint(0, len(XY_split) - block_size, (batch_size,))
    return XY_split[n], mx.stack([Z_split[n], Z_split[n]], axis=1)

# ----------------------------------------------------------------------------------
# Setup Network
# ----------------------------------------------------------------------------------
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

optimizer = SGD(eta=learning_rate, momentum=0.9, weight_decay=1e-4)
optimizer.bind_loss_fn(sequence_ce_loss)
optimizer.bind_network(network)


# ----------------------------------------------------------------------------------
# Evaluation function
# ----------------------------------------------------------------------------------
def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split)

            # forward pass
            logits = network.forward(X, save_ctx=False)

            # compute loss
            loss_per_token = sequence_ce_loss.apply(logits, Y)  # (10, 4) -> expect (2, 2)
            mean_loss = mx.mean(loss_per_token)

            losses.append(mean_loss.item())

        out[split] = np.mean(losses)

    return out


# ----------------------------------------------------------------------------------
# Train Loop
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

total = 0
correct = 0
for x in range(max_num + 1):
    for y in range(1, max_modulo + 1):
        logits = network.forward(mx.array([[x, y]], dtype=mx.int32), save_ctx=False)
        max_logit = mx.argmax(logits, axis=-1)[0]

        pred = max_logit[0]
        if (x % y) == int(pred):
            correct += 1
        else:
            print(f"Error: {x} % {y} = {x % y}, not {pred}")

        total += 1

print(f"Accuracy: {100 * correct / total:.2f}%")


# Error: 15 % 12 = 3, not 7
# Error: 35 % 10 = 5, not 7
# Error: 57 % 7 = 1, not 2
# Error: 60 % 12 = 0, not 6
# Error: 64 % 13 = 12, not 13
# Error: 64 % 18 = 10, not 4
# Error: 67 % 7 = 4, not 2
# Error: 81 % 18 = 9, not 6
# Error: 84 % 16 = 4, not 12
# Error: 94 % 19 = 18, not 9
# Error: 95 % 11 = 7, not 4
# Error: 96 % 7 = 5, not 1
# Error: 99 % 13 = 8, not 0
# Accuracy: 99.36%