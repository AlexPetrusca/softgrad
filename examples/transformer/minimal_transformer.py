import math

import mlx.core as mx
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


# todo: We need to train with padding tokens, since prompt context is usually much smaller than window
#   - Bug: getting bad generation until the context window is filled up

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
                MultiHeadAttention(n_head, n_embd // n_head)
                # MLX(MLXCausalSelfAttention())
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
block_size = 128
max_iters = 25000
eval_interval = 100
learning_rate = 3e-2
eval_iters = 50
n_embd = 128            # each token -> 128
n_head = 4              # 4 heads -> 32
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


def generate_text(network, start_text="", max_new_tokens=500, temperature=1.0, top_k=None):
    if start_text:
        context = encode(start_text)
    else:
        context = [0]

    context = list(context)

    for _ in range(max_new_tokens):
        if len(context) < block_size:
            context_padded = [0] * (block_size - len(context)) + context  # pad 0s on the left
        else:
            context_padded = context[-block_size:]  # take as much as we can fit into context

        context_array = mx.array(context_padded)[None, :]  # (1, block_size)
        logits = network.forward(context_array, save_ctx=False)  # (1, block_size, vocab_size)

        if len(context) < block_size:
            logits = logits[:, len(context) - 1, :]  # (1, vocab_size)
        else:
            logits = logits[:, -1, :]  # (1, vocab_size)

        logits = logits / temperature

        if top_k is not None:
            top_values = mx.sort(logits[0])[-top_k:]
            threshold = top_values[0]
            logits_filtered = mx.where(logits[0] >= threshold, logits[0], float('-inf'))
            logits = logits_filtered[None, :]

        probs = mx.softmax(logits, axis=-1) # convert to probabilities
        idx_next = mx.random.categorical(mx.log(probs[0]), num_samples=1) # sample from distribution
        context.append(int(idx_next[0]))

    if start_text:
        generated_tokens = context[len(encode(start_text)):]
    else:
        generated_tokens = context[1:]

    return decode(generated_tokens)


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
            loss_per_token = sequence_ce_loss.apply(logits, Y)
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
        temperature=0.8,
        top_k=40
    )
    print(prompt + generated)
    print()


# Starting with: 'First Citizen:\n'
# ICIHAIIITiwhiIhITMIATIISICkOYYhhCFiTTBWiIISAEMJREMSTCipTOsvW::ei:in:ns :yIrhitsIyW:ii:ii idii ie:  tke nt
# tihwh  zectringielth word I spriselflike grat
# ThinD his kin hath bendsech nill onswemby.
# AUT-l
# --------------------------------------------------------------------------------
# Starting with: '\n\n'
# hiiiIIIIEhhhihIBISIASwiIIIIIAIBItiWIhLWLIXWeSiEHITIIYPGOGSEICISTATUCienaTHy:c;j,idwiinaaI:isinnh:y:  :;  hi
#  :a,  aiss ur mnoase waas hatngess
# I gath I all, good try ackiffe,
# So I surd, shald dries
# T
# --------------------------------------------------------------------------------
# Starting with: 'The '
# hNIA:IhIIIwWTiIiTIAITIIhIYSIIIhIwCNCRETCMSGHI:IIIgJIuApTISCVRIIILiSTIfaIAni:w:hUOciIiIn rI: iOntI:t ,ikhshh w
# sc  iaiobkne anyI
# atERncest thy have yenmmoms'Oorok the not that caman
# Musfienfole it su
# --------------------------------------------------------------------------------