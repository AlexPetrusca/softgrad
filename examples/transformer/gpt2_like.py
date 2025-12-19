import math
import os

import mlx.core as mx
import numpy as np
import tiktoken
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

# ----------------------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------------------
vocab_size = 50257
batch_size = 64
block_size = 1024
max_iters = 5000
eval_interval = 100
learning_rate = 3e-3
eval_iters = 200
n_embd = 768
n_head = 12
n_layer = 12

# ----------------------------------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = mx.array(npt, dtype=mx.uint32)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "rsc/bookcorpus"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).reshape(B, T) # inputs
        y = (buf[1:]).reshape(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y



# ----------------------------------------------------------------------------------
# Setup Network
# ----------------------------------------------------------------------------------

enc = tiktoken.get_encoding("gpt2")

# data loaders
B = 8  # micro batch size
T = 1024  # sequence length

train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")
# print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")


def generate_with_prompt(network, prompt: str = "Hello, I'm a language model,"):
    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode(prompt)
    tokens = mx.array(tokens, dtype=mx.int32)  # (8 tokens,)
    xgen = mx.repeat(mx.expand_dims(tokens, axis=0), num_return_sequences, axis=0)  # (5 rows, 8 tokens)
    while xgen.shape[1] < max_length:
        # forward the model to get the logits
        logits = network.forward(xgen)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = nn.softmax(logits, axis=-1)

        # get the top k probabilities
        k = 50
        topk_indices = mx.argsort(logits, axis=-1)[:, -k:]
        topk_logits = mx.sort(logits, axis=-1)[:, -k:]

        # select a token from the top probabilities
        ix = mx.random.categorical(topk_logits, num_samples=1)  # (B, 1)
        xcol = mx.take_along_axis(topk_indices, indices=ix, axis=-1)

        xgen = mx.concatenate([xgen, xcol], axis=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"sample {i}: {decoded}")


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


def estimate_loss():
    """Estimate loss on train and val sets"""
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            if split == "train":
                X, Y = train_loader.next_batch()
            else:
                X, Y = val_loader.next_batch()

            # Forward pass
            logits = network.forward(X, save_ctx=False)

            # Compute loss
            loss_per_token = sequence_ce_loss.apply(logits, Y)
            mean_loss = mx.mean(loss_per_token)

            losses.append(mean_loss.item())

        out[split] = np.mean(losses)

    return out


# ----------------------------------------------------------------------------------
# Train Network
# ----------------------------------------------------------------------------------
print("-" * 50)
print("Training...")
print("-" * 50)

for iter in range(max_iters):
    # Evaluate periodically
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if iter % (eval_interval * 5):
            generate_with_prompt(network)

    # Get batch
    print(".", end="")
    xb, yb = train_loader.next_batch()

    # Optimization step (forward + backward + update)
    optimizer.step(xb, yb)


print("-" * 50)
print("Results")
print("-" * 50)

losses = estimate_loss()
print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

generate_with_prompt(network)
generate_with_prompt(network, "First Citizen:\n")
generate_with_prompt(network, "To be or not to be")