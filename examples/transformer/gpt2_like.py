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

enc = tiktoken.get_encoding("gpt2")

# data loaders
B = 8  # micro batch size
T = 1024  # sequence length

train_loader = DataLoaderLite(B=B, T=T, split="train")
val_loader = DataLoaderLite(B=B, T=T, split="val")

# ----------------------------------------------------------------------------------
# Setup Network
# ----------------------------------------------------------------------------------


def generate_with_prompt(network, prompt: str = "Hello, I'm a language model,"):
    priming_str = """
        A country is an area of land, which has its own government and laws, or used to have them, such as a state, a nation, a nation state, or other political entity. When referring to a specific polity, the term "country" may refer to a sovereign state, a state with limited recognition, a constituent country, or a dependent territory. Most sovereign states, but not all countries, are members of the United Nations.
        Water is an inorganic compound with the chemical formula H2O. It is a transparent, tasteless, odorless, and nearly colorless chemical substance. It is the main constituent of Earth's hydrosphere and the fluids of all known living organisms, in which it acts as a solvent. Water, being a polar molecule, undergoes strong intermolecular hydrogen bonding which is a large contributor to its physical and chemical properties. It is vital for all known forms of life, despite not providing food energy or being an organic micronutrient. Due to its presence in all organisms, its chemical stability, its worldwide abundance, and its strong polarity relative to its small molecular size, water is often referred to as the "universal solvent".
        Earth is the third planet from the Sun and the only astronomical object known to harbor life. This is made possible by Earth being an ocean world, the only one in the Solar System sustaining liquid surface water. Almost all of Earth's water is contained in its global ocean, covering 70.8% of Earth's crust. The remaining 29.2% of Earth's crust is land, most of which is located in the form of continental landmasses within Earth's land hemisphere. Most of Earth's land is at least somewhat humid and covered by vegetation, while large ice sheets at Earth's polar deserts retain more water than Earth's groundwater, lakes, rivers, and atmospheric water combined. Earth's crust consists of slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes. Earth has a liquid outer core that generates a magnetosphere capable of deflecting most of the destructive solar winds and cosmic radiation.
        Earth has a dynamic atmosphere, which sustains Earth's surface conditions and protects it from most meteoroids and UV-light at entry. It is composed primarily of nitrogen and oxygen. Water vapor is widely present in the atmosphere, forming clouds that cover most of the planet. The water vapor acts as a greenhouse gas and, together with other greenhouse gases in the atmosphere, particularly carbon dioxide (CO2), creates the conditions for both liquid surface water and water vapor to persist via the capturing of energy from the Sun's light. This process maintains the current average surface temperature of 14.76 °C (58.57 °F), at which water is liquid under normal atmospheric pressure. Differences in the amount of captured energy between geographic regions (as with the equatorial region receiving more sunlight than the polar regions) drive atmospheric and ocean currents, producing a global climate system with different climate regions, and a range of weather phenomena such as precipitation, allowing components such as carbon and nitrogen to cycle.
        A year is a unit of time based on how long it takes the Earth to orbit the Sun. In scientific use, the tropical year (approximately 365 solar days, 5 hours, 48 minutes, 45 seconds) and the sidereal year (about 20 minutes longer) are more exact. The modern calendar year, as reckoned according to the Gregorian calendar, approximates the tropical year by using a system of leap years.
        The term 'year' is also used to indicate other periods of roughly similar duration, such as the lunar year (a roughly 354-day cycle of twelve of the Moon's phases – see lunar calendar), as well as periods loosely associated with the calendar or astronomical year, such as the seasonal year, the fiscal year, the academic year, etc.
    """

    num_return_sequences = 4
    max_length = 32
    tokens = enc.encode(priming_str + prompt)
    tokens = mx.array(tokens, dtype=mx.int32)  # (8 tokens,)
    xgen = mx.repeat(mx.expand_dims(tokens, axis=0), num_return_sequences, axis=0)  # (5 rows, 8 tokens)
    while xgen.shape[1] < max_length:
        # forward the model to get the logits
        logits = network.forward(xgen)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)

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
        print(f"> {decoded}")


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


def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            if split == "train":
                X, Y = train_loader.next_batch()
            else:
                X, Y = val_loader.next_batch()

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

        if iter % (eval_interval * 5):
            generate_with_prompt(network)

    print(".", end="")
    xb, yb = train_loader.next_batch()
    optimizer.step(xb, yb)

losses = estimate_loss()
print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

for i in range(5):
    generate_with_prompt(network)

# step    0: train loss 4.3638, val loss 4.3552
# step  100: train loss 3.2845, val loss 3.3251
# step  200: train loss 3.1693, val loss 3.2127
# step  300: train loss 2.9192, val loss 2.9483
# step  400: train loss 2.7731, val loss 2.7905
# step  500: train loss 2.6946, val loss 2.7135
# step  600: train loss 2.6488, val loss 2.6597
# step  700: train loss 2.6190, val loss 2.6262
# step  800: train loss 2.5955, val loss 2.6084
# step  900: train loss 2.5808, val loss 2.5860
# step 1000: train loss 2.5655, val loss 2.5707
# step 1100: train loss 2.5494, val loss 2.5558
# step 1200: train loss 2.5418, val loss 2.5430
# step 1300: train loss 2.5304, val loss 2.5376
# step 1400: train loss 2.5214, val loss 2.5270
# step 1500: train loss 2.5118, val loss 2.5235
# step 1600: train loss 2.5060, val loss 2.5156
# step 1700: train loss 2.4980, val loss 2.5044
# step 1800: train loss 2.4966, val loss 2.5014
# step 1900: train loss 2.4848, val loss 2.4934
# step 2000: train loss 2.4827, val loss 2.4896
# step 2100: train loss 2.4764, val loss 2.4818
# step 2200: train loss 2.4720, val loss 2.4810
# step 2300: train loss 2.4688, val loss 2.4778
# step 2400: train loss 2.4673, val loss 2.4747
# step 2500: train loss 2.4595, val loss 2.4679
# step 2600: train loss 2.4565, val loss 2.4703
# step 2700: train loss 2.4538, val loss 2.4581
# step 2800: train loss 2.4462, val loss 2.4557
# step 2900: train loss 2.4424, val loss 2.4587
# step 3000: train loss 2.4425, val loss 2.4536
# step 3100: train loss 2.4392, val loss 2.4467
# step 3200: train loss 2.4357, val loss 2.4511
# step 3300: train loss 2.4327, val loss 2.4466
# step 3400: train loss 2.4253, val loss 2.4429
# step 3500: train loss 2.4272, val loss 2.4433
# step 3600: train loss 2.4177, val loss 2.4322
# step 3700: train loss 2.4170, val loss 2.4339
# step 3800: train loss 2.4104, val loss 2.4262
# step 3900: train loss 2.4084, val loss 2.4276
# step 4000: train loss 2.4086, val loss 2.4222
# step 4100: train loss 2.4028, val loss 2.4198
# step 4200: train loss 2.3997, val loss 2.4177
# step 4300: train loss 2.4003, val loss 2.4191
# step 4400: train loss 2.3968, val loss 2.4123
# step 4500: train loss 2.3921, val loss 2.4095
# step 4600: train loss 2.3900, val loss 2.4105
# step 4700: train loss 2.3890, val loss 2.4100
# step 4800: train loss 2.3860, val loss 2.4037
# step 4900: train loss 2.3777, val loss 2.3932
# Final: train loss 2.3773, val loss 2.3995

# > Hello, I'm a language model, that can be to all his time that they is the water. (K/5.
# A the most
# > Hello, I'm a language model, was more or one, the largest the same energy in the country),
# -based, you have taken that
# > Hello, I'm a language model, a place of the second part. I use. The most commonly found by the water to the National Development and
# > Hello, I'm a language model, this as which and the state, which of any the "the Don. In the two, and the
# > Hello, I'm a language model,,.
# If one). This is it can be the area.
# The word.
# -year water