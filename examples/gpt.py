import mlx.core as mx

from softgrad import Network
from softgrad.function.activation import Relu, Softmax
from softgrad.function.core import Add, Concatenate
from softgrad.layer.attn import CausalSelfAttentionHead
from softgrad.layer.core import Parallel, Embedding, Sequential, Linear, Residual, Activation
from softgrad.layer.norm import LayerNorm
from softgrad.layer.transform.PositionIndices import PositionIndices

with open('rsc/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 3e-4  # reduce learning rate
eval_iters = 200
n_embd = 768
n_head = 12  # every head is 64 dimensional
n_block = 3  # number of transformer blocks
dropout = 0.2


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = mx.array(encode(text), mx.int64)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = mx.random.randint(0, len(data) - block_size, (batch_size,))
    x = mx.stack([data[i:i + block_size] for i in ix])
    y = mx.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


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
            ])),
            # computation
            Residual(Sequential([
                LayerNorm(),
                FeedForward(n_embd)
            ]))
        ])


network = Network(input_shape=(block_size,))
network.add_layer(Parallel([
    Embedding(vocab_size, n_embd),  # Semantic encoding
    Sequential([
        PositionIndices(),
        Embedding(block_size, n_embd)  # Positional encoding
    ])
], Add()))
network.add_layer(Sequential(
    [TransformerBlock(n_embd, n_head) for _ in range(n_block)]  # transformer blocks
))
network.add_layer(LayerNorm())
network.add_layer(Linear(vocab_size))  # LLM head


def generate(network, idx, max_new_tokens, block_size, temperature=1.0):
    """
    Generate new tokens autoregressively.

    Args:
        network: The trained language model
        idx: (batch, seq_length) array of token indices
        max_new_tokens: Number of new tokens to generate
        block_size: Maximum context length
        temperature: Sampling temperature (higher = more random)

    Returns:
        idx: (batch, seq_length + max_new_tokens) array with generated tokens
    """
    softmax = Softmax()

    for _ in range(max_new_tokens):
        # Crop to last block_size tokens (context window limit)
        idx_cond = idx[:, -block_size:]

        # Get predictions from model
        # Output: (batch, block_size, vocab_size)
        logits = network.forward(idx_cond, save_ctx=False)

        # Focus only on the last time step
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # Apply softmax with temperature to get probabilities
        probs = softmax.apply(logits, temperature=temperature)  # (batch, vocab_size)

        # Sample from the distribution
        idx_next = sample_categorical(probs)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = mx.concatenate([idx, idx_next], axis=1)  # (batch, seq_length + 1)

    return idx


def sample_categorical(probs):
    """
    Sample from categorical distribution using inverse transform sampling.

    Args:
        probs: (batch, vocab_size) probability distribution

    Returns:
        samples: (batch, 1) sampled indices
    """
    batch_size, vocab_size = probs.shape
    samples = []

    for i in range(batch_size):
        # Generate random value in [0, 1)
        random_val = float(mx.random.uniform(shape=(1,)))

        # Use cumulative sum for sampling
        cumsum = mx.cumsum(probs[i])

        # Find first index where cumsum >= random_val
        # This is the sampled token
        mask = (cumsum >= random_val).astype(mx.int32)
        idx = int(mx.argmax(mask))

        samples.append(idx)

    # Convert to (batch, 1) array
    result = mx.array(samples).reshape(batch_size, 1)
    return result


def prepare_context(token_ids, block_size, pad_token=0):
    """
    Prepare context by padding or truncating to block_size.

    Args:
        token_ids: List or array of token IDs
        block_size: Required sequence length (256)
        pad_token: Token ID to use for padding (default: 0)

    Returns:
        context: (1, block_size) array ready for generation
    """
    token_ids = list(token_ids)

    if len(token_ids) < block_size:
        # Pad with pad_token at the beginning
        padding = [pad_token] * (block_size - len(token_ids))
        token_ids = padding + token_ids
    elif len(token_ids) > block_size:
        # Truncate to last block_size tokens
        token_ids = token_ids[-block_size:]

    # Add batch dimension: (block_size,) -> (1, block_size)
    return mx.array([token_ids])


def generate_with_prompt(network, prompt, encode_fn, decode_fn, max_new_tokens,
                         block_size=256, temperature=1.0):
    """
    Generate text from a prompt string.

    Args:
        network: The trained model
        prompt: String to start generation
        encode_fn: Function to convert string -> token IDs
        decode_fn: Function to convert token IDs -> string
        max_new_tokens: Number of tokens to generate
        block_size: Context window size (256)
        temperature: Sampling temperature

    Returns:
        generated_text: String with prompt + generated tokens
    """
    # Encode prompt
    token_ids = encode_fn(prompt)

    # Prepare context (pad to block_size)
    context = prepare_context(token_ids, block_size)

    # Generate
    output = generate(network, context, max_new_tokens, block_size, temperature)

    # Decode and return
    return decode_fn(output[0].tolist()).lstrip()

generated_text = generate_with_prompt(
    network=network,
    prompt="Hello, my name is",  # Short prompt
    encode_fn=encode,
    decode_fn=decode,
    max_new_tokens=100,
    block_size=256,  # Will be padded to 256
    temperature=0.8
)
print(generated_text)

# for iter in range(max_iters):
#
#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#
#     # sample a batch of data
#     xb, yb = get_batch('train')
#
#     # evaluate the loss
#     logits, loss = m(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
# losses = estimate_loss()
# print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#
# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))