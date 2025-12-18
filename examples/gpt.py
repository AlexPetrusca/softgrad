import datetime
import time

import mlx.core as mx
import matplotlib.pyplot as plt
import numpy as np

from softgrad import Network
from softgrad.function.activation import Relu, Softmax, softmax
from softgrad.function.core import Add, Concatenate
from softgrad.function.loss import CrossEntropyLoss
from softgrad.layer.attn import CausalSelfAttentionHead
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


# --------------------------------------------------------------------------
# TRAIN
# --------------------------------------------------------------------------


class CharTokenizer:
    def __init__(self, text):
        # Get unique characters
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {''.join(self.chars[:20])}")

    def encode(self, text):
        """Convert text to list of integers."""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Convert list of integers to text."""
        return ''.join([self.idx_to_char[i] for i in indices])


def viz_history(history, figsize=(6, 4)):
    plt.figure(figsize=figsize, num="Loss Curves")
    plt.plot(history['epoch'], history['train_loss'], 'black', linewidth=2.0)
    plt.plot(history['epoch'], history['test_loss'], 'green', linewidth=2.0)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Loss vs Epoch', fontsize=12)


def load_shakespeare():
    """Download and load tinyshakespeare dataset."""
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    # Download if not exists
    try:
        with open('rsc/tinyshakespeare.txt', 'r') as f:
            text = f.read()
    except FileNotFoundError:
        print("Downloading tinyshakespeare.txt...")
        urllib.request.urlretrieve(url, 'rsc/tinyshakespeare.txt')
        with open('rsc/tinyshakespeare.txt', 'r') as f:
            text = f.read()

    return text


def get_batch(data, block_size, batch_size):
    # Random starting positions
    ix = mx.random.randint(0, len(data) - block_size, (batch_size,))

    # Extract sequences
    x = mx.stack([data[int(i):int(i) + block_size] for i in ix])
    y = mx.stack([data[int(i) + 1:int(i) + block_size + 1] for i in ix])

    return x, y


def train(
        network,
        train_data,
        val_data,
        tokenizer,
        block_size=256,
        batch_size=64,
        epochs=100,
        learning_rate=0.0001,
        eval_interval=10,
        eval_batches=10
):
    # Setup optimizer
    optimizer = SGD(eta=learning_rate, momentum=0.9, weight_decay=1e-4)
    loss_fn = CrossEntropyLoss()
    optimizer.bind_network(network)
    optimizer.bind_loss_fn(loss_fn)

    # Training loop
    step = 0
    start_time = time.time()

    for epoch in range(epochs):
        # Number of batches per epoch
        num_batches = len(train_data) // (block_size * batch_size)

        epoch_loss = 0
        for batch_idx in range(num_batches):
            # Get batch
            x_batch, y_batch = get_batch(train_data, block_size, batch_size)

            # Convert targets to one-hot
            y_onehot = mx.eye(tokenizer.vocab_size)[y_batch]

            # Training step
            optimizer.step(x_batch, y_onehot)

            # Log progress
            # if step % eval_interval == 0:
            if True:
                # Evaluate on validation set
                network.eval()
                val_losses = []

                for _ in range(eval_batches):
                    x_val, y_val = get_batch(val_data, block_size, batch_size)
                    y_val_onehot = mx.eye(tokenizer.vocab_size)[y_val]

                    output = network.forward(x_val, save_ctx=False)
                    val_loss = loss_fn.apply(output, y_val_onehot)
                    val_losses.append(float(mx.mean(val_loss)))

                total_val_loss = sum(val_losses)
                elapsed = time.time() - start_time

                print(f"Step {step} | Epoch {epoch + 1}/{epochs} | "
                      f"Val Loss: {total_val_loss:.4f} | "
                      f"Time: {elapsed:.1f}s")

                # Generate sample
                # if step % (eval_interval * 10) == 0:
                if True:
                    sample = generate_sample(network, tokenizer, block_size)
                    print(f"\nSample generation:\n{sample}\n")

                network.train()

            step += 1

        # Learning rate decay
        if (epoch + 1) % 30 == 0:
            optimizer.eta *= 0.1
            print(f"Learning rate decreased to {optimizer.eta}")

    return network


# --------------------------------------------------------------------------
# GENERATE
# --------------------------------------------------------------------------


def generate(network, idx, max_new_tokens, block_size, temperature=1.0):
    """
    Generate text autoregressively.

    Args:
        network: Trained model
        idx: (batch, seq_length) starting context
        max_new_tokens: Number of tokens to generate
        block_size: Maximum context length
        temperature: Sampling temperature

    Returns:
        idx: (batch, seq_length + max_new_tokens)
    """
    network.eval()

    for _ in range(max_new_tokens):
        # Crop to block_size
        idx_cond = idx[:, -block_size:] if idx.shape[1] > block_size else idx

        # Pad if needed
        if idx_cond.shape[1] < block_size:
            pad_length = block_size - idx_cond.shape[1]
            padding = mx.zeros((idx_cond.shape[0], pad_length), dtype=idx_cond.dtype)
            idx_cond = mx.concatenate([padding, idx_cond], axis=1)

        # Get predictions
        logits = network.forward(idx_cond, save_ctx=False)

        # Focus on last time step
        logits = logits[:, -1, :] / temperature

        # Softmax
        logits = logits - mx.max(logits, axis=-1, keepdims=True)
        probs = mx.exp(logits) / mx.sum(mx.exp(logits), axis=-1, keepdims=True)

        # Sample (simple categorical sampling)
        idx_next = sample_categorical(probs)

        # Append
        idx = mx.concatenate([idx, idx_next], axis=1)

    return idx


def sample_categorical(probs):
    """Sample from categorical distribution."""
    batch_size = probs.shape[0]
    samples = []

    for i in range(batch_size):
        # Cumulative sum
        cumsum = mx.cumsum(probs[i])

        # Random value
        rand_val = float(mx.random.uniform(shape=(1,)))

        # Find index
        idx = int(mx.argmax((cumsum >= rand_val).astype(mx.int32)))
        samples.append(idx)

    return mx.array(samples).reshape(batch_size, 1)


def generate_sample(network, tokenizer, block_size, prompt="ROMEO:", max_tokens=200, temperature=0.8):
    """Generate a text sample."""
    # Encode prompt
    context = tokenizer.encode(prompt)
    context = mx.array([context])

    # Generate
    generated = generate(network, context, max_tokens, block_size, temperature)

    # Decode
    text = tokenizer.decode(generated[0].tolist())

    return text


if __name__ == "__main__":
    # Hyperparameters
    block_size = 256  # Context length
    batch_size = 64  # Batch size
    n_embd = 384  # Embedding dimension
    n_head = 6  # Number of attention heads
    n_block = 3  # Number of transformer blocks
    learning_rate = 3e-4  # Learning rate
    epochs = 50  # Training epochs

    # Load data
    print("Loading data...")
    text = load_shakespeare()

    # Create tokenizer
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    # Encode dataset
    data = mx.array(tokenizer.encode(text))

    # Train/val split (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"Train size: {len(train_data):,} tokens")
    print(f"Val size: {len(val_data):,} tokens")

    # Build model
    print("\nBuilding model...")
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
    print(f"Model built with {vocab_size} vocab size")

    # Train
    print("\nStarting training...")
    trained_network = train(
        network=network,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        eval_interval=100,
        eval_batches=20
    )

    # Final generation
    print("\n" + "=" * 80)
    print("Final sample generation:")
    print("=" * 80)
    sample = generate_sample(trained_network, tokenizer, block_size, prompt="ROMEO:", max_tokens=500)
    print(sample)
