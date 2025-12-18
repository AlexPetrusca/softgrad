from datetime import datetime
import math
import time

import mlx.core as mx
import matplotlib.pyplot as plt
import numpy as np
from mlx import nn

from softgrad import Network
from softgrad.function.activation import Relu, Softmax, softmax
from softgrad.function.core import Add, Concatenate
from softgrad.function.loss import CrossEntropyLoss
from softgrad.layer.attn import CausalSelfAttentionHead
from softgrad.layer.core import Parallel, Embedding, Sequential, Linear, Residual, Activation
from softgrad.layer.norm import LayerNorm
from softgrad.layer.shim import MLX
from softgrad.layer.transform.PositionIndices import PositionIndices
from softgrad.optim import SGD


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.xdropout = nn.Dropout(dropout)

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
        wei = self.xdropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = nn.Linear(n_embd, n_embd)
        self.xdropout = nn.Dropout(dropout)

    def __call__(self, x):
        out = mx.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.proj(out)
        out = self.xdropout(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, n_embd)
        self.xdropout = nn.Dropout(dropout)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        x = self.xdropout(x)
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
        self.blocks = [TransformerBlock(n_embd, n_head) for _ in range(n_block)]
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
            return logits

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
        learning_rate=0.001,
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

        for batch_idx in range(num_batches):
            # Get batch
            x_batch, y_batch = get_batch(train_data, block_size, batch_size)

            # Convert targets to one-hot
            y_onehot = mx.eye(tokenizer.vocab_size)[y_batch]

            # Training step
            print(f"{datetime.now()} - Batch {batch_idx}")
            optimizer.step(x_batch, y_onehot)

            # Log progress
            if step % eval_interval == 0:
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
                if step % (eval_interval * 10) == 0:
                    sample = generate_sample(network, tokenizer, block_size)
                    print(f"\nSample generation:\n{sample}\n")

                network.train()

            step += 1

        # Learning rate decay
        # if (epoch + 1) % 30 == 0:
        #     optimizer.eta *= 0.1
        #     print(f"Learning rate decreased to {optimizer.eta}")

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
    n_block = 6  # Number of transformer blocks
    learning_rate = 3e-5 / block_size  # Learning rate
    epochs = 50  # Training epochs
    dropout = 0.2

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
    network.add_layer(MLX(BigramLanguageModel(), dtype=mx.int32))
    print(f"Model built with {vocab_size} vocab size")

    # # Train
    # print("\nStarting training...")
    # trained_network = train(
    #     network=network,
    #     train_data=train_data,
    #     val_data=val_data,
    #     tokenizer=tokenizer,
    #     block_size=block_size,
    #     batch_size=batch_size,
    #     epochs=epochs,
    #     learning_rate=learning_rate,
    #     eval_interval=10,
    #     eval_batches=20
    # )
    #
    # # Final generation
    # print("\n" + "=" * 80)
    # print("Final sample generation:")
    # print("=" * 80)
    # sample = generate_sample(trained_network, tokenizer, block_size, prompt="ROMEO:", max_tokens=500)
    # print(sample)




    optimizer = SGD(eta=learning_rate, momentum=0.9, weight_decay=1e-4)
    loss_fn = CrossEntropyLoss()
    optimizer.bind_network(network)
    optimizer.bind_loss_fn(loss_fn)

    def debug_training_step(network, optimizer, train_data, tokenizer, block_size, batch_size):
        """
        Debug a single training step to see where gradient explosion happens.
        """

        print("\n" + "=" * 70)
        print("GRADIENT EXPLOSION DEBUGGER")
        print("=" * 70)

        # Get a batch
        x_batch, y_batch = get_batch(train_data, block_size, batch_size)
        y_onehot = mx.eye(tokenizer.vocab_size)[y_batch]

        print(f"\n1. INPUT SHAPES:")
        print(f"   x_batch: {x_batch.shape}, dtype: {x_batch.dtype}")
        print(f"   y_batch: {y_batch.shape}, dtype: {y_batch.dtype}")
        print(f"   y_onehot: {y_onehot.shape}, dtype: {y_onehot.dtype}")

        # Forward pass
        print(f"\n2. FORWARD PASS:")
        y_pred = network.forward(x_batch, save_ctx=True)
        mx.eval(y_pred)

        print(f"   y_pred shape: {y_pred.shape}")
        print(f"   y_pred range: [{float(mx.min(y_pred)):.3f}, {float(mx.max(y_pred)):.3f}]")
        print(f"   y_pred mean: {float(mx.mean(y_pred)):.3f}")
        print(f"   y_pred std: {float(mx.std(y_pred)):.3f}")

        # Check for NaN/Inf
        if mx.any(mx.isnan(y_pred)) or mx.any(mx.isinf(y_pred)):
            print("   ⚠️ WARNING: y_pred contains NaN or Inf!")
            return

        # Loss
        print(f"\n3. LOSS COMPUTATION:")
        loss = optimizer.loss_fn.apply(y_pred, y_onehot)
        mx.eval(loss)

        print(f"   loss shape: {loss.shape}")
        print(f"   loss range: [{float(mx.min(loss)):.3f}, {float(mx.max(loss)):.3f}]")
        print(f"   loss mean: {float(mx.mean(loss)):.3f}")
        print(f"   Total loss: {float(mx.sum(loss)):.3f}")

        # Loss derivative
        print(f"\n4. LOSS DERIVATIVE:")
        grad = optimizer.loss_fn.derivative(y_pred, y_onehot)
        mx.eval(grad)

        print(f"   grad shape: {grad.shape}")
        print(f"   grad range: [{float(mx.min(grad)):.3f}, {float(mx.max(grad)):.3f}]")
        print(f"   grad mean: {float(mx.mean(grad)):.3f}")
        print(f"   grad std: {float(mx.std(grad)):.3f}")
        print(f"   grad sum: {float(mx.sum(grad)):.3f}")

        # Check gradient sanity
        # For softmax-cross-entropy, gradients should be in range [-1, 1]
        grad_abs_max = float(mx.max(mx.abs(grad)))
        if grad_abs_max > 1.0:
            print(f"   ⚠️ WARNING: Gradients larger than 1.0! Max abs: {grad_abs_max:.3f}")

        # Backward pass through each layer
        print(f"\n5. BACKWARD PASS (layer by layer):")
        current_grad = grad

        for i, layer in enumerate(reversed(network.layers)):
            layer_idx = len(network.layers) - 1 - i
            layer_name = f"Layer {layer_idx}: {layer.__class__.__name__}"

            print(f"\n   {layer_name}")
            print(f"     Input grad shape: {current_grad.shape}")
            print(f"     Input grad range: [{float(mx.min(current_grad)):.3f}, {float(mx.max(current_grad)):.3f}]")

            # Do backward
            current_grad = layer.backward(current_grad)
            mx.eval(current_grad)

            print(f"     Output grad shape: {current_grad.shape}")
            print(f"     Output grad range: [{float(mx.min(current_grad)):.3f}, {float(mx.max(current_grad)):.3f}]")

            # Check parameter gradients if trainable
            if hasattr(layer, 'get_trainable_layers'):
                trainable = layer.get_trainable_layers()
                if trainable:
                    print(f"     Trainable sublayers: {len(trainable)}")
                    for j, tlayer in enumerate(trainable):
                        for param in tlayer.params:
                            if param.grad is not None:
                                mx.eval(param.grad)
                                grad_norm = float(mx.sqrt(mx.sum(param.grad ** 2)))
                                grad_max = float(mx.max(mx.abs(param.grad)))
                                print(f"       Param '{param.name}': grad_norm={grad_norm:.3f}, grad_max={grad_max:.3f}")

                                if grad_norm > 1000 or grad_max > 100:
                                    print(f"       ⚠️ WARNING: Large gradient detected!")

        # Parameter updates
        print(f"\n6. PARAMETER UPDATES:")
        trainable_layers = []
        for layer in network.layers:
            trainable_layers.extend(layer.get_trainable_layers())

        for i, layer in enumerate(trainable_layers):
            batch_size_actual = layer.ctx.dx_out.shape[0]

            # Check what the effective learning rate is
            eta = optimizer.eta / batch_size_actual

            print(f"\n   Trainable layer {i}:")
            print(f"     dx_out shape: {layer.ctx.dx_out.shape}")
            print(f"     batch_size: {batch_size_actual}")
            print(f"     eta (before scaling): {optimizer.eta}")
            print(f"     eta (after scaling): {eta}")

            for param in layer.params:
                if param.grad is not None:
                    mx.eval(param.grad)
                    grad_norm = float(mx.sqrt(mx.sum(param.grad ** 2)))
                    param_norm = float(mx.sqrt(mx.sum(param.value ** 2)))
                    update_size = eta * grad_norm

                    print(f"       {param.name}:")
                    print(f"         param_norm: {param_norm:.3f}")
                    print(f"         grad_norm: {grad_norm:.3f}")
                    print(f"         update_size: {update_size:.6f}")
                    # print(f"         update/param ratio: {update_size / param_norm:.6f}")

                    if update_size > param_norm * 0.1:
                        print(f"         ⚠️ WARNING: Update is >10% of parameter magnitude!")

        print("\n" + "=" * 70)
        print("DIAGNOSIS:")
        print("=" * 70)

        # Calculate actual gradient scale
        total_elements = batch_size * block_size
        print(f"Total tokens: {total_elements} (batch={batch_size} × seq={block_size})")
        print(f"Current LR scaling: ÷ {batch_size}")
        print(f"Should scale by: ÷ {total_elements}")
        print(f"Missing scale factor: {total_elements / batch_size}")

        print("\nRECOMMENDED FIX:")
        print(f"  learning_rate = {optimizer.eta} / {block_size} = {optimizer.eta / block_size:.2e}")
        print(f"  Or fix SGD to divide by (batch_size × seq_len)")

        print("=" * 70)


    def compare_with_mlx_native(mlx_model, train_data, tokenizer, block_size, batch_size):
        """
        Compare your framework's gradients with native MLX gradients.
        """
        print("\n" + "=" * 70)
        print("COMPARING WITH NATIVE MLX")
        print("=" * 70)

        x_batch, y_batch = get_batch(train_data, block_size, batch_size)

        # Native MLX forward + loss
        print("\n1. Native MLX computation:")
        logits = mlx_model(x_batch)
        mx.eval(logits)
        print(f"   Logits shape: {logits.shape}")
        print(f"   Logits range: [{float(mx.min(logits)):.3f}, {float(mx.max(logits)):.3f}]")

        # Compute loss the MLX way
        from mlx import nn
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = y_batch.reshape(-1)
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')
        print(f"   Loss (mean): {float(loss):.3f}")

        # Get parameter snapshot
        param_snapshot = {}
        for name, param in mlx_model.trainable_parameters().items():
            param_snapshot[name] = mx.array(param)

        # Compute gradients the MLX way
        def loss_fn(model, x, y):
            logits = model(x)
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = y.reshape(-1)
            return nn.losses.cross_entropy(logits_flat, targets_flat, reduction='mean')

        loss_and_grad_fn = nn.value_and_grad(mlx_model, loss_fn)
        loss_val, grads = loss_and_grad_fn(mlx_model, x_batch, y_batch)

        print(f"\n2. Native MLX gradients:")
        print(f"   Loss: {float(loss_val):.3f}")

        # Show some gradient statistics
        for name, grad in list(grads.items())[:3]:  # First 3 params
            if isinstance(grad, mx.array):
                mx.eval(grad)
                grad_norm = float(mx.sqrt(mx.sum(grad ** 2)))
                print(f"   {name}: grad_norm = {grad_norm:.3f}")

        print("\n3. Compare with your framework:")
        print("   Run your framework's backward pass and compare gradient norms")
        print("   They should match!")

        print("=" * 70)


    # Run the debugger
    print("🔍 DEBUGGING GRADIENT EXPLOSION\n")

    # Debug a single step
    debug_training_step(network, optimizer, train_data, tokenizer, block_size, batch_size)

    # Compare with native MLX (if you have the model accessible)
    # mlx_model = network.layers[0].layer  # Get the BigramLanguageModel
    # compare_with_mlx_native(mlx_model, train_data, tokenizer, block_size, batch_size)