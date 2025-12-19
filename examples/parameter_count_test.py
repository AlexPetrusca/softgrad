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


vocab_size = 50257
batch_size = 64
block_size = 512
max_iters = 5000
eval_interval = 100
learning_rate = 3e-3
eval_iters = 200
n_embd = 768
n_head = 6
n_layer = 6


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


# ----------------------------------------------------------------------------------
# PARAMETER COUNTING UTILITIES (FIXED)
# ----------------------------------------------------------------------------------

def count_parameters(network, trainable_only=False):
    """
    Count the total number of parameters in a network.

    Args:
        network: Your Network instance
        trainable_only: If True, only count trainable parameters

    Returns:
        int: Total number of parameters
    """
    total_params = 0
    counted_params = set()  # Track parameter IDs to avoid double counting

    def count_layer_params(layer, check_trainable=False):
        nonlocal total_params

        # If checking trainable, skip frozen layers
        if check_trainable and hasattr(layer, 'trainable') and not layer.trainable:
            return

        # Count this layer's direct parameters
        for param in layer.params:
            if param.value is not None:
                param_id = id(param.value)
                if param_id not in counted_params:
                    counted_params.add(param_id)
                    total_params += param.value.size

        # Recursively count nested layers (for RecursiveLayer)
        if hasattr(layer, 'layers') and layer.layers:
            for nested_layer in layer.layers:
                count_layer_params(nested_layer, check_trainable)

    # Count all layers in the network
    for layer in network.layers:
        count_layer_params(layer, check_trainable=trainable_only)

    return total_params


def print_parameter_summary(network):
    """
    Print a detailed summary of parameters in the network.
    """
    print("\n" + "=" * 80)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 80)

    # Count unique parameters
    all_param_ids = set()
    trainable_param_ids = set()

    def collect_params(layer, is_trainable=True):
        # Check if this layer is frozen
        layer_trainable = is_trainable and (not hasattr(layer, 'trainable') or layer.trainable)

        for param in layer.params:
            if param.value is not None:
                param_id = id(param.value)
                all_param_ids.add(param_id)
                if layer_trainable:
                    trainable_param_ids.add(param_id)

        # Recurse into nested layers
        if hasattr(layer, 'layers') and layer.layers:
            for nested in layer.layers:
                collect_params(nested, layer_trainable)

    # Collect all unique parameters
    for layer in network.layers:
        collect_params(layer)

    # Print layer-by-layer breakdown
    print(f"\n{'Layer Type':<30} {'Direct Params':<15} {'Nested Params':<15} {'Trainable':<12}")
    print("-" * 80)

    def print_layer_info(layer, indent=0):
        layer_name = "  " * indent + layer.__class__.__name__

        # Count direct parameters (not nested)
        direct_params = 0
        param_shapes = []
        for param in layer.params:
            if param.value is not None:
                direct_params += param.value.size
                param_shapes.append(f"{param.name}: {tuple(param.value.shape)}")

        # Count nested parameters
        nested_params = 0
        if hasattr(layer, 'layers') and layer.layers:
            nested_counted = set()
            for nested in layer.layers:
                for param in nested.params:
                    if param.value is not None:
                        param_id = id(param.value)
                        if param_id not in nested_counted:
                            nested_counted.add(param_id)
                            nested_params += param.value.size
                # Recursively count deeper nesting
                if hasattr(nested, 'layers') and nested.layers:
                    def count_deep(l):
                        total = 0
                        for p in l.params:
                            if p.value is not None:
                                pid = id(p.value)
                                if pid not in nested_counted:
                                    nested_counted.add(pid)
                                    total += p.value.size
                        if hasattr(l, 'layers'):
                            for nl in l.layers:
                                total += count_deep(nl)
                        return total

                    nested_params += count_deep(nested)

        is_trainable = not hasattr(layer, 'trainable') or layer.trainable
        trainable_str = "✓" if is_trainable else "✗"

        print(f"{layer_name:<30} {direct_params:>14,} {nested_params:>14,} {trainable_str:>11}")

        # Print parameter shapes if any
        if param_shapes and indent == 0:
            for shape in param_shapes[:3]:
                print(f"{'':>30}   └─ {shape}")

        # Recursively print nested layers
        if hasattr(layer, 'layers') and layer.layers:
            for nested in layer.layers:
                print_layer_info(nested, indent + 1)

    for layer in network.layers:
        print_layer_info(layer)

    # Calculate totals from unique parameter IDs
    total_params = sum(
        p.value.size for layer in network.layers
        for p in get_all_params(layer)
        if p.value is not None
    )

    trainable_params = sum(
        p.value.size for layer in network.layers
        for p in get_all_params(layer, trainable_only=True)
        if p.value is not None
    )

    print("-" * 80)
    print(f"{'Total Parameters:':<30} {total_params:>14,}")
    print(f"{'Trainable Parameters:':<30} {trainable_params:>14,}")
    print(f"{'Non-trainable Parameters:':<30} {(total_params - trainable_params):>14,}")
    print("=" * 80)

    # Memory estimate
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"\nEstimated Memory (float32): {memory_mb:.2f} MB")
    print("=" * 80)


def get_all_params(layer, trainable_only=False):
    """Recursively get all parameters from a layer and its nested layers."""
    params = []
    seen = set()

    def collect(l, is_trainable=True):
        # Check trainability
        layer_trainable = is_trainable and (not hasattr(l, 'trainable') or l.trainable)

        # Skip if we want only trainable and this isn't trainable
        if trainable_only and not layer_trainable:
            return

        # Collect this layer's params
        for p in l.params:
            if p.value is not None:
                param_id = id(p.value)
                if param_id not in seen:
                    seen.add(param_id)
                    params.append(p)

        # Recurse
        if hasattr(l, 'layers') and l.layers:
            for nested in l.layers:
                collect(nested, layer_trainable)

    collect(layer)
    return params


# Simpler, more reliable version
def count_parameters_simple(network):
    """Simple parameter counter - counts unique parameter arrays."""
    seen = set()
    total = 0

    def count(layer):
        nonlocal total
        for p in layer.params:
            if p.value is not None:
                param_id = id(p.value)
                if param_id not in seen:
                    seen.add(param_id)
                    total += p.value.size

        if hasattr(layer, 'layers'):
            for nested in layer.layers:
                count(nested)

    for layer in network.layers:
        count(layer)

    return total


# ----------------------------------------------------------------------------------
# USAGE EXAMPLES
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

# 1. Quick count
total = count_parameters(network)
print(f"Total parameters: {total:,}")

# 2. Count only trainable
trainable = count_parameters(network, trainable_only=True)
print(f"Trainable parameters: {trainable:,}")

# 3. Detailed summary
print_parameter_summary(network)