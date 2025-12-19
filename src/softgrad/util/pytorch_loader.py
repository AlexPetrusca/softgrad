# softgrad/util/pytorch_loader.py

import mlx.core as mx
import numpy as np
import torch
from softgrad.layer.conv import Conv2d
from softgrad.layer.core import Linear


def convert_pytorch_conv2d_weights(weight_torch, bias_torch=None):
    """
    Convert PyTorch Conv2d weights to your framework format

    PyTorch: (out_channels, in_channels, kernel_h, kernel_w)
    Your framework: (kernel_h, kernel_w, in_channels, out_channels)

    Args:
        weight_torch: PyTorch weight tensor
        bias_torch: PyTorch bias tensor (optional)

    Returns:
        weight_mlx, bias_mlx (or None)
    """
    # Convert to numpy
    weight_np = weight_torch.detach().cpu().numpy()

    # Transpose: (out, in, h, w) -> (h, w, in, out)
    # Axes: 0=out, 1=in, 2=h, 3=w -> want: 2=h, 3=w, 1=in, 0=out
    weight_np = np.transpose(weight_np, (2, 3, 1, 0))  # ← FIXED THIS LINE

    # Convert to MLX
    weight_mlx = mx.array(weight_np)

    # Convert bias if present
    bias_mlx = None
    if bias_torch is not None:
        bias_np = bias_torch.detach().cpu().numpy()
        bias_mlx = mx.array(bias_np)

    return weight_mlx, bias_mlx


def convert_pytorch_linear_weights(weight_torch, bias_torch=None):
    """
    Convert PyTorch Linear weights to your framework format

    PyTorch: (out_features, in_features)
    Your framework: (in_features, out_features)

    Args:
        weight_torch: PyTorch weight tensor
        bias_torch: PyTorch bias tensor (optional)

    Returns:
        weight_mlx, bias_mlx (or None)
    """
    # Convert to numpy and transpose
    weight_np = weight_torch.detach().cpu().numpy()
    weight_np = weight_np.T  # (out, in) -> (in, out)
    weight_mlx = mx.array(weight_np)

    # Convert bias
    bias_mlx = None
    if bias_torch is not None:
        bias_np = bias_torch.detach().cpu().numpy()
        bias_mlx = mx.array(bias_np)

    return weight_mlx, bias_mlx


def extract_pytorch_conv_layers(pytorch_model):
    """
    Extract all Conv2d layers from a PyTorch model in order

    Returns:
        List of (name, layer) tuples for Conv2d layers
    """
    conv_layers = []
    for name, module in pytorch_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers


def extract_pytorch_linear_layers(pytorch_model):
    """
    Extract all Linear layers from a PyTorch model in order

    Returns:
        List of (name, layer) tuples for Linear layers
    """
    linear_layers = []
    for name, module in pytorch_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))
    return linear_layers


def load_pytorch_weights_into_network(network, pytorch_model, layer_mapping=None, verbose=True):
    """
    Load PyTorch weights into your network

    Args:
        network: Your Network instance
        pytorch_model: PyTorch model or submodule
        layer_mapping: Optional dict {pytorch_name_or_idx: your_layer_idx}
        verbose: Print progress
    """
    if layer_mapping is None:
        # Auto-mapping: match Conv2d and Linear layers in order
        layer_mapping = _auto_map_layers(network, pytorch_model)

    # Load weights
    loaded_count = 0
    for pt_identifier, your_idx in layer_mapping.items():
        # Get PyTorch layer
        if isinstance(pt_identifier, int):
            # If pytorch_model is Sequential, access by index
            pt_layer = list(pytorch_model.children())[pt_identifier]
        else:
            # Access by name
            pt_layer = dict(pytorch_model.named_modules())[pt_identifier]

        # Get your layer
        your_layer = network.layers[your_idx]

        # Load weights based on layer type
        if isinstance(pt_layer, torch.nn.Conv2d) and isinstance(your_layer, Conv2d):
            weight_mlx, bias_mlx = convert_pytorch_conv2d_weights(
                pt_layer.weight,
                pt_layer.bias
            )
            your_layer.params["W"] = weight_mlx
            if bias_mlx is not None:
                your_layer.params["b"] = bias_mlx

            if verbose:
                print(f"✓ Loaded Conv2d: {pt_identifier} -> layer {your_idx} | W: {weight_mlx.shape}")
            loaded_count += 1

        elif isinstance(pt_layer, torch.nn.Linear) and isinstance(your_layer, Linear):
            weight_mlx, bias_mlx = convert_pytorch_linear_weights(
                pt_layer.weight,
                pt_layer.bias
            )
            your_layer.params["W"] = weight_mlx
            if bias_mlx is not None:
                your_layer.params["b"] = bias_mlx

            if verbose:
                print(f"✓ Loaded Linear: {pt_identifier} -> layer {your_idx} | W: {weight_mlx.shape}")
            loaded_count += 1

        else:
            if verbose:
                print(f"⚠ Skipped {pt_identifier} -> layer {your_idx} (type mismatch)")

    if verbose:
        print(f"\n✓ Loaded {loaded_count} layers successfully")


def _auto_map_layers(network, pytorch_model):
    """
    Automatically map PyTorch layers to your network layers
    Matches Conv2d and Linear layers in order
    """
    # Extract PyTorch Conv2d layers
    pt_conv_layers = extract_pytorch_conv_layers(pytorch_model)

    # Extract your Conv2d layers
    your_conv_indices = [
        i for i, layer in enumerate(network.layers)
        if isinstance(layer, Conv2d)
    ]

    if len(pt_conv_layers) != len(your_conv_indices):
        print(f"Warning: PyTorch has {len(pt_conv_layers)} Conv2d layers, "
              f"your network has {len(your_conv_indices)} Conv2d layers")

    # Create mapping
    mapping = {}
    for (pt_name, pt_layer), your_idx in zip(pt_conv_layers, your_conv_indices):
        mapping[pt_name] = your_idx

    # Also handle Linear layers if present
    pt_linear_layers = extract_pytorch_linear_layers(pytorch_model)
    your_linear_indices = [
        i for i, layer in enumerate(network.layers)
        if isinstance(layer, Linear)
    ]

    for (pt_name, pt_layer), your_idx in zip(pt_linear_layers, your_linear_indices):
        mapping[pt_name] = your_idx

    return mapping