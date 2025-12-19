# vgg16_native.py
import mlx.core as mx
import numpy as np
import torch
from torchvision import models

from softgrad import Network
from softgrad.layer.conv import Conv2d, MaxPool2d
from softgrad.layer.core import Activation
from softgrad.layer.core import Sequential
from softgrad.function.activation import relu


def build_vgg16():
    """Build VGG16 architecture using native framework layers"""

    network = Network(input_shape=(224, 224, 3))

    # Block 1
    network.add_layer(Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(MaxPool2d(kernel_size=2, stride=2))

    # Block 2
    network.add_layer(Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(MaxPool2d(kernel_size=2, stride=2))

    # Block 3
    network.add_layer(Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(MaxPool2d(kernel_size=2, stride=2))

    # Block 4
    network.add_layer(Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(MaxPool2d(kernel_size=2, stride=2))

    # Block 5
    network.add_layer(Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1))
    network.add_layer(Activation(relu))
    network.add_layer(MaxPool2d(kernel_size=2, stride=2))

    return network


def load_vgg16_weights_native(network):
    """
    Load pretrained VGG16 weights from PyTorch into your framework

    Args:
        network: Your Network instance with VGG16 architecture
    """
    print("Loading PyTorch VGG16 weights...")
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_features = torch_model.features

    print("Converting weights to your framework...")

    # Map PyTorch feature indices to your network layer indices
    # PyTorch VGG16.features structure:
    # 0: Conv2d, 1: ReLU, 2: Conv2d, 3: ReLU, 4: MaxPool2d, ...

    # Your network structure:
    # 0: Conv2d, 1: Activation(relu), 2: Conv2d, 3: Activation(relu), 4: MaxPool2d, ...

    # Extract only Conv2d layers from both
    pytorch_conv_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    your_conv_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]  # Same pattern!

    for pt_idx, your_idx in zip(pytorch_conv_indices, your_conv_indices):
        print(f"Loading layer {your_idx}...")

        # Get PyTorch layer
        torch_conv = torch_features[pt_idx]

        # Get your framework layer
        your_conv = network.layers[your_idx]

        # Convert weight format
        # PyTorch Conv2d: (out_channels, in_channels, kernel_h, kernel_w)
        # Check your Conv2d implementation to see what format it expects

        weight_torch = torch_conv.weight.detach().numpy()
        bias_torch = torch_conv.bias.detach().numpy() if torch_conv.bias is not None else None

        # You need to check your Conv2d implementation!
        # If it uses MLX format: (out_channels, kernel_h, kernel_w, in_channels)
        # Then transpose: (out, in, h, w) -> (out, h, w, in)
        weight = np.transpose(weight_torch, (0, 2, 3, 1))

        # Set parameters in your framework
        your_conv.params["W"] = mx.array(weight)
        if bias_torch is not None:
            your_conv.params["b"] = mx.array(bias_torch)

    print("✓ Weights loaded successfully!")
    return network


def get_layer_name_mapping():
    """
    Create a mapping of layer names for activation extraction
    Returns dict: {name: layer_index}
    """
    mapping = {
        'conv1_1': 0, 'conv1_2': 2,
        'conv2_1': 5, 'conv2_2': 7,
        'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14,
        'conv4_1': 17, 'conv4_2': 19, 'conv4_3': 21,
        'conv5_1': 24, 'conv5_2': 26, 'conv5_3': 28,
    }
    return mapping


def get_activations(network, img_tensor, layer_names):
    """
    Extract activations from specific layers

    Args:
        network: Your VGG16 Network
        img_tensor: Input image (B, H, W, C)
        layer_names: List like ['conv4_3', 'conv5_2']

    Returns:
        dict of {layer_name: activation}
    """
    name_to_idx = get_layer_name_mapping()
    target_indices = {name_to_idx[name]: name for name in layer_names}

    activations = {}
    x = img_tensor

    for i, layer in enumerate(network.layers):
        x = layer.forward(x, save_ctx=False)

        # Save activation if this is a target layer
        if i in target_indices:
            activations[target_indices[i]] = x

    return activations


# ============================================================================
# TEST LOADING
# ============================================================================

if __name__ == "__main__":
    print("Building VGG16 in your framework...")
    network = build_vgg16()

    print(f"\nNetwork has {len(network.layers)} layers")
    print(f"Input shape: {network.input_shape}")
    print(f"Output shape: {network.output_shape}")

    print("\nLoading pretrained weights...")
    network = load_vgg16_weights_native(network)

    print("\nTesting forward pass...")
    test_img = mx.random.normal((1, 224, 224, 3))
    output = network.forward(test_img, save_ctx=False)
    print(f"Output shape: {output.shape}")

    print("\nTesting activation extraction...")
    activations = get_activations(network, test_img, ['conv4_3', 'conv5_2'])
    for name, act in activations.items():
        print(f"{name}: {act.shape}")

    print("\n✓ VGG16 loaded successfully in your framework!")