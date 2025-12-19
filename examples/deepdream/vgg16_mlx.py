# vgg16_mlx.py
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from torchvision import models


class VGG16Features(nn.Module):
    """VGG16 feature extractor in MLX"""

    def __init__(self):
        super().__init__()

        # VGG16 architecture
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Store layers for easy access by name
        self.layers = {
            'conv1_1': self.conv1_1, 'conv1_2': self.conv1_2, 'pool1': self.pool1,
            'conv2_1': self.conv2_1, 'conv2_2': self.conv2_2, 'pool2': self.pool2,
            'conv3_1': self.conv3_1, 'conv3_2': self.conv3_2, 'conv3_3': self.conv3_3, 'pool3': self.pool3,
            'conv4_1': self.conv4_1, 'conv4_2': self.conv4_2, 'conv4_3': self.conv4_3, 'pool4': self.pool4,
            'conv5_1': self.conv5_1, 'conv5_2': self.conv5_2, 'conv5_3': self.conv5_3, 'pool5': self.pool5,
        }

    def __call__(self, x, return_layers=None):
        """
        Forward pass with optional intermediate layer returns

        Args:
            x: Input image (B, H, W, C) in MLX format
            return_layers: List of layer names to return activations from

        Returns:
            If return_layers is None: final output
            Otherwise: dict of {layer_name: activation}
        """
        activations = {}

        # Block 1
        x = nn.relu(self.conv1_1(x))
        if return_layers and 'conv1_1' in return_layers:
            activations['conv1_1'] = x

        x = nn.relu(self.conv1_2(x))
        if return_layers and 'conv1_2' in return_layers:
            activations['conv1_2'] = x

        x = self.pool1(x)
        if return_layers and 'pool1' in return_layers:
            activations['pool1'] = x

        # Block 2
        x = nn.relu(self.conv2_1(x))
        if return_layers and 'conv2_1' in return_layers:
            activations['conv2_1'] = x

        x = nn.relu(self.conv2_2(x))
        if return_layers and 'conv2_2' in return_layers:
            activations['conv2_2'] = x

        x = self.pool2(x)
        if return_layers and 'pool2' in return_layers:
            activations['pool2'] = x

        # Block 3
        x = nn.relu(self.conv3_1(x))
        if return_layers and 'conv3_1' in return_layers:
            activations['conv3_1'] = x

        x = nn.relu(self.conv3_2(x))
        if return_layers and 'conv3_2' in return_layers:
            activations['conv3_2'] = x

        x = nn.relu(self.conv3_3(x))
        if return_layers and 'conv3_3' in return_layers:
            activations['conv3_3'] = x

        x = self.pool3(x)
        if return_layers and 'pool3' in return_layers:
            activations['pool3'] = x

        # Block 4
        x = nn.relu(self.conv4_1(x))
        if return_layers and 'conv4_1' in return_layers:
            activations['conv4_1'] = x

        x = nn.relu(self.conv4_2(x))
        if return_layers and 'conv4_2' in return_layers:
            activations['conv4_2'] = x

        x = nn.relu(self.conv4_3(x))
        if return_layers and 'conv4_3' in return_layers:
            activations['conv4_3'] = x

        x = self.pool4(x)
        if return_layers and 'pool4' in return_layers:
            activations['pool4'] = x

        # Block 5
        x = nn.relu(self.conv5_1(x))
        if return_layers and 'conv5_1' in return_layers:
            activations['conv5_1'] = x

        x = nn.relu(self.conv5_2(x))
        if return_layers and 'conv5_2' in return_layers:
            activations['conv5_2'] = x

        x = nn.relu(self.conv5_3(x))
        if return_layers and 'conv5_3' in return_layers:
            activations['conv5_3'] = x

        x = self.pool5(x)
        if return_layers and 'pool5' in return_layers:
            activations['pool5'] = x

        if return_layers:
            return activations
        return x


def load_vgg16_weights():
    """Load pretrained VGG16 weights from PyTorch and convert to MLX"""
    print("Loading PyTorch VGG16 weights...")
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_features = torch_model.features

    print("Creating MLX model...")
    mlx_model = VGG16Features()

    print("Converting weights to MLX...")
    # Map PyTorch layers to MLX layers
    # PyTorch VGG16 features has indices like: 0, 2, 5, 7, etc. (conv layers)
    pytorch_to_mlx_mapping = {
        0: 'conv1_1', 2: 'conv1_2',
        5: 'conv2_1', 7: 'conv2_2',
        10: 'conv3_1', 12: 'conv3_2', 14: 'conv3_3',
        17: 'conv4_1', 19: 'conv4_2', 21: 'conv4_3',
        24: 'conv5_1', 26: 'conv5_2', 28: 'conv5_3',
    }

    for torch_idx, mlx_name in pytorch_to_mlx_mapping.items():
        torch_conv = torch_features[torch_idx]
        mlx_conv = mlx_model.layers[mlx_name]

        # Convert weights: PyTorch is (out_ch, in_ch, H, W), MLX is (out_ch, H, W, in_ch)
        weight = torch_conv.weight.detach().numpy()
        weight = np.transpose(weight, (0, 2, 3, 1))  # (out, H, W, in)
        mlx_conv.weight = mx.array(weight)

        # Convert bias
        if torch_conv.bias is not None:
            bias = torch_conv.bias.detach().numpy()
            mlx_conv.bias = mx.array(bias)

    print("✓ VGG16 weights loaded successfully!")
    return mlx_model


# Test loading
if __name__ == "__main__":
    model = load_vgg16_weights()

    # Test forward pass
    test_img = mx.random.normal((1, 224, 224, 3))  # MLX format: (B, H, W, C)
    output = model(test_img)
    print(f"Output shape: {output.shape}")

    # Test with intermediate layers
    activations = model(test_img, return_layers=['conv4_3', 'conv5_3'])
    for name, act in activations.items():
        print(f"{name}: {act.shape}")