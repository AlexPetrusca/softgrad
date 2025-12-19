import os
import cv2 as cv
import torch
from torchvision import models
import mlx.core as mx
import numpy as np

from softgrad import Network
from softgrad.layer.shim import MLX
from vgg16_mlx import load_vgg16_weights
from image_utils_mlx import read_image_mlx, write_image_mlx
from softgrad import Network
from softgrad.layer.conv import Conv2d, MaxPool2d
from softgrad.layer.core import Activation
from softgrad.function.activation import relu


IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])


def read_image_mlx(img_path, target_shape=None):
    """Read image and convert to MLX tensor"""
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')

    img = cv.imread(img_path)[:, :, ::-1]  # BGR to RGB

    # Resize if needed
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0

    # Normalize with ImageNet stats
    img = (img - IMAGENET_MEAN.tolist()) / IMAGENET_STD.tolist()

    # Convert to MLX: (H, W, C) -> add batch dim -> (1, H, W, C)
    img_tensor = mx.array(img)[None, ...]

    return img_tensor


def write_image_mlx(img_path, img_tensor):
    """Write MLX tensor to image file"""
    # Remove batch dimension and convert to numpy
    img = np.array(img_tensor[0])  # (H, W, C)

    # Denormalize
    mean = IMAGENET_MEAN.tolist()
    std = IMAGENET_STD.tolist()
    img = (img * std) + mean

    # Clip and convert to uint8
    img = (np.clip(img, 0., 1.) * 255).astype(np.uint8)

    # RGB to BGR for OpenCV
    cv.imwrite(img_path, img[:, :, ::-1])



def build_vgg16_network():
    """Build VGG16 architecture in your framework"""
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


def load_vgg16_pretrained():
    """
    Build VGG16 and load pretrained ImageNet weights from PyTorch

    Returns:
        network: Your Network with loaded weights
    """
    print("Building VGG16 architecture...")
    network = build_vgg16_network()

    print("\nLoading pretrained weights from PyTorch...")
    pytorch_vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Load weights (auto-mapping will match Conv2d layers in order)
    network.load_from_pytorch(pytorch_vgg16.features, verbose=True)

    # Freeze all parameters (we're not training)
    network.freeze()

    print("\n✓ VGG16 loaded successfully!")
    return network


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Load VGG16
    vgg16 = load_vgg16_pretrained()

    # Test forward pass
    import mlx.core as mx

    test_img = mx.random.normal((1, 224, 224, 3))
    output = vgg16.forward(test_img, save_ctx=False)
    print(f"\nTest output shape: {output.shape}")

