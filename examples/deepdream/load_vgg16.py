import mlx.core as mx
from torchvision import models
from softgrad import Network
from softgrad.layer.conv import Conv2d, MaxPool2d
from softgrad.layer.core import Activation
from softgrad.function.activation import relu


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
    network = build_vgg16_network()

    pytorch_vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    network.load_from_pytorch(pytorch_vgg16.features)
    network.freeze()

    return network


if __name__ == "__main__":
    vgg16 = load_vgg16_pretrained()

    test_img = mx.random.normal((1, 224, 224, 3))
    output = vgg16.forward(test_img, save_ctx=False)
    print(f"Output shape: {output.shape}")

