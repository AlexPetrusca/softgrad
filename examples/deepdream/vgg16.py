import mlx.core as mx
import numpy as np
from torchvision import models

from softgrad import Network
from softgrad.layer.conv import Conv2d, MaxPool2d
from softgrad.layer.core import Activation
from softgrad.function.activation import relu


def build_vgg16():
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
    torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    torch_features = torch_model.features

    for idx in get_layer_name_mapping().values():
        torch_conv = torch_features[idx]
        your_conv = network.layers[idx]

        weight_torch = torch_conv.weight.detach().numpy()
        bias_torch = torch_conv.bias.detach().numpy() if torch_conv.bias is not None else None

        # (out, in, h, w) -> (out, h, w, in)
        weight = np.transpose(weight_torch, (0, 2, 3, 1))

        your_conv.params["W"] = mx.array(weight)
        if bias_torch is not None:
            your_conv.params["b"] = mx.array(bias_torch)

    return network


def get_layer_name_mapping():
    mapping = {
        'conv1_1': 0,
        'conv1_2': 2,
        'conv2_1': 5,
        'conv2_2': 7,
        'conv3_1': 10,
        'conv3_2': 12,
        'conv3_3': 14,
        'conv4_1': 17,
        'conv4_2': 19,
        'conv4_3': 21,
        'conv5_1': 24,
        'conv5_2': 26,
        'conv5_3': 28,
    }
    return mapping


def get_activations(network, img, layer_names):
    name_to_idx = get_layer_name_mapping()
    target_indices = {name_to_idx[name]: name for name in layer_names}

    activations = {}
    x = img

    for i, layer in enumerate(network.layers):
        x = layer.forward(x, save_ctx=False)

        # save activation if this is a target
        if i in target_indices:
            activations[target_indices[i]] = x

    return activations

if __name__ == "__main__":
    network = build_vgg16()

    print(f"Layer count: {len(network.layers)}")
    print(f"Input shape: {network.input_shape}")
    print(f"Output shape: {network.output_shape}")
    print()

    network = load_vgg16_weights_native(network)

    # test network
    test_img = mx.random.normal((1, 224, 224, 3))
    output = network.forward(test_img, save_ctx=False)
    print(test_img.shape, "-->", output.shape)