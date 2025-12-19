import mlx.core as mx
import numpy as np
import torch
from softgrad.layer.conv import Conv2d
from softgrad.layer.core import Linear


def convert_pytorch_conv2d_weights(weight_torch, bias_torch=None):
    weight_np = weight_torch.detach().cpu().numpy()
    weight_np = np.transpose(weight_np, (2, 3, 1, 0))  # (out, in, h, w) -> (h, w, in, out)
    weight_mlx = mx.array(weight_np)
    bias_mlx = None
    if bias_torch is not None:
        bias_np = bias_torch.detach().cpu().numpy()
        bias_mlx = mx.array(bias_np)
    return weight_mlx, bias_mlx


def convert_pytorch_linear_weights(weight_torch, bias_torch=None):
    weight_np = weight_torch.detach().cpu().numpy()
    weight_np = weight_np.T  # (out, in) -> (in, out)
    weight_mlx = mx.array(weight_np)
    bias_mlx = None
    if bias_torch is not None:
        bias_np = bias_torch.detach().cpu().numpy()
        bias_mlx = mx.array(bias_np)
    return weight_mlx, bias_mlx


def extract_pytorch_conv_layers(pytorch_model):
    conv_layers = []
    for name, module in pytorch_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers


def extract_pytorch_linear_layers(pytorch_model):
    linear_layers = []
    for name, module in pytorch_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))
    return linear_layers


def load_pytorch_weights_into_network(network, pytorch_model, layer_mapping=None):
    if layer_mapping is None:
        layer_mapping = _auto_map_layers(network, pytorch_model) # auto-map

    loaded_count = 0
    for pytorch_id, id in layer_mapping.items():
        # Get PyTorch layer
        if isinstance(pytorch_id, int):
            pytorch_layer = list(pytorch_model.children())[pytorch_id]
        else:
            pytorch_layer = dict(pytorch_model.named_modules())[pytorch_id]

        # Get SoftGrad layer
        layer = network.layers[id]

        # Load weights
        if isinstance(pytorch_layer, torch.nn.Conv2d) and isinstance(layer, Conv2d):
            weight_mlx, bias_mlx = convert_pytorch_conv2d_weights(
                pytorch_layer.weight,
                pytorch_layer.bias
            )
            layer.params["W"] = weight_mlx
            if bias_mlx is not None:
                layer.params["b"] = bias_mlx

            loaded_count += 1
        elif isinstance(pytorch_layer, torch.nn.Linear) and isinstance(layer, Linear):
            weight_mlx, bias_mlx = convert_pytorch_linear_weights(
                pytorch_layer.weight,
                pytorch_layer.bias
            )
            layer.params["W"] = weight_mlx
            if bias_mlx is not None:
                layer.params["b"] = bias_mlx

            loaded_count += 1


def _auto_map_layers(network, pytorch_network):
    pytorch_convs = extract_pytorch_conv_layers(pytorch_network)
    convs = [
        i for i, layer in enumerate(network.layers)
        if isinstance(layer, Conv2d)
    ]

    if len(pytorch_convs) != len(convs):
        raise Exception("Number of conv layers does not match number of conv layers")

    # Create mapping
    mapping = {}
    for (pytorch_name, pytorch_layer), idx in zip(pytorch_convs, convs):
        mapping[pytorch_name] = idx

    pytorch_linears = extract_pytorch_linear_layers(pytorch_network)
    linear_indices = [
        i for i, layer in enumerate(network.layers)
        if isinstance(layer, Linear)
    ]

    for (pytorch_name, pytorch_layer), idx in zip(pytorch_linears, linear_indices):
        mapping[pytorch_name] = idx

    return mapping