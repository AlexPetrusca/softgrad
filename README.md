# SoftGrad

A lightweight, educational deep learning framework built on [MLX](https://github.com/ml-explore/mlx) for Apple Silicon. SoftGrad provides a clean, intuitive API for building and training neural networks while maintaining full transparency into the forward and backward pass computations.

## Philosophy

SoftGrad is designed to help you **understand** deep learning by implementing it from scratch:

- **Explicit gradients**: See exactly how backpropagation flows through each layer
- **Clean abstractions**: Simple, readable code that mirrors mathematical definitions
- **Native MLX**: Leverages Apple Silicon's Neural Engine for performance
- **Educational focus**: Learn by building real models that actually work

## Features

- **Core Layers**: Linear, Conv2d, MaxPool2d, Embedding, CausalSelfAttention
- **Structural Layers**: Sequential, Parallel, Residual, ProjectionResidual
- **Normalization Layers**: LayerNorm, BatchNorm
- **Activations**: ReLU, LeakyReLU, Softmax, and custom function support
- **Loss Functions**: Cross Entropy, Binary Cross Entropy, MSELoss
- **Optimizers**: SGD with momentum and weight decay
- **Checkpointing**: Save and load model weights
- **MLX Interop**: Use MLX models directly or load PyTorch weights

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexPetrusca/softgrad.git
cd softgrad

# Install dependencies
pip install -r requirements.txt
```

### Hello World: Training a Simple Network

```python
import mlx.core as mx
from softgrad import Network
from softgrad.layer.core import Linear, Activation
from softgrad.function.activation import relu
from softgrad.optim import SGD
from softgrad.function.loss import cross_entropy_loss

# Build network
network = Network(input_shape=784)
network.add_layer(Linear(256))
network.add_layer(Activation(relu))
network.add_layer(Linear(128))
network.add_layer(Activation(relu))
network.add_layer(Linear(10))

# Setup optimizer
optimizer = SGD(eta=0.01, momentum=0.9)
optimizer.bind_loss_fn(cross_entropy_loss)
optimizer.bind_network(network)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in dataloader:
        optimizer.step(x_batch, y_batch)
```

## Examples

Some examples of what Softgrad is capable of.

### 1. Image Classification with CNN

```python
from softgrad import Network
from softgrad.layer.conv import Conv2d, MaxPool2d
from softgrad.layer.core import Linear, Activation
from softgrad.layer.transform import Flatten
from softgrad.function.activation import relu

# Build a simple CNN
network = Network(input_shape=(32, 32, 3))

# Convolutional layers
network.add_layer(Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
network.add_layer(Activation(relu))
network.add_layer(MaxPool2d(kernel_size=2, stride=2))

network.add_layer(Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
network.add_layer(Activation(relu))
network.add_layer(MaxPool2d(kernel_size=2, stride=2))

# Classification head
network.add_layer(Flatten())
network.add_layer(Linear(256))
network.add_layer(Activation(relu))
network.add_layer(Linear(10))
```

### 2. Transformer for Language Modeling (GPT)

```python
from softgrad import Network
from softgrad.function.activation import Relu
from softgrad.function.core import Concatenate, Add
from softgrad.layer.attn import CausalSelfAttention
from softgrad.layer.core import Linear, Activation, Embedding
from softgrad.layer.core import Sequential, Parallel, Residual
from softgrad.layer.norm import LayerNorm
from softgrad.layer.transform.PositionIndices import PositionIndices

class FeedForward(Sequential):
    """Position-wise MLP with expansion and non-linearity"""

    def __init__(self, n_embd):
        super().__init__([
            Linear(4 * n_embd),
            Activation(Relu()),
            Linear(n_embd)
        ])


class MultiHeadAttention(Sequential):
    """Multiple heads of causal self-attention in parallel"""

    def __init__(self, num_heads, head_size, block_size):
        super().__init__([
            Parallel([
                CausalSelfAttention(n_embd, head_size, block_size) # heads
                for _ in range(num_heads)
            ], Concatenate()),
            Linear(n_embd)  # projection
        ])

class TransformerBlock(Sequential):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__([
            Residual(Sequential([
                LayerNorm(),
                MultiHeadAttention(n_head, n_embd // n_head, block_size)
            ])),
            Residual(Sequential([
                LayerNorm(),
                FeedForward(n_embd)
            ]))
        ])

network = Network(input_shape=(block_size,))

# Token and positional embeddings
network.add_layer(Parallel([
    Embedding(vocab_size, n_embd),  
    Sequential([
        PositionIndices(),
        Embedding(block_size, n_embd)
    ])
], Add()))

# Transformer blocks
network.add_layer(Sequential([
    TransformerBlock(n_embd, n_head)
    for _ in range(n_layer)
]))

# LLM head
network.add_layer(LayerNorm())
network.add_layer(Linear(vocab_size))
```

See `examples/transformer/minimal_transformer.py` for a complete GPT-style transformer trained on Shakespeare.

### 3. DeepDream with VGG16

```python
from examples.deepdream import deep_dream_octaves
from load_vgg16 import load_vgg16_pretrained

# Load pretrained VGG16
vgg16 = load_vgg16_pretrained()

# Generate DeepDream
deep_dream_octaves(
    img_path="input.png",
    output_path="output.png",
    layer_names=['conv4_3', 'conv5_2'],
    octaves=4,
    n_iterations=10
)
```

See `examples/deepdream/` for complete DeepDream implementation.

## Architecture

### Forward and Backward Flow

Every layer implements three core methods:

```python
class Layer:
    def _link(self):
        """Initialize parameters based on input shape"""
        
    def _forward(self, x_in: mx.array) -> mx.array:
        """Compute forward pass"""
        
    def _backward(self, dx_out: mx.array) -> mx.array:
        """Compute backward pass (gradient w.r.t. input)"""
```

### Parameter Management

Parameters are stored with explicit gradient tracking:

```python
# Setting parameters
layer.params["W"] = mx.array(weights)
layer.params["b"] = mx.array(bias)

# Accessing gradients (automatic "d" prefix)
weight_grad = layer.params["dW"]
bias_grad = layer.params["db"]
```

### Context Saving

Layers automatically save forward pass values for backward computation:

```python
def _forward(self, x_in: mx.array) -> mx.array:
    x_out = x_in @ self.params["W"] + self.params["b"]
    # Context automatically stores x_in and x_out
    return x_out

def _backward(self, dx_out: mx.array) -> mx.array:
    # Access saved values
    x_in = self.ctx.x_in
    self.params["dW"] += x_in.T @ dx_out
    return dx_out @ self.params["W"].T
```

### Gradient Accumulation

Gradients accumulate across mini-batches:

```python
# In backward pass
self.params["dW"] += gradient  # Accumulate

# After optimizer step
layer.params.zero_grad()  # Reset for next batch
```

## Advanced Features

### Loading PyTorch Weights

```python
from softgrad.util.pytorch_loader import load_pytorch_weights_into_network

# Automatic layer mapping
network.load_from_pytorch(pytorch_model.features)
```

### Using MLX Models Directly (MLX Iterop)

```python
from mlx import nn
from softgrad.layer.shim import MLX

# Wrap any MLX model
mlx_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

network = Network(input_shape=784)
network.add_layer(MLX(mlx_model))
```

## Contributing

Contributions welcome! Areas of interest:

- [ ] Additional optimizers (Adam, AdamW, RMSprop)
- [ ] Data augmentation utilities
- [ ] More layer types (GroupNorm, Dropout, etc.)
- [ ] Visualization tools
- [ ] Performance benchmarks
- [ ] More examples

## Acknowledgments

- Built on [MLX](https://github.com/ml-explore/mlx) by Apple
- DeepDream implementation based on Google's original work
- GPT implementations based on Andrej Karpathy's minGPT and nanoGPT

---

**⭐ If you find this project helpful, please consider starring it!**

---

### Why SoftGrad?

Because understanding comes from building. This framework is intentionally simple, readable, and educational. Every abstraction serves a pedagogical purpose. If you want to truly understand how neural networks work under the hood, build them yourself with SoftGrad.

Happy learning! 🚀
