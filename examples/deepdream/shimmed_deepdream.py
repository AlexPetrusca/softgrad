import mlx.core as mx
import numpy as np
from mlx import nn
from image_utils import read_image_mlx, write_image_mlx
from torchvision import models


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

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

        self.layers = {
            'conv1_1': self.conv1_1, 'conv1_2': self.conv1_2, 'pool1': self.pool1,
            'conv2_1': self.conv2_1, 'conv2_2': self.conv2_2, 'pool2': self.pool2,
            'conv3_1': self.conv3_1, 'conv3_2': self.conv3_2, 'conv3_3': self.conv3_3, 'pool3': self.pool3,
            'conv4_1': self.conv4_1, 'conv4_2': self.conv4_2, 'conv4_3': self.conv4_3, 'pool4': self.pool4,
            'conv5_1': self.conv5_1, 'conv5_2': self.conv5_2, 'conv5_3': self.conv5_3, 'pool5': self.pool5,
        }

        self.load_pretrained_weights()

    def __call__(self, x, return_layers=None):
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

    def load_pretrained_weights(self):
        torch_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        torch_features = torch_model.features

        pytorch_to_mlx_mapping = {
            0: 'conv1_1', 2: 'conv1_2',
            5: 'conv2_1', 7: 'conv2_2',
            10: 'conv3_1', 12: 'conv3_2', 14: 'conv3_3',
            17: 'conv4_1', 19: 'conv4_2', 21: 'conv4_3',
            24: 'conv5_1', 26: 'conv5_2', 28: 'conv5_3',
        }

        for torch_idx, mlx_name in pytorch_to_mlx_mapping.items():
            torch_conv = torch_features[torch_idx]
            mlx_conv = self.layers[mlx_name]

            # convert weights from pytorch to mlx
            weight = torch_conv.weight.detach().numpy()
            weight = np.transpose(weight, (0, 2, 3, 1))  # (out_ch, in_ch, H, W) -> (out_ch, H, W, in_ch)
            mlx_conv.weight = mx.array(weight)
            if torch_conv.bias is not None:
                bias = torch_conv.bias.detach().numpy()
                mlx_conv.bias = mx.array(bias)

    def get_activations(self, img_tensor, layer_names):
        return self(img_tensor, return_layers=layer_names)

    def compute_loss_and_gradients(self, img_tensor, layer_names):
        # Define loss function
        def loss_fn(img):
            activations = self(img, return_layers=layer_names)
            loss = sum(mx.mean(act) for act in activations.values())
            return loss

        # Compute loss and gradients
        loss_and_grad_fn = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad_fn(img_tensor)

        return loss, grads


def deep_dream_simple(
        img_path,
        output_path,
        layer_names=None,
        n_iterations=20,
        learning_rate=0.3,
        target_size=500
):
    if layer_names is None:
        layer_names = ['conv4_3']

    print(f"Starting DeepDream (simple)...")
    print(f"Layers: {layer_names}")
    print(f"Iterations: {n_iterations}")
    print(f"Learning rate: {learning_rate}")

    # Load image
    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    print(f"Image shape: {img_tensor.shape}")

    # Load model
    dream = VGG16()

    # Gradient ascent loop
    for iter in range(n_iterations):
        # Compute loss and gradients
        loss, grads = dream.compute_loss_and_gradients(img_tensor, layer_names)

        # Normalize gradients (smooth them)
        grad_std = mx.std(grads)
        smooth_grads = grads / (grad_std + 1e-8)

        # Gradient ASCENT (maximize activations)
        img_tensor = img_tensor + learning_rate * smooth_grads

        if iter % 10 == 0 or iter == n_iterations - 1:
            print(f"Iteration {iter:3d}, loss: {loss.item():.4f}")

    # Save result
    write_image_mlx(output_path, img_tensor)


def deep_dream_with_jitter(
        img_path,
        output_path,
        layer_names=None,
        n_iterations=20,
        learning_rate=0.3,
        jitter=32,
        target_size=500
):
    if layer_names is None:
        layer_names = ['conv4_3']

    print(f"Starting DeepDream (with jitter)...")
    print(f"Layers: {layer_names}")
    print(f"Jitter: {jitter}px")

    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    dream = VGG16()

    def random_shift(tensor, h_shift, w_shift):
        """Circularly shift the image"""
        # Use positional arguments, not keyword arguments
        tensor = mx.roll(tensor, h_shift, 1)  # Roll along height axis
        tensor = mx.roll(tensor, w_shift, 2)  # Roll along width axis
        return tensor

    for iter in range(n_iterations):
        # Random jitter
        h_shift = np.random.randint(-jitter, jitter + 1)
        w_shift = np.random.randint(-jitter, jitter + 1)

        # Shift image
        img_shifted = random_shift(img_tensor, h_shift, w_shift)

        # Compute gradients
        loss, grads = dream.compute_loss_and_gradients(img_shifted, layer_names)

        # Normalize gradients
        smooth_grads = grads / (mx.std(grads) + 1e-8)

        # Update shifted image
        img_shifted = img_shifted + learning_rate * smooth_grads

        # Shift back
        img_tensor = random_shift(img_shifted, -h_shift, -w_shift)

        if iter % 10 == 0 or iter == n_iterations - 1:
            print(f"Iteration {iter:3d}, loss: {loss.item():.4f}")

    write_image_mlx(output_path, img_tensor)


def deep_dream_octaves(
        img_path,
        output_path,
        layer_names=None,
        octaves=4,
        octave_scale=1.4,
        n_iterations=10,
        learning_rate=0.1,
        jitter=32,
        target_size=800
):
    """
    DeepDream with octave pyramid for multi-scale patterns

    Args:
        octaves: Number of scales
        octave_scale: Scale factor between octaves
    """
    if layer_names is None:
        layer_names = ['conv4_3']

    print(f"Starting DeepDream (octaves)...")
    print(f"Octaves: {octaves}, scale: {octave_scale}")

    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    base_shape = img_tensor.shape[1:3]  # (H, W)

    dream = VGG16()

    def random_shift(tensor, h_shift, w_shift):
        """Circularly shift the image"""
        # Use positional arguments, not keyword arguments
        tensor = mx.roll(tensor, h_shift, 1)  # Roll along height axis
        tensor = mx.roll(tensor, w_shift, 2)  # Roll along width axis
        return tensor

    def resize_image(img, new_shape):
        """Resize using simple interpolation"""
        # For simplicity, use numpy resize
        img_np = np.array(img[0])
        from scipy.ndimage import zoom

        h_scale = new_shape[0] / img_np.shape[0]
        w_scale = new_shape[1] / img_np.shape[1]

        resized = zoom(img_np, (h_scale, w_scale, 1), order=1)
        return mx.array(resized)[None, ...]

    # Process each octave (small to large)
    for octave in range(octaves):
        # Calculate new shape for this octave
        exponent = octave - octaves + 1
        new_h = int(base_shape[0] * (octave_scale ** exponent))
        new_w = int(base_shape[1] * (octave_scale ** exponent))
        new_shape = (new_h, new_w)

        print(f"\nOctave {octave + 1}/{octaves}: {new_shape}")

        # Resize image to this octave
        img_tensor = resize_image(img_tensor, new_shape)

        # DeepDream iterations for this octave
        for iter in range(n_iterations):
            # Random jitter
            h_shift = np.random.randint(-jitter, jitter + 1)
            w_shift = np.random.randint(-jitter, jitter + 1)

            img_shifted = random_shift(img_tensor, h_shift, w_shift)

            # Compute gradients
            loss, grads = dream.compute_loss_and_gradients(img_shifted, layer_names)
            smooth_grads = grads / (mx.std(grads) + 1e-8)

            # Gradient ascent
            img_shifted = img_shifted + learning_rate * smooth_grads

            # Clamp to valid range
            # Approximate bounds (ImageNet normalized)
            img_shifted = mx.clip(img_shifted, -3.0, 3.0)

            # Shift back
            img_tensor = random_shift(img_shifted, -h_shift, -w_shift)

            if iter % 5 == 0:
                print(f"  Iter {iter:2d}, loss: {loss.item():.4f}")

    write_image_mlx(output_path, img_tensor)


# -----------------------------------------------------------------------------
# Generate
# -----------------------------------------------------------------------------
deep_dream_simple(
    img_path="in/starry_night.png",
    output_path="out/shimmed/dream_simple.png",
    layer_names=['conv4_3'],
    n_iterations=20,
    learning_rate=0.3
)

deep_dream_with_jitter(
    img_path="in/starry_night.png",
    output_path="out/shimmed/dream_jitter.png",
    layer_names=['conv4_3'],
    n_iterations=20,
    learning_rate=0.3,
    jitter=32
)

deep_dream_octaves(
    img_path="in/starry_night.png",
    output_path="out/shimmed/dream_octaves.png",
    layer_names=['conv4_3', 'conv5_2'],
    octaves=4,
    octave_scale=1.4,
    n_iterations=10,
    learning_rate=0.1,
    jitter=32,
    target_size=800
)

for i, layer in enumerate(['conv3_3', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_3']):
    deep_dream_octaves(
        img_path="in/starry_night.png",
        output_path=f"out/shimmed/dream_layer_{layer}.png",
        layer_names=[layer],
        octaves=3,
        n_iterations=10
    )