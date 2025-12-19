import mlx.core as mx
import numpy as np
from softgrad import Network
from softgrad.layer.shim import MLX
from vgg16_mlx import load_vgg16_weights
from image_utils_mlx import read_image_mlx, write_image_mlx


class DeepDreamNetwork:
    def __init__(self, model_name="vgg16"):
        print(f"Loading {model_name}...")
        if model_name == "vgg16":
            self.mlx_model = load_vgg16_weights()
        else:
            raise ValueError(f"Model {model_name} not supported yet")

    def get_activations(self, img_tensor, layer_names):
        return self.mlx_model(img_tensor, return_layers=layer_names)

    def compute_loss_and_gradients(self, img_tensor, layer_names):
        # Define loss function
        def loss_fn(img):
            activations = self.mlx_model(img, return_layers=layer_names)
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
    dream = DeepDreamNetwork("vgg16")

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
    print(f"✓ Saved to {output_path}")


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
    dream = DeepDreamNetwork("vgg16")

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
    print(f"✓ Saved to {output_path}")


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

    dream = DeepDreamNetwork("vgg16")

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
    print(f"✓ Saved to {output_path}")


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple DeepDream
    deep_dream_simple(
        img_path="in/starry_night.png",
        output_path="out/shimmed/dream_simple.png",
        layer_names=['conv4_3'],
        n_iterations=20,
        learning_rate=0.3
    )

    # Example 2: With jitter
    deep_dream_with_jitter(
        img_path="in/starry_night.png",
        output_path="out/shimmed/dream_jitter.png",
        layer_names=['conv4_3'],
        n_iterations=20,
        learning_rate=0.3,
        jitter=32
    )

    # Example 3: Multi-octave (best quality)
    deep_dream_octaves(
        img_path="in/starry_night.png",
        output_path="out/shimmed/dream_octaves.png",
        layer_names=['conv4_3', 'conv5_2'],  # Multiple layers
        octaves=4,
        octave_scale=1.4,
        n_iterations=10,
        learning_rate=0.1,
        jitter=32,
        target_size=800
    )

    # Example 4: Explore different layers
    for i, layer in enumerate(['conv3_3', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_3']):
        deep_dream_octaves(
            img_path="in/starry_night.png",
            output_path=f"out/shimmed/dream_layer_{layer}.png",
            layer_names=[layer],
            octaves=3,
            n_iterations=10
        )