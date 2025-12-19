# deepdream_native.py - Complete implementation using your framework

import mlx.core as mx
import numpy as np
from load_vgg16 import load_vgg16_pretrained
from image_utils_mlx import read_image_mlx, write_image_mlx


# ============================================================================
# HELPER: Extract layer activations
# ============================================================================

def get_layer_name_to_index():
    """Map layer names to indices - AFTER ReLU activation"""
    return {
        # Extract AFTER ReLU, not at Conv2d
        'conv1_1': 1,  # After first ReLU (was 0)
        'conv1_2': 3,  # After ReLU (was 2)
        'conv2_1': 6,  # After ReLU (was 5)
        'conv2_2': 8,  # After ReLU (was 7)
        'conv3_1': 11, # After ReLU (was 10)
        'conv3_2': 13, # After ReLU (was 12)
        'conv3_3': 15, # After ReLU (was 14)
        'conv4_1': 18, # After ReLU (was 17)
        'conv4_2': 20, # After ReLU (was 19)
        'conv4_3': 22, # After ReLU (was 21) ← KEY FIX
        'conv5_1': 25, # After ReLU (was 24)
        'conv5_2': 27, # After ReLU (was 26)
        'conv5_3': 29, # After ReLU (was 28)
    }


def get_activations(network, img_tensor, layer_names):
    """
    Extract activations from specific layers by running forward pass

    Args:
        network: Your VGG16 Network
        img_tensor: Input image (B, H, W, C)
        layer_names: List of layer names like ['conv4_3', 'conv5_2']

    Returns:
        dict of {layer_name: activation}, final_output
    """
    name_to_idx = get_layer_name_to_index()
    target_indices = {name_to_idx[name]: name for name in layer_names}

    activations = {}
    x = img_tensor

    # Run forward pass and capture target activations
    for i, layer in enumerate(network.layers):
        x = layer.forward(x, save_ctx=True)  # Save context for backward

        if i in target_indices:
            activations[target_indices[i]] = x

    return activations, x


# ============================================================================
# APPROACH 1: Using MLX autodiff with framework forward pass
# ============================================================================

def compute_loss_and_gradients_hybrid(network, img_tensor, layer_names):
    """
    Compute loss and gradients using MLX autodiff over your framework
    This is a hybrid approach: your framework for forward, MLX for gradients

    Args:
        network: Your VGG16 Network
        img_tensor: Input image
        layer_names: Layers to maximize

    Returns:
        loss, gradients w.r.t. input
    """

    def loss_fn(img):
        # Forward through your framework
        activations, _ = get_activations(network, img, layer_names)

        # Loss = sum of mean activations (we want to maximize this)
        loss = sum(mx.mean(act) for act in activations.values())
        return loss

    # Use MLX's automatic differentiation
    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(img_tensor)

    return loss, grads


# ============================================================================
# APPROACH 2: Manual backward pass through your framework
# ============================================================================

def compute_loss_and_gradients_manual(network, img_tensor, layer_names):
    """
    Compute loss and gradients using your framework's backward pass
    This is fully native to your framework

    Args:
        network: Your VGG16 Network
        img_tensor: Input image
        layer_names: Layers to maximize

    Returns:
        loss, gradients w.r.t. input
    """
    name_to_idx = get_layer_name_to_index()
    target_indices = set(name_to_idx[name] for name in layer_names)

    # Forward pass (already done in get_activations)
    activations, final_output = get_activations(network, img_tensor, layer_names)

    # Compute loss
    loss = sum(mx.mean(act) for act in activations.values())

    # Start backward pass from the end
    # Initial gradient: we want to maximize the loss, so gradient = 1
    dx = mx.ones_like(final_output) / final_output.size

    # Backward through layers (in reverse)
    for i in reversed(range(len(network.layers))):
        layer = network.layers[i]

        # If this is a target layer, inject gradient signal
        if i in target_indices:
            # Add gradient to maximize this layer's mean activation
            # Gradient of mean(x) w.r.t. x is 1/size
            layer_act = layer.ctx.x_out
            gradient_contribution = mx.ones_like(layer_act) / layer_act.size
            dx = dx + gradient_contribution

        # Backprop through this layer
        dx = layer.backward(dx, save_ctx=False)

    return loss, dx


# ============================================================================
# DEEPDREAM IMPLEMENTATIONS
# ============================================================================

def deep_dream_simple(
        img_path,
        output_path,
        layer_names=None,
        n_iterations=20,
        learning_rate=0.3,
        target_size=500,
        use_manual_backward=False
):
    """
    Simple DeepDream using your native framework

    Args:
        img_path: Input image path
        output_path: Output image path
        layer_names: Layers to maximize (e.g., ['conv4_3'])
        n_iterations: Number of gradient ascent steps
        learning_rate: Step size
        target_size: Resize width
        use_manual_backward: Use manual backward (True) or hybrid MLX autodiff (False)
    """
    if layer_names is None:
        layer_names = ['conv4_3']

    print(f"Starting DeepDream (native framework)...")
    print(f"Layers: {layer_names}")
    print(f"Method: {'Manual backward' if use_manual_backward else 'Hybrid MLX autodiff'}")

    # Load image
    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    print(f"Image shape: {img_tensor.shape}")

    # Load VGG16
    print("Loading VGG16...")
    network = load_vgg16_pretrained()

    # Choose gradient computation method
    compute_grads = (compute_loss_and_gradients_manual if use_manual_backward
                     else compute_loss_and_gradients_hybrid)

    # Gradient ascent loop
    for iter in range(n_iterations):
        # Compute loss and gradients
        loss, grads = compute_grads(network, img_tensor, layer_names)

        # Normalize gradients
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
    """
    DeepDream with random jitter for more diverse results
    """
    if layer_names is None:
        layer_names = ['conv4_3']

    print(f"Starting DeepDream (with jitter)...")
    print(f"Layers: {layer_names}")
    print(f"Jitter: {jitter}px")

    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    network = load_vgg16_pretrained()

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

        # Compute gradients (using hybrid approach)
        loss, grads = compute_loss_and_gradients_hybrid(network, img_shifted, layer_names)

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
    """
    if layer_names is None:
        layer_names = ['conv4_3']

    print(f"Starting DeepDream (octaves)...")
    print(f"Octaves: {octaves}, scale: {octave_scale}")

    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    base_shape = img_tensor.shape[1:3]  # (H, W)

    network = load_vgg16_pretrained()

    def random_shift(tensor, h_shift, w_shift):
        """Circularly shift the image"""
        # Use positional arguments, not keyword arguments
        tensor = mx.roll(tensor, h_shift, 1)  # Roll along height axis
        tensor = mx.roll(tensor, w_shift, 2)  # Roll along width axis
        return tensor

    def resize_image(img, new_shape):
        """Resize using scipy"""
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
            loss, grads = compute_loss_and_gradients_hybrid(network, img_shifted, layer_names)
            smooth_grads = grads / (mx.std(grads) + 1e-8)

            # Gradient ascent
            img_shifted = img_shifted + learning_rate * smooth_grads

            # Clamp to valid range (approximate ImageNet normalized bounds)
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
    # Example 1: Simple DeepDream with hybrid approach
    deep_dream_simple(
        img_path="in/starry_night.png",
        output_path="out/native/dream_simple.png",
        layer_names=['conv4_3'],
        n_iterations=20,
        learning_rate=0.3,
        use_manual_backward=False  # Use hybrid MLX autodiff
    )

    # Example 2: With jitter
    deep_dream_with_jitter(
        img_path="in/starry_night.png",
        output_path="out/native/dream_jitter.png",
        layer_names=['conv4_3'],
        n_iterations=20,
        learning_rate=0.3,
        jitter=32
    )

    # Example 3: Multi-octave (best quality)
    deep_dream_octaves(
        img_path="in/starry_night.png",
        output_path="out/native/dream_octaves.png",
        layer_names=['conv4_3', 'conv5_2'],
        octaves=4,
        octave_scale=1.4,
        n_iterations=10,
        learning_rate=0.1,
        jitter=32,
        target_size=800
    )

    # Example 4: Explore different layers
    for layer in ['conv3_3', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_3']:
        deep_dream_octaves(
            img_path="in/starry_night.png",
            output_path=f"out/native/dream_{layer}.png",
            layer_names=[layer],
            octaves=3,
            n_iterations=10
        )

