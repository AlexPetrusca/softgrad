import mlx.core as mx
import numpy as np
from load_vgg16 import load_vgg16_pretrained
from image_utils import read_image_mlx, write_image_mlx


def get_activations(network, img_tensor, layer_names):
    name_to_idx = {
        'conv1_1': 1, 'conv1_2': 3,
        'conv2_1': 6, 'conv2_2': 8,
        'conv3_1': 11, 'conv3_2': 13, 'conv3_3': 15,
        'conv4_1': 18, 'conv4_2': 20, 'conv4_3': 22,
        'conv5_1': 25, 'conv5_2': 27, 'conv5_3': 29,
    }
    target_indices = {name_to_idx[name]: name for name in layer_names}

    activations = {}
    x = img_tensor

    for i, layer in enumerate(network.layers):
        x = layer.forward(x, save_ctx=True)  # save context for backward
        if i in target_indices:
            activations[target_indices[i]] = x

    return activations, x


def compute_loss_and_gradients(network, img_tensor, layer_names):
    def loss_fn(img):
        activations, _ = get_activations(network, img, layer_names)
        loss = sum(mx.mean(act) for act in activations.values())
        return loss

    loss_and_grad_fn = mx.value_and_grad(loss_fn)
    loss, grads = loss_and_grad_fn(img_tensor)
    return loss, grads


# -----------------------------------------------------------------------------
# DeepDream Implementations
# -----------------------------------------------------------------------------

def deep_dream_simple(img_path, output_path, layer_names=None, n_iterations=20, learning_rate=0.3, target_size=500):
    if layer_names is None:
        layer_names = ['conv4_3']

    print()
    print("------------------------------------")
    print(f"DeepDream simple")
    print(layer_names)
    print("------------------------------------")

    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    network = load_vgg16_pretrained()

    for iter in range(n_iterations):
        # compute & normalize gradients
        loss, grads = compute_loss_and_gradients(network, img_tensor, layer_names)
        grad_std = mx.std(grads)
        smooth_grads = grads / (grad_std + 1e-8)

        # update image
        img_tensor = img_tensor + learning_rate * smooth_grads

        if iter % 10 == 0 or iter == n_iterations - 1:
            print(f"Iteration {iter:3d}, loss: {loss.item():.4f}")

    write_image_mlx(output_path, img_tensor)


def deep_dream_with_jitter(img_path, output_path, layer_names=None, n_iterations=20, learning_rate=0.3, jitter=32, target_size=500):
    if layer_names is None:
        layer_names = ['conv4_3']

    print()
    print("------------------------------------")
    print(f"DeepDream with jitter...")
    print(layer_names)
    print("------------------------------------")

    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    network = load_vgg16_pretrained()

    def random_shift(tensor, h_shift, w_shift):
        tensor = mx.roll(tensor, h_shift, 1)
        tensor = mx.roll(tensor, w_shift, 2)
        return tensor

    for iter in range(n_iterations):
        h_shift = np.random.randint(-jitter, jitter + 1)
        w_shift = np.random.randint(-jitter, jitter + 1)

        # shift image
        img_shifted = random_shift(img_tensor, h_shift, w_shift)

        # compute + normalize gradients
        loss, grads = compute_loss_and_gradients(network, img_shifted, layer_names)
        smooth_grads = grads / (mx.std(grads) + 1e-8)

        # Update image
        img_shifted = img_shifted + learning_rate * smooth_grads

        # Shift image back
        img_tensor = random_shift(img_shifted, -h_shift, -w_shift)

        if iter % 10 == 0 or iter == n_iterations - 1:
            print(f"Iteration {iter:3d}, loss: {loss.item():.4f}")

    write_image_mlx(output_path, img_tensor)


def deep_dream_octaves(img_path, output_path, layer_names=None, octaves=4, octave_scale=1.4, n_iterations=10,
                       learning_rate=0.1, jitter=32, target_size=800):
    if layer_names is None:
        layer_names = ['conv4_3']

    print()
    print("------------------------------------")
    print(f"DeepDream with jitter + octaves")
    print(layer_names)
    print("------------------------------------")

    img_tensor = read_image_mlx(img_path, target_shape=target_size)
    base_shape = img_tensor.shape[1:3]  # (H, W)

    network = load_vgg16_pretrained()

    def random_shift(tensor, h_shift, w_shift):
        tensor = mx.roll(tensor, h_shift, 1)
        tensor = mx.roll(tensor, w_shift, 2)
        return tensor

    def resize_image(img, new_shape):
        img_np = np.array(img[0])
        from scipy.ndimage import zoom

        h_scale = new_shape[0] / img_np.shape[0]
        w_scale = new_shape[1] / img_np.shape[1]

        resized = zoom(img_np, (h_scale, w_scale, 1), order=1)
        return mx.array(resized)[None, ...]

    for octave in range(octaves):
        exponent = octave - octaves + 1
        new_h = int(base_shape[0] * (octave_scale ** exponent))
        new_w = int(base_shape[1] * (octave_scale ** exponent))
        new_shape = (new_h, new_w)

        print(f"Octave {octave + 1}/{octaves}: {new_shape}")

        img_tensor = resize_image(img_tensor, new_shape)

        for iter in range(n_iterations):
            h_shift = np.random.randint(-jitter, jitter + 1)
            w_shift = np.random.randint(-jitter, jitter + 1)

            # shift image
            img_shifted = random_shift(img_tensor, h_shift, w_shift)

            # compute & normalize gradients
            loss, grads = compute_loss_and_gradients(network, img_shifted, layer_names)
            smooth_grads = grads / (mx.std(grads) + 1e-8)

            # update image
            img_shifted = img_shifted + learning_rate * smooth_grads
            img_shifted = mx.clip(img_shifted, -3.0, 3.0)  # clamp

            # shift image back
            img_tensor = random_shift(img_shifted, -h_shift, -w_shift)

            if iter % 5 == 0:
                print(f"  Iter {iter:2d}, loss: {loss.item():.4f}")

        print()

    write_image_mlx(output_path, img_tensor)


# -----------------------------------------------------------------------------
# Generate
# -----------------------------------------------------------------------------
deep_dream_simple(
    img_path="in/starry_night.png",
    output_path="out/native/dream_simple.png",
    layer_names=['conv4_3'],
    n_iterations=20,
    learning_rate=0.3
)

deep_dream_with_jitter(
    img_path="in/starry_night.png",
    output_path="out/native/dream_jitter.png",
    layer_names=['conv4_3'],
    n_iterations=20,
    learning_rate=0.3,
    jitter=32
)

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

for layer in ['conv3_3', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_3']:
    deep_dream_octaves(
        img_path="in/starry_night.png",
        output_path=f"out/native/dream_{layer}.png",
        layer_names=[layer],
        octaves=3,
        n_iterations=10
    )

