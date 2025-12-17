# train the model

from datetime import datetime

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
from PIL import Image
from mlx import nn

from softgrad import Network
from softgrad.layer.conv import MaxPool2d, Conv2d
from softgrad.layer.core.ProjectionResidual import ProjectionResidual
from softgrad.layer.norm import BatchNorm, LayerNorm
from softgrad.layer.shim import MLX
from softgrad.layer.transform import Flatten
from softgrad.optim import SGD
from softgrad.function.activation import softmax, Relu
from softgrad.function.loss import CrossEntropyLoss, cross_entropy_loss
from softgrad.layer.core import Linear, Activation, Residual, Sequential

from util.dataset import get_mnist, get_fashion_mnist, get_cifar10


# Visualize
def viz_sample_predictions(network, test_data, label_map, rows=5, cols=5, figsize=(10, 10)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize, num="Sample Predictions")
    axes = axes.reshape(-1)  # flatten

    test_data = test_data.to_buffer().shuffle()
    def sample_random():
        for j in np.arange(0, rows * cols):
            i = np.random.randint(0, len(test_data))
            x = mx.array(test_data[i]['image'])
            y = mx.array(test_data[i]['label'])
            y_pred = network.forward(x[mx.newaxis, ...])

            sample = np.array(255 * x)
            if sample.shape[2] == 3:
                image = Image.fromarray(sample.astype('uint8'))
            else:
                image = Image.fromarray(sample.reshape(sample.shape[0], sample.shape[1]))

            raw_label = mx.argmax(y).item()
            label = label_map[raw_label]

            raw_pred = mx.argmax(y_pred).item()
            pred = label_map[raw_pred]

            axes[j].imshow(image)
            axes[j].set_title(f"True: {label} \nPredict: {pred}")
            axes[j].axis('off')
            plt.subplots_adjust(wspace=1)

    def on_key(event):
        if event.key == ' ':
            sample_random()
            fig.show()

    fig.canvas.mpl_connect('key_press_event', on_key)

    sample_random()


def viz_history(history, figsize=(6, 4)):
    plt.figure(figsize=figsize, num="Loss Curves")
    plt.plot(history['epoch'], history['train_loss'], 'black', linewidth=2.0)
    plt.plot(history['epoch'], history['test_loss'], 'green', linewidth=2.0)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Loss vs Epoch', fontsize=12)

    plt.figure(figsize=figsize, num="Accuracy Curves")
    plt.plot(history['epoch'], history['train_accuracy'], 'black', linewidth=2.0)
    plt.plot(history['epoch'], history['test_accuracy'], 'green', linewidth=2.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Accuracy vs Epoch', fontsize=12)


# Evaluate
def eval_model(model, dataset, epoch=None):
    mean_losses = []
    accuracies = []
    predictions = []

    for batch in dataset:
        x_batch = mx.array(batch["image"])
        y_batch = mx.array(batch["label"])

        y_pred = model.forward(x_batch)
        predictions.append(y_pred)

        loss = optimizer.loss_fn(y_pred, y_batch)
        mean_loss = mx.mean(mx.sum(loss, axis=1))
        mean_losses.append(mean_loss.item())

        if isinstance(optimizer.loss_fn, CrossEntropyLoss):
            y_pred = softmax(y_pred)

        errors = mx.sum(mx.abs(y_batch - mx.round(y_pred)), axis=1)
        accuracy = mx.sum(errors == 0) / y_batch.shape[0]
        accuracies.append(accuracy.item())

    mean_loss = sum(mean_losses) / len(mean_losses)
    accuracy = sum(accuracies) / len(accuracies)
    predictions = np.concatenate(predictions)

    dataset.reset()

    if epoch is not None:
        print(f"Epoch {epoch}: Accuracy {accuracy:.3f}, Average Loss {mean_loss}")
    else:
        print(f"Accuracy {accuracy:.3f}, Average Loss {mean_loss}")

    return predictions, accuracy, mean_loss


def train(train_data, epochs, batch_size=1, test_data=None, cb=None):
    batched_train_data = train_data.batch(batch_size)
    batched_test_data = test_data.batch(batch_size)

    def train_epoch():
        for batch in batched_train_data:
            x_batch = mx.array(batch["image"])
            y_batch = mx.array(batch["label"])
            optimizer.step(x_batch, y_batch)
        batched_train_data.reset()

    history = {"epoch": [], "train_loss": [], "test_loss": [], "train_accuracy": [], "test_accuracy": []}

    _, train_accuracy, train_loss = eval_model(network, batched_train_data, epoch=0)
    _, test_accuracy, test_loss = eval_model(network, batched_test_data, epoch=0)
    print(f"{datetime.now()}")
    print()
    history["epoch"].append(0)
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    history["train_accuracy"].append(train_accuracy)
    history["test_accuracy"].append(test_accuracy)

    for epoch in range(1, epochs + 1):
        train_epoch()

        _, train_accuracy, train_loss = eval_model(network, batched_train_data, epoch=epoch)
        _, test_accuracy, test_loss = eval_model(network, batched_test_data, epoch=epoch)
        print(f"{datetime.now()}")
        print()
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)

    test_data.reset()
    eval_model(network, batched_test_data)
    print()

    viz_sample_predictions(network, test_data, label_map)
    viz_history(history)
    plt.show()


# train_data, test_data = get_mnist(static=True)
# label_map = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]

# train_data, test_data = get_fashion_mnist(static=True)
# label_map = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

train_data, test_data = get_cifar10(static=False)
label_map = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


def build_resnet_cifar10():
    """
    Build ResNet for CIFAR-10 with padding and projection shortcuts.

    Architecture:
    - Input: 32x32x3
    - Conv1 + BN + ReLU: 32x32x32
    - 2x ResBlock (identity): 32x32x32
    - MaxPool: 16x16x32
    - ResBlock (projection): 16x16x64
    - ResBlock (identity): 16x16x64
    - MaxPool: 8x8x64
    - ResBlock (projection): 8x8x128
    - ResBlock (identity): 8x8x128
    - MaxPool: 4x4x128
    - Flatten + FC: 2048 -> 256 -> 10
    """
    
    network = Network(input_shape=(32, 32, 3))

    # Initial convolution (with padding to preserve 32x32)
    network.add_layer(MLX(nn.Conv2d(3, 32, kernel_size=3, padding=1)))  # 32x32x3 -> 32x32x64
    network.add_layer(BatchNorm())
    network.add_layer(Activation(Relu()))

    # === Stage 1: 32x32x32 ===

    # ResBlock 1a: Identity (32->32, no projection needed)
    network.add_layer(Residual(Sequential([
        MLX(nn.Conv2d(32, 32, kernel_size=3, padding=1)),  # Padding preserves 32x32
        BatchNorm(),
        Activation(Relu()),
        MLX(nn.Conv2d(32, 32, kernel_size=3, padding=1)),  # Padding preserves 32x32
        BatchNorm()
    ])))
    network.add_layer(Activation(Relu()))

    # ResBlock 1b: Identity (64->64, no projection needed)
    network.add_layer(Residual(Sequential([
        MLX(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        BatchNorm(),
        Activation(Relu()),
        MLX(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        BatchNorm()
    ])))
    network.add_layer(Activation(Relu()))

    # Downsample: 32x32 -> 16x16
    network.add_layer(MaxPool2d(2))

    # === Stage 2: 16x16x64 ===

    # ResBlock 2a: Projection (32->64, needs 1x1 conv projection)
    network.add_layer(ProjectionResidual(Sequential([
        MLX(nn.Conv2d(32, 64, kernel_size=3, padding=1)),  # 16x16x64 -> 16x16x128
        BatchNorm(),
        Activation(Relu()),
        MLX(nn.Conv2d(64, 64, kernel_size=3, padding=1)),  # 16x16x128 -> 16x16x128
        BatchNorm()
    ])))
    network.add_layer(Activation(Relu()))

    # ResBlock 2b: Identity (64->64, no projection needed)
    network.add_layer(Residual(Sequential([
        MLX(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
        BatchNorm(),
        Activation(Relu()),
        MLX(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
        BatchNorm()
    ])))
    network.add_layer(Activation(Relu()))

    # Downsample: 16x16 -> 8x8
    network.add_layer(MaxPool2d(2))

    # === Stage 3: 8x8x128 ===

    # ResBlock 3a: Projection (64->128, needs 1x1 conv projection)
    network.add_layer(ProjectionResidual(Sequential([
        MLX(nn.Conv2d(64, 128, kernel_size=3, padding=1)),  # 8x8x128 -> 8x8x256
        BatchNorm(),
        Activation(Relu()),
        MLX(nn.Conv2d(128, 128, kernel_size=3, padding=1)),  # 8x8x256 -> 8x8x256
        BatchNorm()
    ])))
    network.add_layer(Activation(Relu()))

    # ResBlock 3b: Identity (128->128, no projection needed)
    network.add_layer(Residual(Sequential([
        MLX(nn.Conv2d(128, 128, kernel_size=3, padding=1)),
        BatchNorm(),
        Activation(Relu()),
        MLX(nn.Conv2d(128, 128, kernel_size=3, padding=1)),
        BatchNorm()
    ])))
    network.add_layer(Activation(Relu()))

    # Downsample: 8x8 -> 4x4
    network.add_layer(MaxPool2d(2))

    # === Classification Head ===

    # Flatten: 4x4x128 -> 2048
    network.add_layer(Flatten())
    network.add_layer(Linear(256))
    network.add_layer(Activation(Relu()))
    network.add_layer(Linear(10))  # 10 classes for CIFAR-10

    return network


def build_tiny_resnet_cifar10():
    """
    Tiny ResNet for CIFAR-10 - for quick testing and debugging.

    Architecture:
    - Input: 32x32x3
    - Conv1: 32x32x16
    - 1x ResBlock: 32x32x16
    - MaxPool: 16x16x16
    - 1x ResBlock (projection): 16x16x32
    - MaxPool: 8x8x32
    - Flatten + FC: 2048 -> 10

    Parameters: ~50K (vs 3.8M in full version)
    """
    network = Network(input_shape=(32, 32, 3))

    # Initial convolution: 32x32x3 -> 32x32x16
    network.add_layer(Conv2d(3, 16, kernel_size=3, padding=1))
    network.add_layer(BatchNorm())
    network.add_layer(Activation(Relu()))

    # Stage 1: One residual block at 32x32x16
    network.add_layer(Residual(Sequential([
        MLX(nn.Conv2d(16, 16, kernel_size=3, padding=1)),
        BatchNorm(),
        Activation(Relu()),
        MLX(nn.Conv2d(16, 16, kernel_size=3, padding=1)),
        BatchNorm()
    ])))
    network.add_layer(Activation(Relu()))

    # Downsample: 32x32 -> 16x16
    network.add_layer(MaxPool2d(2))

    # Stage 2: One projection block 16->32 at 16x16
    network.add_layer(ProjectionResidual(Sequential([
        MLX(nn.Conv2d(16, 32, kernel_size=3, padding=1)),
        BatchNorm(),
        Activation(Relu()),
        MLX(nn.Conv2d(32, 32, kernel_size=3, padding=1)),
        BatchNorm()
    ])))
    network.add_layer(Activation(Relu()))

    # Downsample: 16x16 -> 8x8
    network.add_layer(MaxPool2d(2))

    # Classification head: 8x8x32 = 2048
    network.add_layer(Flatten())
    network.add_layer(Linear(256))
    network.add_layer(Activation(Relu()))
    network.add_layer(Linear(10))  # Direct to 10 classes

    return network


def build_minimal_cnn_cifar10():
    """
    Ultra-minimal CNN - no residual connections at all.
    Use this to isolate which component is slow.

    Parameters: ~10K
    """
    network = Network(input_shape=(32, 32, 3))

    # Simple conv layers
    network.add_layer(Conv2d(3, 16, kernel_size=3, padding=1))  # 32x32x16
    # network.add_layer(MLX(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)))
    network.add_layer(Activation(Relu()))
    network.add_layer(MaxPool2d(2))  # 16x16x16
    # network.add_layer(MLX(nn.MaxPool2d(2)))

    network.add_layer(Conv2d(16, 32, kernel_size=3, padding=1))  # 16x16x32
    # network.add_layer(MLX(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)))
    network.add_layer(Activation(Relu()))
    network.add_layer(MaxPool2d(2))  # 8x8x32
    # network.add_layer(MLX(nn.MaxPool2d(2)))

    network.add_layer(Flatten())
    network.add_layer(Linear(10))

    return network


network = build_resnet_cifar10()
# network = build_tiny_resnet_cifar10()
# network = build_minimal_cnn_cifar10()

optimizer = SGD(eta=0.05, momentum=0.9, weight_decay=0.0005)
optimizer.bind_loss_fn(cross_entropy_loss)
optimizer.bind_network(network)

train(train_data, epochs=100, batch_size=128, test_data=test_data)





# Epoch 0: Accuracy 0.000, Average Loss 2.3878236961364747
# Epoch 0: Accuracy 0.000, Average Loss 2.3970799446105957
# 2025-12-16 23:33:21.561488
#
# Epoch 1: Accuracy 0.181, Average Loss 1.536809253692627
# Epoch 1: Accuracy 0.224, Average Loss 1.4783800482749938
# 2025-12-16 23:33:33.597135
#
# Epoch 2: Accuracy 0.286, Average Loss 1.3489913725852967
# Epoch 2: Accuracy 0.341, Average Loss 1.2873443722724915
# 2025-12-16 23:33:45.592386
#
# Epoch 3: Accuracy 0.374, Average Loss 1.2284372520446778
# Epoch 3: Accuracy 0.408, Average Loss 1.1946404457092286
# 2025-12-16 23:33:57.552443
#
# Epoch 4: Accuracy 0.436, Average Loss 1.1453552961349487
# Epoch 4: Accuracy 0.469, Average Loss 1.1168787956237793
# 2025-12-16 23:34:09.600285
#
# Epoch 5: Accuracy 0.471, Average Loss 1.073995382785797
# Epoch 5: Accuracy 0.510, Average Loss 1.0276377379894257
# 2025-12-16 23:34:21.588240
#
# Epoch 6: Accuracy 0.496, Average Loss 1.0207129776477815
# Epoch 6: Accuracy 0.537, Average Loss 0.9689814805984497
# 2025-12-16 23:34:33.698442
#
# Epoch 7: Accuracy 0.515, Average Loss 1.0015710496902466
# Epoch 7: Accuracy 0.553, Average Loss 0.9498353064060211
# 2025-12-16 23:34:45.764489
#
# Epoch 8: Accuracy 0.544, Average Loss 0.9328662264347076
# Epoch 8: Accuracy 0.571, Average Loss 0.8985560953617096
# 2025-12-16 23:34:57.659023
#
# Epoch 9: Accuracy 0.570, Average Loss 0.8891490733623505
# Epoch 9: Accuracy 0.602, Average Loss 0.8541817724704742
# 2025-12-16 23:35:09.601231
#
# Epoch 10: Accuracy 0.580, Average Loss 0.8723116207122803
# Epoch 10: Accuracy 0.610, Average Loss 0.8345556914806366
# 2025-12-16 23:35:21.647526
#
# Epoch 11: Accuracy 0.600, Average Loss 0.8421325957775116
# Epoch 11: Accuracy 0.628, Average Loss 0.8016644299030304
# 2025-12-16 23:35:33.753248
#
# Epoch 12: Accuracy 0.612, Average Loss 0.8191316092014312
# Epoch 12: Accuracy 0.633, Average Loss 0.7883911073207855
# 2025-12-16 23:35:45.870624
#
# Epoch 13: Accuracy 0.622, Average Loss 0.7890808594226837
# Epoch 13: Accuracy 0.642, Average Loss 0.7630502939224243
# 2025-12-16 23:35:57.819757
#
# Epoch 14: Accuracy 0.638, Average Loss 0.7708827185630799
# Epoch 14: Accuracy 0.654, Average Loss 0.7522406339645386
# 2025-12-16 23:36:09.810656
#
# Epoch 15: Accuracy 0.646, Average Loss 0.7512317037582398
# Epoch 15: Accuracy 0.661, Average Loss 0.7390144765377045
# 2025-12-16 23:36:21.878653
#
# Epoch 16: Accuracy 0.661, Average Loss 0.7276965701580047
# Epoch 16: Accuracy 0.669, Average Loss 0.7169862926006317
# 2025-12-16 23:36:33.947507
#
# Epoch 17: Accuracy 0.662, Average Loss 0.7188396108150482
# Epoch 17: Accuracy 0.666, Average Loss 0.721380352973938
# 2025-12-16 23:36:46.074555
#
# Epoch 18: Accuracy 0.670, Average Loss 0.7113757634162903
# Epoch 18: Accuracy 0.677, Average Loss 0.7028112530708313
# 2025-12-16 23:36:58.036702
#
# Epoch 19: Accuracy 0.681, Average Loss 0.6913924217224121
# Epoch 19: Accuracy 0.686, Average Loss 0.6922819912433624
# 2025-12-16 23:37:10.024269
#
# Epoch 20: Accuracy 0.684, Average Loss 0.6856132674217225
# Epoch 20: Accuracy 0.685, Average Loss 0.6916674792766571
# 2025-12-16 23:37:22.143207
#
# Epoch 21: Accuracy 0.695, Average Loss 0.664209532737732
# Epoch 21: Accuracy 0.690, Average Loss 0.6821536362171173
# 2025-12-16 23:37:34.355647
#
# Epoch 22: Accuracy 0.698, Average Loss 0.6631007790565491
# Epoch 22: Accuracy 0.698, Average Loss 0.6712006151676178
# 2025-12-16 23:37:46.459623
#
# Epoch 23: Accuracy 0.698, Average Loss 0.6633051979541779
# Epoch 23: Accuracy 0.691, Average Loss 0.6885047674179077
# 2025-12-16 23:37:58.433209
#
# Epoch 24: Accuracy 0.711, Average Loss 0.6389949834346771
# Epoch 24: Accuracy 0.711, Average Loss 0.6569840908050537
# 2025-12-16 23:38:10.458569
#
# Epoch 25: Accuracy 0.715, Average Loss 0.6311036264896392
# Epoch 25: Accuracy 0.721, Average Loss 0.6374032676219941
# 2025-12-16 23:38:22.592850
#
# Epoch 26: Accuracy 0.719, Average Loss 0.6165428042411805
# Epoch 26: Accuracy 0.718, Average Loss 0.6345078825950623
# 2025-12-16 23:38:34.827807
#
# Epoch 27: Accuracy 0.729, Average Loss 0.603079115152359
# Epoch 27: Accuracy 0.724, Average Loss 0.6298355758190155
# 2025-12-16 23:38:46.910396
#
# Epoch 28: Accuracy 0.731, Average Loss 0.5995101761817933
# Epoch 28: Accuracy 0.727, Average Loss 0.6288614392280578
# 2025-12-16 23:38:58.863807
#
# Epoch 29: Accuracy 0.734, Average Loss 0.5910204815864563
# Epoch 29: Accuracy 0.733, Average Loss 0.6211180746555328
# 2025-12-16 23:39:10.934930
#
# Epoch 30: Accuracy 0.739, Average Loss 0.5814471209049225
# Epoch 30: Accuracy 0.737, Average Loss 0.6115723013877868
# 2025-12-16 23:39:23.090715
#
# Epoch 31: Accuracy 0.741, Average Loss 0.5800284385681153
# Epoch 31: Accuracy 0.737, Average Loss 0.6105065047740936
# 2025-12-16 23:39:35.143861
#
# Epoch 32: Accuracy 0.743, Average Loss 0.5772187435626983
# Epoch 32: Accuracy 0.740, Average Loss 0.6056788265705109
# 2025-12-16 23:39:47.169133
#
# Epoch 33: Accuracy 0.746, Average Loss 0.5791151630878448
# Epoch 33: Accuracy 0.742, Average Loss 0.6066583693027496
# 2025-12-16 23:39:59.386894
#
# Epoch 34: Accuracy 0.750, Average Loss 0.5595126569271087
# Epoch 34: Accuracy 0.744, Average Loss 0.5939499497413635
# 2025-12-16 23:40:11.938763
#
# Epoch 35: Accuracy 0.749, Average Loss 0.5730531942844391
# Epoch 35: Accuracy 0.745, Average Loss 0.6006378829479218
# 2025-12-16 23:40:24.049857
#
# Epoch 36: Accuracy 0.759, Average Loss 0.5464904928207397
# Epoch 36: Accuracy 0.754, Average Loss 0.587732458114624
# 2025-12-16 23:40:36.087386
#
# Epoch 37: Accuracy 0.759, Average Loss 0.5438936167955398
# Epoch 37: Accuracy 0.751, Average Loss 0.5880672633647919
# 2025-12-16 23:40:48.076456
#
# Epoch 38: Accuracy 0.763, Average Loss 0.5405130231380463
# Epoch 38: Accuracy 0.749, Average Loss 0.5798226892948151
# 2025-12-16 23:41:00.074581
#
# Epoch 39: Accuracy 0.768, Average Loss 0.5241562044620514
# Epoch 39: Accuracy 0.754, Average Loss 0.5716970920562744
# 2025-12-16 23:41:12.198883
#
# Epoch 40: Accuracy 0.771, Average Loss 0.5203417980670929
# Epoch 40: Accuracy 0.763, Average Loss 0.5647498309612274
# 2025-12-16 23:41:24.307594
#
# Epoch 41: Accuracy 0.772, Average Loss 0.5207567840814591
# Epoch 41: Accuracy 0.757, Average Loss 0.5724258065223694
# 2025-12-16 23:41:36.374063
#
# Epoch 42: Accuracy 0.773, Average Loss 0.5143599390983582
# Epoch 42: Accuracy 0.758, Average Loss 0.5666775286197663
# 2025-12-16 23:41:48.386853
#
# Epoch 43: Accuracy 0.777, Average Loss 0.5058855581283569
# Epoch 43: Accuracy 0.763, Average Loss 0.5611522793769836
# 2025-12-16 23:42:00.428822
#
# Epoch 44: Accuracy 0.777, Average Loss 0.5130619490146637
# Epoch 44: Accuracy 0.765, Average Loss 0.5577761828899384
# 2025-12-16 23:42:12.625700
#
# Epoch 45: Accuracy 0.781, Average Loss 0.4998840773105621
# Epoch 45: Accuracy 0.768, Average Loss 0.5613275289535522
# 2025-12-16 23:42:24.726988
#
# Epoch 46: Accuracy 0.785, Average Loss 0.4995611023902893
# Epoch 46: Accuracy 0.770, Average Loss 0.5540822386741638
# 2025-12-16 23:42:36.709906
#
# Epoch 47: Accuracy 0.780, Average Loss 0.5039605516195297
# Epoch 47: Accuracy 0.769, Average Loss 0.5609137177467346
# 2025-12-16 23:42:48.769362
#
# Epoch 48: Accuracy 0.785, Average Loss 0.49338057816028597
# Epoch 48: Accuracy 0.769, Average Loss 0.5615080654621124
# 2025-12-16 23:43:00.872011
#
# Epoch 49: Accuracy 0.789, Average Loss 0.47845917582511904
# Epoch 49: Accuracy 0.773, Average Loss 0.5448439836502075
# 2025-12-16 23:43:13.546799
#
# Epoch 50: Accuracy 0.786, Average Loss 0.49409462928771974
# Epoch 50: Accuracy 0.767, Average Loss 0.5636634409427643
# 2025-12-16 23:43:25.927439
#
# Epoch 51: Accuracy 0.794, Average Loss 0.4726226806640625
# Epoch 51: Accuracy 0.773, Average Loss 0.5445248126983643
# 2025-12-16 23:43:38.051833
#
# Epoch 52: Accuracy 0.790, Average Loss 0.47803321659564973
# Epoch 52: Accuracy 0.770, Average Loss 0.5479773044586181
# 2025-12-16 23:43:50.138407
#
# Epoch 53: Accuracy 0.794, Average Loss 0.47027849197387694
# Epoch 53: Accuracy 0.768, Average Loss 0.5588865458965302
# 2025-12-16 23:44:02.315782
#
# Epoch 54: Accuracy 0.797, Average Loss 0.4644635301828384
# Epoch 54: Accuracy 0.775, Average Loss 0.5492976605892181
# 2025-12-16 23:44:14.591742
#
# Epoch 55: Accuracy 0.799, Average Loss 0.4647639578580856
# Epoch 55: Accuracy 0.774, Average Loss 0.5537640571594238
# 2025-12-16 23:44:26.650494
#
# Epoch 56: Accuracy 0.796, Average Loss 0.4692451423406601
# Epoch 56: Accuracy 0.774, Average Loss 0.547060227394104
# 2025-12-16 23:44:38.810020
#
# Epoch 57: Accuracy 0.800, Average Loss 0.46427099108695985
# Epoch 57: Accuracy 0.778, Average Loss 0.5430551111698151
# 2025-12-16 23:44:50.934548
#
# Epoch 58: Accuracy 0.801, Average Loss 0.45711080968379975
# Epoch 58: Accuracy 0.773, Average Loss 0.5484555900096894
# 2025-12-16 23:45:03.148354
#
# Epoch 59: Accuracy 0.801, Average Loss 0.45626101672649383
# Epoch 59: Accuracy 0.779, Average Loss 0.541111096739769
# 2025-12-16 23:45:15.409629
#
# Epoch 60: Accuracy 0.805, Average Loss 0.450824590921402
# Epoch 60: Accuracy 0.778, Average Loss 0.544403326511383
# 2025-12-16 23:45:27.484480
#
# Epoch 61: Accuracy 0.804, Average Loss 0.4457694500684738
# Epoch 61: Accuracy 0.778, Average Loss 0.541935133934021
# 2025-12-16 23:45:39.558150
#
# Epoch 62: Accuracy 0.805, Average Loss 0.4469910722970962
# Epoch 62: Accuracy 0.769, Average Loss 0.5651351988315583
# 2025-12-16 23:45:51.739664
#
# Epoch 63: Accuracy 0.811, Average Loss 0.4355882573127747
# Epoch 63: Accuracy 0.788, Average Loss 0.5274522483348847
# 2025-12-16 23:46:04.024249
#
# Epoch 64: Accuracy 0.813, Average Loss 0.424167343378067
# Epoch 64: Accuracy 0.782, Average Loss 0.5373016566038131
# 2025-12-16 23:46:16.239386
#
# Epoch 65: Accuracy 0.815, Average Loss 0.42877495527267456
# Epoch 65: Accuracy 0.786, Average Loss 0.5318762540817261
# 2025-12-16 23:46:28.365832
#
# Epoch 66: Accuracy 0.816, Average Loss 0.42501271605491636
# Epoch 66: Accuracy 0.787, Average Loss 0.527976569533348
# 2025-12-16 23:46:41.003492
#
# Epoch 67: Accuracy 0.820, Average Loss 0.4200869160890579
# Epoch 67: Accuracy 0.796, Average Loss 0.517764613032341
# 2025-12-16 23:46:53.375139
#
# Epoch 68: Accuracy 0.821, Average Loss 0.415711727142334
# Epoch 68: Accuracy 0.787, Average Loss 0.5345859080553055
# 2025-12-16 23:47:05.463713
#
# Epoch 69: Accuracy 0.818, Average Loss 0.421553515791893
# Epoch 69: Accuracy 0.784, Average Loss 0.5332637667655945
# 2025-12-16 23:47:17.565081
#
# Epoch 70: Accuracy 0.818, Average Loss 0.4225474578142166
# Epoch 70: Accuracy 0.784, Average Loss 0.5398734629154205
# 2025-12-16 23:47:29.627302
#
# Epoch 71: Accuracy 0.822, Average Loss 0.41245874226093293
# Epoch 71: Accuracy 0.787, Average Loss 0.5320187836885453
# 2025-12-16 23:47:41.672927
#
# Epoch 72: Accuracy 0.824, Average Loss 0.4150648754835129
# Epoch 72: Accuracy 0.792, Average Loss 0.5215415954589844
# 2025-12-16 23:47:53.721814
#
# Epoch 73: Accuracy 0.821, Average Loss 0.41347881972789763
# Epoch 73: Accuracy 0.784, Average Loss 0.5372830003499984
# 2025-12-16 23:48:05.708922
#
# Epoch 74: Accuracy 0.821, Average Loss 0.40598000049591065
# Epoch 74: Accuracy 0.788, Average Loss 0.5328147768974304
# 2025-12-16 23:48:17.755814
#
# Epoch 75: Accuracy 0.827, Average Loss 0.39848814010620115
# Epoch 75: Accuracy 0.791, Average Loss 0.5138240605592728
# 2025-12-16 23:48:29.823098
#
# Epoch 76: Accuracy 0.818, Average Loss 0.4170483547449112
# Epoch 76: Accuracy 0.780, Average Loss 0.5465898036956787
# 2025-12-16 23:48:41.959477
#
# Epoch 77: Accuracy 0.827, Average Loss 0.40687919676303863
# Epoch 77: Accuracy 0.794, Average Loss 0.5287404119968414
# 2025-12-16 23:48:54.083258
#
# Epoch 78: Accuracy 0.827, Average Loss 0.4015543609857559
# Epoch 78: Accuracy 0.791, Average Loss 0.5250610291957856
# 2025-12-16 23:49:06.107238
#
# Epoch 79: Accuracy 0.824, Average Loss 0.404433890581131
# Epoch 79: Accuracy 0.796, Average Loss 0.5224721550941467
# 2025-12-16 23:49:18.150257
#
# Epoch 80: Accuracy 0.833, Average Loss 0.3888774424791336
# Epoch 80: Accuracy 0.792, Average Loss 0.5274093478918076
# 2025-12-16 23:49:30.193620
#
# Epoch 81: Accuracy 0.825, Average Loss 0.40706118285655973
# Epoch 81: Accuracy 0.787, Average Loss 0.5435449302196502
# 2025-12-16 23:49:42.361103
#
# Epoch 82: Accuracy 0.825, Average Loss 0.40878864586353303
# Epoch 82: Accuracy 0.795, Average Loss 0.5326241612434387
# 2025-12-16 23:49:54.503069
#
# Epoch 83: Accuracy 0.829, Average Loss 0.39695594131946565
# Epoch 83: Accuracy 0.796, Average Loss 0.5232338815927505
# 2025-12-16 23:50:06.520443
#
# Epoch 84: Accuracy 0.830, Average Loss 0.3969815355539322
# Epoch 84: Accuracy 0.795, Average Loss 0.5235252916812897
# 2025-12-16 23:50:18.666481
#
# Epoch 85: Accuracy 0.828, Average Loss 0.39606105506420136
# Epoch 85: Accuracy 0.794, Average Loss 0.535141259431839
# 2025-12-16 23:50:30.757993
#
# Epoch 86: Accuracy 0.835, Average Loss 0.38937445759773254
# Epoch 86: Accuracy 0.793, Average Loss 0.5355280637741089
# 2025-12-16 23:50:43.014741
#
# Epoch 87: Accuracy 0.834, Average Loss 0.3873615300655365
# Epoch 87: Accuracy 0.796, Average Loss 0.5247216761112213
# 2025-12-16 23:50:55.152260
#
# Epoch 88: Accuracy 0.833, Average Loss 0.38865217328071594
# Epoch 88: Accuracy 0.793, Average Loss 0.5265263378620147
# 2025-12-16 23:51:07.183989
#
# Epoch 89: Accuracy 0.837, Average Loss 0.3814611166715622
# Epoch 89: Accuracy 0.796, Average Loss 0.52188580930233
# 2025-12-16 23:51:19.246421
#
# Epoch 90: Accuracy 0.836, Average Loss 0.37850500881671906
# Epoch 90: Accuracy 0.795, Average Loss 0.5306227236986161
# 2025-12-16 23:51:31.395771
#
# Epoch 91: Accuracy 0.833, Average Loss 0.3905739325284958
# Epoch 91: Accuracy 0.793, Average Loss 0.5331213176250458
# 2025-12-16 23:51:43.757010
#
# Epoch 92: Accuracy 0.843, Average Loss 0.36819047152996065
# Epoch 92: Accuracy 0.799, Average Loss 0.512345477938652
# 2025-12-16 23:51:55.841210
#
# Epoch 93: Accuracy 0.843, Average Loss 0.3665289044380188
# Epoch 93: Accuracy 0.799, Average Loss 0.5210216403007507
# 2025-12-16 23:52:07.908021
#
# Epoch 94: Accuracy 0.847, Average Loss 0.36474002301692965
# Epoch 94: Accuracy 0.806, Average Loss 0.5162823289632797
# 2025-12-16 23:52:19.959202
#
# Epoch 95: Accuracy 0.848, Average Loss 0.3586702412366867
# Epoch 95: Accuracy 0.804, Average Loss 0.5089481741189956
# 2025-12-16 23:52:32.154915
#
# Epoch 96: Accuracy 0.842, Average Loss 0.3689343571662903
# Epoch 96: Accuracy 0.797, Average Loss 0.528161633014679
# 2025-12-16 23:52:44.562418
#
# Epoch 97: Accuracy 0.847, Average Loss 0.35793533861637117
# Epoch 97: Accuracy 0.797, Average Loss 0.5247050195932388
# 2025-12-16 23:52:56.721722
#
# Epoch 98: Accuracy 0.841, Average Loss 0.37622842252254485
# Epoch 98: Accuracy 0.800, Average Loss 0.5266700088977814
# 2025-12-16 23:53:08.743926
#
# Epoch 99: Accuracy 0.846, Average Loss 0.35905235290527343
# Epoch 99: Accuracy 0.803, Average Loss 0.5193065494298935
# 2025-12-16 23:53:20.903272
#
# Epoch 100: Accuracy 0.846, Average Loss 0.35801849126815793
# Epoch 100: Accuracy 0.804, Average Loss 0.5195705264806747
# 2025-12-16 23:53:33.153111
#
# Accuracy 0.804, Average Loss 0.5195705264806747








# /Users/apetrusca/alpine/_ai_/softgrad/.venv/bin/python /Users/apetrusca/alpine/_ai_/softgrad/examples/resnet_test.py
# Epoch 0: Accuracy 0.000, Average Loss 2.399763298034668
# Epoch 0: Accuracy 0.000, Average Loss 2.401701641082764
# 2025-12-17 10:36:31.128409
#
# Epoch 1: Accuracy 0.236, Average Loss 1.5185773825645448
# Epoch 1: Accuracy 0.261, Average Loss 1.4828368782997132
# 2025-12-17 10:37:24.646979
#
# Epoch 2: Accuracy 0.431, Average Loss 1.1891379499435424
# Epoch 2: Accuracy 0.434, Average Loss 1.205446982383728
# 2025-12-17 10:38:17.884121
#
# Epoch 3: Accuracy 0.509, Average Loss 1.03972771525383
# Epoch 3: Accuracy 0.504, Average Loss 1.0676378726959228
# 2025-12-17 10:39:11.867294
#
# Epoch 4: Accuracy 0.592, Average Loss 0.860174013376236
# Epoch 4: Accuracy 0.588, Average Loss 0.8921421766281128
# 2025-12-17 10:40:05.220416
#
# Epoch 5: Accuracy 0.650, Average Loss 0.7279231441020966
# Epoch 5: Accuracy 0.628, Average Loss 0.7934684693813324
# 2025-12-17 10:40:59.856261
#
# Epoch 6: Accuracy 0.700, Average Loss 0.6505075669288636
# Epoch 6: Accuracy 0.684, Average Loss 0.7068560302257538
# 2025-12-17 10:41:55.278633
#
# Epoch 7: Accuracy 0.732, Average Loss 0.5934058797359466
# Epoch 7: Accuracy 0.709, Average Loss 0.6646127581596375
# 2025-12-17 10:42:52.086175
#
# Epoch 8: Accuracy 0.756, Average Loss 0.5465143877267837
# Epoch 8: Accuracy 0.732, Average Loss 0.6296410202980042
# 2025-12-17 10:43:47.327894
#
# Epoch 9: Accuracy 0.780, Average Loss 0.5082849872112274
# Epoch 9: Accuracy 0.749, Average Loss 0.6009266316890717
# 2025-12-17 10:44:40.810131
#
# Epoch 10: Accuracy 0.793, Average Loss 0.479155570268631
# Epoch 10: Accuracy 0.763, Average Loss 0.5751410245895385
# 2025-12-17 10:45:34.769019
#
# Epoch 11: Accuracy 0.802, Average Loss 0.4600864851474762
# Epoch 11: Accuracy 0.770, Average Loss 0.567835658788681
# 2025-12-17 10:46:28.446371
#
# Epoch 12: Accuracy 0.821, Average Loss 0.42903706431388855
# Epoch 12: Accuracy 0.781, Average Loss 0.5530341595411301
# 2025-12-17 10:47:22.644128
#
# Epoch 13: Accuracy 0.834, Average Loss 0.3903673714399338
# Epoch 13: Accuracy 0.795, Average Loss 0.5051813185214996
# 2025-12-17 10:48:16.649001
#
# Epoch 14: Accuracy 0.836, Average Loss 0.3800062024593353
# Epoch 14: Accuracy 0.791, Average Loss 0.5048970103263855
# 2025-12-17 10:49:10.916579
#
# Epoch 15: Accuracy 0.861, Average Loss 0.33227994084358214
# Epoch 15: Accuracy 0.816, Average Loss 0.46857639849185945
# 2025-12-17 10:50:04.774028
#
# Epoch 16: Accuracy 0.858, Average Loss 0.3417918372154236
# Epoch 16: Accuracy 0.816, Average Loss 0.4789359211921692
# 2025-12-17 10:50:59.026619
#
# Epoch 17: Accuracy 0.862, Average Loss 0.3307399809360504
# Epoch 17: Accuracy 0.814, Average Loss 0.48986607491970063
# 2025-12-17 10:51:52.640340
#
# Epoch 18: Accuracy 0.874, Average Loss 0.3059078195691109
# Epoch 18: Accuracy 0.818, Average Loss 0.48979044854640963
# 2025-12-17 10:52:46.862046
#
# Epoch 19: Accuracy 0.879, Average Loss 0.29568975627422334
# Epoch 19: Accuracy 0.825, Average Loss 0.46802301704883575
# 2025-12-17 10:53:41.096879
#
# Epoch 20: Accuracy 0.883, Average Loss 0.2919000652432442
# Epoch 20: Accuracy 0.833, Average Loss 0.4554599553346634
# 2025-12-17 10:54:34.436844
#
# Epoch 21: Accuracy 0.883, Average Loss 0.2935939663648605
# Epoch 21: Accuracy 0.828, Average Loss 0.47691793739795685
# 2025-12-17 10:55:27.962130
#
# Epoch 22: Accuracy 0.886, Average Loss 0.28496628046035766
# Epoch 22: Accuracy 0.834, Average Loss 0.477019664645195
# 2025-12-17 10:56:21.982170
#
# Epoch 23: Accuracy 0.896, Average Loss 0.26153313994407656
# Epoch 23: Accuracy 0.836, Average Loss 0.4667419224977493
# 2025-12-17 10:57:16.290975
#
# Epoch 24: Accuracy 0.893, Average Loss 0.2613766080141067
# Epoch 24: Accuracy 0.831, Average Loss 0.46871417760849
# 2025-12-17 10:58:10.372834
#
# Epoch 25: Accuracy 0.903, Average Loss 0.24580485254526138
# Epoch 25: Accuracy 0.842, Average Loss 0.46917031705379486
# 2025-12-17 10:59:09.194022
#
# Epoch 26: Accuracy 0.913, Average Loss 0.21901365578174592
# Epoch 26: Accuracy 0.848, Average Loss 0.44593915343284607
# 2025-12-17 11:00:13.087192
#
# Epoch 27: Accuracy 0.906, Average Loss 0.23865731418132782
# Epoch 27: Accuracy 0.839, Average Loss 0.48495192229747774
# 2025-12-17 11:01:19.671723
#
# Epoch 28: Accuracy 0.912, Average Loss 0.21680707663297652
# Epoch 28: Accuracy 0.844, Average Loss 0.4710092663764954
# 2025-12-17 11:02:28.929510
#
# Epoch 29: Accuracy 0.919, Average Loss 0.19578304171562194
# Epoch 29: Accuracy 0.846, Average Loss 0.4454454243183136
# 2025-12-17 11:03:38.210229
#
# Epoch 30: Accuracy 0.918, Average Loss 0.19945539712905883
# Epoch 30: Accuracy 0.851, Average Loss 0.4381631463766098
# 2025-12-17 11:04:46.844104
#
# Epoch 31: Accuracy 0.921, Average Loss 0.20169068455696107
# Epoch 31: Accuracy 0.845, Average Loss 0.47120547890663145
# 2025-12-17 11:05:56.442747
#
# Epoch 32: Accuracy 0.928, Average Loss 0.17902175486087799
# Epoch 32: Accuracy 0.851, Average Loss 0.4490579843521118
# 2025-12-17 11:07:05.445423
#
# Epoch 33: Accuracy 0.931, Average Loss 0.1734014083445072
# Epoch 33: Accuracy 0.857, Average Loss 0.44978311359882356
# 2025-12-17 11:08:13.488922
#
# Epoch 34: Accuracy 0.935, Average Loss 0.16954993307590485
# Epoch 34: Accuracy 0.857, Average Loss 0.47252547442913057
# 2025-12-17 11:09:24.370185
#
# Epoch 35: Accuracy 0.934, Average Loss 0.16634514883160592
# Epoch 35: Accuracy 0.856, Average Loss 0.47009427547454835
# 2025-12-17 11:10:36.387218
#
# Epoch 36: Accuracy 0.934, Average Loss 0.1637691766023636
# Epoch 36: Accuracy 0.856, Average Loss 0.44683597087860105
# 2025-12-17 11:11:44.963701
#
# Epoch 37: Accuracy 0.938, Average Loss 0.1575833448767662
# Epoch 37: Accuracy 0.855, Average Loss 0.47846522331237795
# 2025-12-17 11:12:38.472289
#
# Epoch 38: Accuracy 0.941, Average Loss 0.14906472310423852
# Epoch 38: Accuracy 0.855, Average Loss 0.4613454341888428
# 2025-12-17 11:13:33.300007
#
# Epoch 39: Accuracy 0.938, Average Loss 0.1561955614387989
# Epoch 39: Accuracy 0.852, Average Loss 0.4648822218179703
# 2025-12-17 11:14:27.815558
#
# Epoch 40: Accuracy 0.946, Average Loss 0.13530653581023216
# Epoch 40: Accuracy 0.858, Average Loss 0.44781287014484406
# 2025-12-17 11:15:21.809470
#
# Epoch 41: Accuracy 0.944, Average Loss 0.13917971447110175
# Epoch 41: Accuracy 0.858, Average Loss 0.4446490824222565
# 2025-12-17 11:16:15.436530
#
# Epoch 42: Accuracy 0.945, Average Loss 0.13955775380134583
# Epoch 42: Accuracy 0.859, Average Loss 0.4708430618047714
# 2025-12-17 11:17:09.025117
#
# Epoch 43: Accuracy 0.952, Average Loss 0.12219816893339157
# Epoch 43: Accuracy 0.866, Average Loss 0.44661819338798525
# 2025-12-17 11:18:29.798099
#
# Epoch 44: Accuracy 0.949, Average Loss 0.12907622292637824
# Epoch 44: Accuracy 0.859, Average Loss 0.4797675132751465
# 2025-12-17 11:19:58.741744
#
# Epoch 45: Accuracy 0.950, Average Loss 0.12860046818852425
# Epoch 45: Accuracy 0.862, Average Loss 0.4771183133125305
# 2025-12-17 11:20:53.278071
#
# Epoch 46: Accuracy 0.955, Average Loss 0.1136617286503315
# Epoch 46: Accuracy 0.863, Average Loss 0.47692079842090607
# 2025-12-17 11:21:48.178563
#
# Epoch 47: Accuracy 0.959, Average Loss 0.10839168891310692
# Epoch 47: Accuracy 0.869, Average Loss 0.46323247253894806
# 2025-12-17 11:22:42.251696
#
# Epoch 48: Accuracy 0.958, Average Loss 0.1048437775671482
# Epoch 48: Accuracy 0.869, Average Loss 0.4597880125045776
# 2025-12-17 11:23:36.408120
#
# Epoch 49: Accuracy 0.960, Average Loss 0.1054905879497528
# Epoch 49: Accuracy 0.868, Average Loss 0.46037556529045104
# 2025-12-17 11:24:30.631966
#
# Epoch 50: Accuracy 0.957, Average Loss 0.10930463388562202
# Epoch 50: Accuracy 0.862, Average Loss 0.48702858984470365
# 2025-12-17 11:25:24.064543
#
# Epoch 51: Accuracy 0.958, Average Loss 0.10850019976496697
# Epoch 51: Accuracy 0.867, Average Loss 0.4729307770729065
# 2025-12-17 11:26:17.230279
#
# Epoch 52: Accuracy 0.955, Average Loss 0.11672230035066605
# Epoch 52: Accuracy 0.865, Average Loss 0.48235872089862825
# 2025-12-17 11:27:10.362393
#
# Epoch 53: Accuracy 0.962, Average Loss 0.0994389757514
# Epoch 53: Accuracy 0.866, Average Loss 0.5067284166812897
# 2025-12-17 11:28:03.689493
#
# Epoch 54: Accuracy 0.961, Average Loss 0.100523322224617
# Epoch 54: Accuracy 0.864, Average Loss 0.4990663081407547
# 2025-12-17 11:28:56.868376
#
# Epoch 55: Accuracy 0.968, Average Loss 0.08344742849469185
# Epoch 55: Accuracy 0.877, Average Loss 0.4638302892446518
# 2025-12-17 11:29:50.023726
#
# Epoch 56: Accuracy 0.962, Average Loss 0.09958713300526142
# Epoch 56: Accuracy 0.870, Average Loss 0.5079587697982788
# 2025-12-17 11:30:43.073706
#
# Epoch 57: Accuracy 0.964, Average Loss 0.09434141144156456
# Epoch 57: Accuracy 0.871, Average Loss 0.5077556669712067
# 2025-12-17 11:31:36.038979
#
# Epoch 58: Accuracy 0.966, Average Loss 0.08750909656286239
# Epoch 58: Accuracy 0.876, Average Loss 0.4814422994852066
# 2025-12-17 11:32:35.795101
#
# Epoch 59: Accuracy 0.961, Average Loss 0.1052808864414692
# Epoch 59: Accuracy 0.866, Average Loss 0.5417987912893295
# 2025-12-17 11:33:54.281276
#
# Epoch 60: Accuracy 0.958, Average Loss 0.10980910524725913
# Epoch 60: Accuracy 0.865, Average Loss 0.5145908981561661
# 2025-12-17 11:34:47.647012
#
# Epoch 61: Accuracy 0.967, Average Loss 0.08512763872742653
# Epoch 61: Accuracy 0.875, Average Loss 0.49939723312854767
# 2025-12-17 11:35:40.860413
#
# Epoch 62: Accuracy 0.964, Average Loss 0.09597448572516441
# Epoch 62: Accuracy 0.864, Average Loss 0.5420180648565293
# 2025-12-17 11:36:34.403071
#
# Epoch 63: Accuracy 0.968, Average Loss 0.08032141506671905
# Epoch 63: Accuracy 0.871, Average Loss 0.4903275936841965
# 2025-12-17 11:37:27.594866
#
# Epoch 64: Accuracy 0.970, Average Loss 0.07745135925710202
# Epoch 64: Accuracy 0.871, Average Loss 0.49148730635643006
# 2025-12-17 11:38:20.920796
#
# Epoch 65: Accuracy 0.971, Average Loss 0.07711224615573883
# Epoch 65: Accuracy 0.875, Average Loss 0.4871603399515152
# 2025-12-17 11:39:14.342639
#
# Epoch 66: Accuracy 0.971, Average Loss 0.07899293042719364
# Epoch 66: Accuracy 0.878, Average Loss 0.5116829186677933
# 2025-12-17 11:40:07.763653
#
# Epoch 67: Accuracy 0.971, Average Loss 0.07562357731163502
# Epoch 67: Accuracy 0.876, Average Loss 0.49393502473831175
# 2025-12-17 11:41:00.781998
#
# Epoch 68: Accuracy 0.970, Average Loss 0.07700040303170681
# Epoch 68: Accuracy 0.870, Average Loss 0.5282128244638443
# 2025-12-17 11:41:53.691361
#
# Epoch 69: Accuracy 0.975, Average Loss 0.06640409380197525
# Epoch 69: Accuracy 0.872, Average Loss 0.5156196594238281
# 2025-12-17 11:42:46.838379
#
# Epoch 70: Accuracy 0.976, Average Loss 0.06371830254793168
# Epoch 70: Accuracy 0.877, Average Loss 0.5050972461700439
# 2025-12-17 11:43:40.372284
#
# Epoch 71: Accuracy 0.970, Average Loss 0.08048991471529007
# Epoch 71: Accuracy 0.869, Average Loss 0.5439565241336822
# 2025-12-17 11:44:33.815682
#
# Epoch 72: Accuracy 0.974, Average Loss 0.06806726828217506
# Epoch 72: Accuracy 0.875, Average Loss 0.5133730471134186
# 2025-12-17 11:45:27.182016
#
# Epoch 73: Accuracy 0.972, Average Loss 0.07404525876045227
# Epoch 73: Accuracy 0.869, Average Loss 0.537673482298851
# 2025-12-17 11:46:20.548940
#
# Epoch 74: Accuracy 0.974, Average Loss 0.0686636296659708
# Epoch 74: Accuracy 0.872, Average Loss 0.5413649380207062
# 2025-12-17 11:47:14.331541
#
# Epoch 75: Accuracy 0.968, Average Loss 0.08574860773980618
# Epoch 75: Accuracy 0.866, Average Loss 0.5505047559738159
# 2025-12-17 11:48:07.665020
#
# Epoch 76: Accuracy 0.971, Average Loss 0.07691219992935658
# Epoch 76: Accuracy 0.871, Average Loss 0.5231620579957962
# 2025-12-17 11:49:01.341687
#
# Epoch 77: Accuracy 0.979, Average Loss 0.056685169860720634
# Epoch 77: Accuracy 0.878, Average Loss 0.523070615530014
# 2025-12-17 11:49:54.639582
#
# Epoch 78: Accuracy 0.978, Average Loss 0.05792125709354878
# Epoch 78: Accuracy 0.876, Average Loss 0.5241350173950196
# 2025-12-17 11:50:47.763432
#
# Epoch 79: Accuracy 0.982, Average Loss 0.04919388066977262
# Epoch 79: Accuracy 0.883, Average Loss 0.5373786062002182
# 2025-12-17 11:51:41.173441
#
# Epoch 80: Accuracy 0.979, Average Loss 0.05900942452251911
# Epoch 80: Accuracy 0.875, Average Loss 0.5581758260726929
# 2025-12-17 11:52:34.506130
#
# Epoch 81: Accuracy 0.981, Average Loss 0.051392066143453125
# Epoch 81: Accuracy 0.877, Average Loss 0.5572047501802444
# 2025-12-17 11:53:27.540004
#
# Epoch 82: Accuracy 0.984, Average Loss 0.042202875092625616
# Epoch 82: Accuracy 0.878, Average Loss 0.5305818527936935
# 2025-12-17 11:54:20.683456
#
# Epoch 83: Accuracy 0.980, Average Loss 0.05373537693172693
# Epoch 83: Accuracy 0.873, Average Loss 0.5764262020587921
# 2025-12-17 11:55:14.077477
#
# Epoch 84: Accuracy 0.984, Average Loss 0.042791818454861644
# Epoch 84: Accuracy 0.882, Average Loss 0.5354415953159333
# 2025-12-17 11:56:07.267599
#
# Epoch 85: Accuracy 0.982, Average Loss 0.04640121091157198
# Epoch 85: Accuracy 0.881, Average Loss 0.5431466609239578
# 2025-12-17 11:57:00.802656
#
# Epoch 86: Accuracy 0.985, Average Loss 0.03974864199757576
# Epoch 86: Accuracy 0.879, Average Loss 0.5307063698768616
# 2025-12-17 11:57:54.229068
#
# Epoch 87: Accuracy 0.985, Average Loss 0.040793640576303004
# Epoch 87: Accuracy 0.878, Average Loss 0.5603337317705155
# 2025-12-17 11:58:47.775768
#
# Epoch 88: Accuracy 0.986, Average Loss 0.03762641873210668
# Epoch 88: Accuracy 0.883, Average Loss 0.5402174890041351
# 2025-12-17 11:59:41.176640
#
# Epoch 89: Accuracy 0.987, Average Loss 0.036084131821990016
# Epoch 89: Accuracy 0.883, Average Loss 0.5511980265378952
# 2025-12-17 12:00:34.346553
#
# Epoch 90: Accuracy 0.986, Average Loss 0.03745948359370232
# Epoch 90: Accuracy 0.884, Average Loss 0.5509742081165314
# 2025-12-17 12:01:27.332858
#
# Epoch 91: Accuracy 0.984, Average Loss 0.0452014135196805
# Epoch 91: Accuracy 0.882, Average Loss 0.5684744834899902
# 2025-12-17 12:02:20.316541
#
# Epoch 92: Accuracy 0.985, Average Loss 0.041361701153218744
# Epoch 92: Accuracy 0.883, Average Loss 0.5559793978929519
# 2025-12-17 12:03:13.342781
#
# Epoch 93: Accuracy 0.988, Average Loss 0.0320158464461565
# Epoch 93: Accuracy 0.882, Average Loss 0.5313341796398163
# 2025-12-17 12:04:06.315664
#
# Epoch 94: Accuracy 0.987, Average Loss 0.033663969039916995
# Epoch 94: Accuracy 0.883, Average Loss 0.5574939757585525
# 2025-12-17 12:04:59.282450
#
# Epoch 95: Accuracy 0.987, Average Loss 0.03368118664249778
# Epoch 95: Accuracy 0.887, Average Loss 0.549571442604065
# 2025-12-17 12:05:52.549688
#
# Epoch 96: Accuracy 0.984, Average Loss 0.042790349759161475
# Epoch 96: Accuracy 0.879, Average Loss 0.5814576387405396
# 2025-12-17 12:06:45.848776
#
# Epoch 97: Accuracy 0.986, Average Loss 0.03859481628984213
# Epoch 97: Accuracy 0.881, Average Loss 0.5682190507650375
# 2025-12-17 12:07:38.860731
#
# Epoch 98: Accuracy 0.986, Average Loss 0.04047925181686878
# Epoch 98: Accuracy 0.884, Average Loss 0.569600087404251
# 2025-12-17 12:08:31.852451
#
# Epoch 99: Accuracy 0.987, Average Loss 0.03499506263062358
# Epoch 99: Accuracy 0.887, Average Loss 0.5387698411941528
# 2025-12-17 12:09:24.789652
#
# Epoch 100: Accuracy 0.987, Average Loss 0.03684636894613504
# Epoch 100: Accuracy 0.885, Average Loss 0.5612133920192719
# 2025-12-17 12:10:17.774149
#
# Accuracy 0.885, Average Loss 0.5612133920192719

