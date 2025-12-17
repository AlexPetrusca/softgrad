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

train(train_data, epochs=100, batch_size=1000, test_data=test_data)





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








# Epoch 0: Accuracy 0.000, Average Loss 2.410071973800659
# Epoch 0: Accuracy 0.000, Average Loss 2.4217883825302122
# 2025-12-17 00:26:15.640918
#
# Epoch 1: Accuracy 0.126, Average Loss 1.6484799933433534
# Epoch 1: Accuracy 0.161, Average Loss 1.609223973751068
# 2025-12-17 00:27:14.780027
#
# Epoch 2: Accuracy 0.325, Average Loss 1.3321566796302795
# Epoch 2: Accuracy 0.343, Average Loss 1.3250127196311952
# 2025-12-17 00:28:13.699045
#
# Epoch 3: Accuracy 0.428, Average Loss 1.1648745560646057
# Epoch 3: Accuracy 0.433, Average Loss 1.1822108149528503
# 2025-12-17 00:29:12.580789
#
# Epoch 4: Accuracy 0.534, Average Loss 0.9454136312007904
# Epoch 4: Accuracy 0.533, Average Loss 0.9882292866706848
# 2025-12-17 00:30:11.543184
#
# Epoch 5: Accuracy 0.602, Average Loss 0.8632068324089051
# Epoch 5: Accuracy 0.590, Average Loss 0.9390227496623993
# 2025-12-17 00:31:14.328471
#
# Epoch 6: Accuracy 0.663, Average Loss 0.7269458770751953
# Epoch 6: Accuracy 0.652, Average Loss 0.7752862691879272
# 2025-12-17 00:33:35.402715
#
# Epoch 7: Accuracy 0.726, Average Loss 0.621644059419632
# Epoch 7: Accuracy 0.711, Average Loss 0.6796978652477265
# 2025-12-17 00:34:40.135261
#
# Epoch 8: Accuracy 0.737, Average Loss 0.602119414806366
# Epoch 8: Accuracy 0.721, Average Loss 0.6702426850795746
# 2025-12-17 00:35:43.367692
#
# Epoch 9: Accuracy 0.754, Average Loss 0.5606626462936402
# Epoch 9: Accuracy 0.725, Average Loss 0.6578136324882508
# 2025-12-17 00:36:46.552776
#
# Epoch 10: Accuracy 0.784, Average Loss 0.4906641572713852
# Epoch 10: Accuracy 0.762, Average Loss 0.5615911722183228
# 2025-12-17 00:38:08.378602
#
# Epoch 11: Accuracy 0.800, Average Loss 0.469146084189415
# Epoch 11: Accuracy 0.780, Average Loss 0.5357256472110749
# 2025-12-17 00:39:17.714419
#
# Epoch 12: Accuracy 0.819, Average Loss 0.4220418727397919
# Epoch 12: Accuracy 0.792, Average Loss 0.5317607820034027
# 2025-12-17 00:40:40.175483
#
# Epoch 13: Accuracy 0.834, Average Loss 0.39595011472702024
# Epoch 13: Accuracy 0.804, Average Loss 0.500785619020462
# 2025-12-17 00:42:03.218140
#
# Epoch 14: Accuracy 0.836, Average Loss 0.3858108627796173
# Epoch 14: Accuracy 0.803, Average Loss 0.4940700173377991
# 2025-12-17 00:43:21.855757
#
# Epoch 15: Accuracy 0.838, Average Loss 0.38728368282318115
# Epoch 15: Accuracy 0.802, Average Loss 0.504344055056572
# 2025-12-17 00:44:43.106850
#
# Epoch 16: Accuracy 0.848, Average Loss 0.3725249755382538
# Epoch 16: Accuracy 0.810, Average Loss 0.4984774738550186
# 2025-12-17 00:46:18.809612
#
# Epoch 17: Accuracy 0.858, Average Loss 0.343439484834671
# Epoch 17: Accuracy 0.816, Average Loss 0.48599493205547334
# 2025-12-17 00:47:26.971948
#
# Epoch 18: Accuracy 0.862, Average Loss 0.33169661700725556
# Epoch 18: Accuracy 0.811, Average Loss 0.4912333369255066
# 2025-12-17 00:49:01.617247
#
# Epoch 19: Accuracy 0.868, Average Loss 0.3191759449243545
# Epoch 19: Accuracy 0.821, Average Loss 0.47060473561286925
# 2025-12-17 00:50:42.232465
#
# Epoch 20: Accuracy 0.872, Average Loss 0.31057926774024963
# Epoch 20: Accuracy 0.824, Average Loss 0.4619213730096817
# 2025-12-17 00:51:56.941996
#
# Epoch 21: Accuracy 0.885, Average Loss 0.28437335669994357
# Epoch 21: Accuracy 0.831, Average Loss 0.45694270431995393
# 2025-12-17 00:53:03.201341
#
# Epoch 22: Accuracy 0.889, Average Loss 0.27530476987361907
# Epoch 22: Accuracy 0.839, Average Loss 0.44483602643013
# 2025-12-17 00:54:04.171196
#
# Epoch 23: Accuracy 0.895, Average Loss 0.2536059042811394
# Epoch 23: Accuracy 0.843, Average Loss 0.43469898104667665
# 2025-12-17 00:55:05.165508
#
# Epoch 24: Accuracy 0.888, Average Loss 0.28198642015457154
# Epoch 24: Accuracy 0.832, Average Loss 0.4831219643354416
# 2025-12-17 00:56:07.110101
#
# Epoch 25: Accuracy 0.906, Average Loss 0.2327580401301384
# Epoch 25: Accuracy 0.851, Average Loss 0.42378042340278627
# 2025-12-17 00:57:09.162729
#
# Epoch 26: Accuracy 0.907, Average Loss 0.2279633328318596
# Epoch 26: Accuracy 0.848, Average Loss 0.4256903797388077
# 2025-12-17 00:58:10.055920
#
# Epoch 27: Accuracy 0.905, Average Loss 0.23778258860111237
# Epoch 27: Accuracy 0.847, Average Loss 0.43446103036403655
# 2025-12-17 00:59:11.206881
#
# Epoch 28: Accuracy 0.906, Average Loss 0.2359294494986534
# Epoch 28: Accuracy 0.846, Average Loss 0.44498705863952637
# 2025-12-17 01:00:11.734971
#
# Epoch 29: Accuracy 0.916, Average Loss 0.20993057608604432
# Epoch 29: Accuracy 0.853, Average Loss 0.4316540062427521
# 2025-12-17 01:01:12.230662
#
# Epoch 30: Accuracy 0.917, Average Loss 0.20705257952213288
# Epoch 30: Accuracy 0.850, Average Loss 0.44044279158115385
# 2025-12-17 01:02:12.512280
#
# Epoch 31: Accuracy 0.922, Average Loss 0.19079286545515062
# Epoch 31: Accuracy 0.854, Average Loss 0.42641997039318086
# 2025-12-17 01:03:12.795397
#
# Epoch 32: Accuracy 0.920, Average Loss 0.19968207508325578
# Epoch 32: Accuracy 0.857, Average Loss 0.43070337176322937
# 2025-12-17 01:04:13.212693
#
# Epoch 33: Accuracy 0.922, Average Loss 0.19469139695167542
# Epoch 33: Accuracy 0.858, Average Loss 0.4179409921169281
# 2025-12-17 01:05:13.597669
#
# Epoch 34: Accuracy 0.933, Average Loss 0.1689568826556206
# Epoch 34: Accuracy 0.860, Average Loss 0.414584743976593
# 2025-12-17 01:06:13.880726
#
# Epoch 35: Accuracy 0.925, Average Loss 0.18490574836730958
# Epoch 35: Accuracy 0.856, Average Loss 0.4419887840747833
# 2025-12-17 01:07:14.281383
#
# Epoch 36: Accuracy 0.935, Average Loss 0.16385447710752488
# Epoch 36: Accuracy 0.862, Average Loss 0.4190028578042984
# 2025-12-17 01:08:14.604845
#
# Epoch 37: Accuracy 0.929, Average Loss 0.17648937970399856
# Epoch 37: Accuracy 0.858, Average Loss 0.42844298481941223
# 2025-12-17 01:09:15.090974
#
# Epoch 38: Accuracy 0.933, Average Loss 0.17063145726919174
# Epoch 38: Accuracy 0.858, Average Loss 0.4511851370334625
# 2025-12-17 01:10:15.469371
#
# Epoch 39: Accuracy 0.940, Average Loss 0.1527801175415516
# Epoch 39: Accuracy 0.868, Average Loss 0.4433560073375702
# 2025-12-17 01:11:15.798894
#
# Epoch 40: Accuracy 0.939, Average Loss 0.1525729078054428
# Epoch 40: Accuracy 0.860, Average Loss 0.4407694756984711
# 2025-12-17 01:12:16.127030
#
# Epoch 41: Accuracy 0.943, Average Loss 0.1472090108692646
# Epoch 41: Accuracy 0.865, Average Loss 0.4518663793802261
# 2025-12-17 01:13:16.564321
#
# Epoch 42: Accuracy 0.946, Average Loss 0.13465359479188918
# Epoch 42: Accuracy 0.873, Average Loss 0.41410922110080717
# 2025-12-17 01:14:17.053517
#
# Epoch 43: Accuracy 0.944, Average Loss 0.1435682789981365
# Epoch 43: Accuracy 0.871, Average Loss 0.44935176372528074
# 2025-12-17 01:15:17.433078
#
# Epoch 44: Accuracy 0.952, Average Loss 0.12263208284974098
# Epoch 44: Accuracy 0.876, Average Loss 0.42090030014514923
# 2025-12-17 01:16:17.877931
#
# Epoch 45: Accuracy 0.952, Average Loss 0.12341078087687492
# Epoch 45: Accuracy 0.866, Average Loss 0.44433772563934326
# 2025-12-17 01:17:18.216929
#
# Epoch 46: Accuracy 0.948, Average Loss 0.13215271294116973
# Epoch 46: Accuracy 0.862, Average Loss 0.46501135230064394
# 2025-12-17 01:18:18.718918
#
# Epoch 47: Accuracy 0.949, Average Loss 0.12979984998703004
# Epoch 47: Accuracy 0.868, Average Loss 0.45821002721786497
# 2025-12-17 01:19:19.631988
#
# Epoch 48: Accuracy 0.948, Average Loss 0.13398414239287376
# Epoch 48: Accuracy 0.864, Average Loss 0.47990223467350007
# 2025-12-17 01:20:20.627924