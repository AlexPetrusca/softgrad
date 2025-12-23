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

# Full Size Resnet Log
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

