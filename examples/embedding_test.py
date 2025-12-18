import mlx.core as mx
from softgrad import Network
from softgrad.function.loss import MSELoss
from softgrad.layer.core import Embedding, Linear
from softgrad.layer.norm import LayerNorm
from softgrad.layer.transform import Flatten
from softgrad.optim import SGD


def test_overfit_single_batch():
    """Layer should be able to perfectly fit a single batch"""
    network = Network(input_shape=(1,))
    network.add_layer(Embedding(10, 8))
    network.add_layer(Flatten())
    network.add_layer(LayerNorm())
    network.add_layer(Linear(2))

    # Single batch of 4 examples
    X = mx.array([[0], [1], [2], [3]])
    Y = mx.array([[1, 0], [0, 1], [1, 0], [0, 1]])

    optimizer = SGD(eta=0.01)
    optimizer.bind_network(network)
    optimizer.bind_loss_fn(MSELoss())

    # Train until perfect fit
    for i in range(500):
        optimizer.step(X, Y)

        predictions = network.forward(X)
        loss = mx.mean((predictions - Y) ** 2)
        print(f"Iter {i}, Loss: {mx.mean(loss)}")

    # Should achieve near-zero loss
    predictions = network.forward(X)
    final_loss = mx.mean((predictions - Y) ** 2)
    print(f"Final loss: {final_loss}")
    assert final_loss < 0.01, "Should overfit single batch"


test_overfit_single_batch()