from softgrad import Network
from softgrad.layer.core import Linear
from softgrad.layer.core.Embedding import Embedding
from softgrad.layer.reshape import Flatten

network = Network(input_shape=(1,))  # Single word input
network.add_layer(Embedding(5000, 100))
network.add_layer(Flatten())
network.add_layer(Linear(3))