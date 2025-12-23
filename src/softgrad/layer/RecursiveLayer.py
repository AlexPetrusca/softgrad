from softgrad.layer import Layer


class RecursiveLayer(Layer):
    def __init__(self):
        super().__init__()
        self.layers = []

    def get_trainable_layers(self):
        trainable_layers = []
        for child in self.layers:
            trainable_layers.extend(child.get_trainable_layers())
        return trainable_layers

    def zero_grad(self):
        for child in self.layers:
            child.zero_grad()

    def freeze(self):
        for child in self.layers:
            child.freeze()

    def unfreeze(self):
        for child in self.layers:
            child.unfreeze()