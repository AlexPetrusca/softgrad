from softgrad.layer import Layer


class RecursiveLayer(Layer):
    def __init__(self):
        super().__init__()
        self.children = []

    def get_trainable_layers(self):
        trainable_layers = []
        for child in self.children:
            trainable_layers.extend(child.get_trainable_layers())
        return trainable_layers

    def freeze(self):
        for child in self.children:
            child.freeze()

    def unfreeze(self):
        for child in self.children:
            child.unfreeze()