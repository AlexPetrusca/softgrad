from softgrad.layer import Layer


class TrainableLayer(Layer):
    def __init__(self):
        super().__init__()
        self.params: Layer.Parameters = Layer.Parameters()
        self.trainable = True

    def get_trainable_layers(self):
        if self.trainable:
            return [self]
        return []

    def zero_grad(self):
        self.params.zero_grad()

    def freeze(self):
        self.trainable = False

    def unfreeze(self):
        self.trainable = True