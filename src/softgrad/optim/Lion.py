# softgrad/optim/Lion.py

import mlx.core as mx
from softgrad.optim import Optimizer


class Lion(Optimizer):
    def __init__(self, eta=1e-4, beta1=0.9, beta2=0.99, weight_decay=0.01):
        super().__init__()
        self.eta: float = eta  # Learning rate
        self.beta1: float = beta1  # Momentum for update direction
        self.beta2: float = beta2  # Momentum for momentum buffer
        self.weight_decay: float = weight_decay  # Weight decay coefficient

    def step(self, x: mx.array, y: mx.array):
        # feed forward
        y_pred = self.network.forward(x, save_ctx=True)

        # backpropagate
        grad = self.loss_fn.derivative(y_pred, y)
        for layer in reversed(self.network.layers):
            grad = layer.backward(grad)

        # update layers
        trainable_layers = self.network.get_trainable_layers()
        for i, layer in enumerate(trainable_layers):
            batch_size = layer.ctx.dx_out.shape[0]

            for param in layer.params:
                g = param.grad / batch_size
                m_key = f"m{param.name}_{i}"

                # compute update direction
                m = self.ctx.get(m_key, 0)
                c = self.beta1 * m + (1 - self.beta1) * g

                decoupled_weight_decay = self.weight_decay * param.value
                param.value = param.value - self.eta * (mx.sign(c) + decoupled_weight_decay)  # Lion update

                # update momentum buffer with beta2
                self.ctx[m_key] = self.beta2 * m + (1 - self.beta2) * g

        # zero gradients
        self.network.zero_grad()

        # eval all momentum buffers to clean computation graphs; avoid memory leak
        mx.eval([v for v in self.ctx.values() if isinstance(v, mx.array)])

        return grad