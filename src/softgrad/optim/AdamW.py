import mlx.core as mx
from softgrad.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, eta=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        super().__init__()
        self.eta: float = eta  # Learning rate
        self.beta1: float = beta1  # Exponential decay rate for first moment
        self.beta2: float = beta2  # Exponential decay rate for second moment
        self.epsilon: float = epsilon  # Constant for numerical stability
        self.weight_decay: float = weight_decay  # Weight decay coefficient
        self.t: int = 0  # Timestep for bias correction

    def step(self, x: mx.array, y: mx.array):
        # feed forward
        y_pred = self.network.forward(x, save_ctx=True)

        # backpropagate
        grad = self.loss_fn.derivative(y_pred, y)
        for layer in reversed(self.network.layers):
            grad = layer.backward(grad)

        # increment timestep
        self.t += 1

        # update layers
        trainable_layers = self.network.get_trainable_layers()
        for i, layer in enumerate(trainable_layers):
            batch_size = layer.ctx.dx_out.shape[0]

            for param in layer.params:
                g = param.grad / batch_size
                m_key = f"m{param.name}_{i}"
                v_key = f"v{param.name}_{i}"

                # update biased moment
                m = self.beta1 * self.ctx.get(m_key, 0) + (1 - self.beta1) * g
                v = self.beta2 * self.ctx.get(v_key, 0) + (1 - self.beta2) * (g * g)
                self.ctx[m_key] = m
                self.ctx[v_key] = v

                # compute bias-corrected moment (account for zero-initialization)
                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                adaptive_update = m_hat / (mx.sqrt(v_hat) + self.epsilon)
                decoupled_weight_decay = self.weight_decay * param.value
                param.value = param.value - self.eta * (adaptive_update + decoupled_weight_decay)  # AdamW update

        # zero gradients
        self.network.zero_grad()

        # eval all optimizer state buffers to clean computation graphs; avoid memory leak
        mx.eval([v for v in self.ctx.values() if isinstance(v, mx.array)])

        return grad