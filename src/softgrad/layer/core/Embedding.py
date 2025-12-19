import math
from mlx import core as mx
from softgrad.layer import TrainableLayer


class Embedding(TrainableLayer):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def _link(self):
        # same initialization that PyTorch uses
        scale = math.sqrt(1.0 / self.embedding_dim)
        shape = (self.num_embeddings, self.embedding_dim)
        self.params["embeddings"] = mx.random.uniform(-scale, scale, shape=shape)
        self.output_shape = (self.input_shape[0], self.embedding_dim)

    def _forward(self, x_in: mx.array) -> mx.array:
        self.ctx['indices'] = x_in
        return self.params["embeddings"][x_in]

    def _backward(self, dx_out: mx.array) -> mx.array:
        indices = self.ctx['indices']

        indices_flat = indices.reshape(-1)
        dx_out_flat = dx_out.reshape(-1, self.embedding_dim)

        # Use bincount-style accumulation for efficiency
        # Create one-hot encoding for each index
        for idx in range(self.num_embeddings):
            mask = (indices_flat == idx).astype(mx.float32)
            if mx.sum(mask) > 0:
                # Sum gradients for all occurrences of this index
                grad_for_idx = mx.sum(dx_out_flat * mask[:, mx.newaxis], axis=0)
                self.params["dembeddings"][idx] += grad_for_idx

        return mx.zeros(indices.shape, dtype=mx.float32)   # no gradient