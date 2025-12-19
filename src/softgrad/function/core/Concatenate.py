from softgrad.function.Function import Function
import mlx.core as mx

class Concatenate(Function):
    @staticmethod
    def apply(*inputs):
        return mx.concatenate(inputs, axis=-1)

    @staticmethod
    def derivative(dx_out, *inputs):
        # Split gradient back along concatenation dimension (last axis)
        splits = [inp.shape[-1] for inp in inputs]
        gradients = []
        start = 0
        for size in splits:
            gradients.append(dx_out[..., start:start + size])
            start += size
        return gradients


concatenate = Concatenate()