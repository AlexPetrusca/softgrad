import mlx.core as mx
from softgrad.layer import RecursiveLayer
from softgrad.function import Function


class Parallel(RecursiveLayer):
    def __init__(self, layers, aggregation_fn):
        super().__init__()
        self.layers = layers
        self.aggregation_fn = aggregation_fn

    def get_trainable_layers(self):
        trainable = []
        for layer in self.layers:
            trainable.extend(layer.get_trainable_layers())
        return trainable

    def _link(self):
        for layer in self.layers:
            layer.link(self.input_shape)

        output_shapes = [layer.output_shape for layer in self.layers]
        if not all(shape == output_shapes[0] for shape in output_shapes):
            raise ValueError(
                f"All parallel layers must have the same output shape for aggregation, but got shapes: {output_shapes}"
            )

        dummy_outputs = [mx.zeros((1,) + shape) for shape in output_shapes]
        dummy_result = self.aggregation_fn.apply(*dummy_outputs)
        self.output_shape = dummy_result.shape[1:]

    def _forward(self, x_in: mx.array) -> mx.array:
        outputs = [layer.forward(x_in) for layer in self.layers]
        result = self.aggregation_fn.apply(*outputs)

        self.ctx['outputs'] = outputs

        return result

    def _backward(self, dx_out: mx.array) -> mx.array:
        outputs = self.ctx['outputs']

        # backprop through aggregation function
        d_outputs = self.aggregation_fn.derivative(dx_out, *outputs)

        # backprop through each layer
        dx_ins = []
        for i, layer in enumerate(self.layers):
            dx_in = layer.backward(d_outputs[i])
            dx_ins.append(dx_in)

        # sum gradients from all branches
        dx_in_total = dx_ins[0]
        for dx_in in dx_ins[1:]:
            dx_in_total = dx_in_total + dx_in

        return dx_in_total
