import mlx.core as mx
from softgrad.layer import RecursiveLayer
from softgrad.function import Function


class Parallel(RecursiveLayer):
    def __init__(self, layers, aggregation_fn):
        super().__init__()
        self.layers = layers
        self.aggregation_fn = aggregation_fn

    def get_trainable_layers(self):
        """Return all trainable layers from all branches."""
        trainable = []
        for layer in self.layers:
            trainable.extend(layer.get_trainable_layers())
        return trainable

    def _link(self):
        # Link all sublayers with the same input shape
        for layer in self.layers:
            layer.link(self.input_shape)

        # Verify all outputs have the same shape (required for aggregation)
        output_shapes = [layer.output_shape for layer in self.layers]
        if not all(shape == output_shapes[0] for shape in output_shapes):
            raise ValueError(
                f"All parallel layers must have the same output shape for aggregation. "
                f"Got shapes: {output_shapes}"
            )

        dummy_outputs = [
            mx.zeros((1,) + shape)  # (batch=1, *output_shape)
            for shape in output_shapes
        ]
        dummy_result = self.aggregation_fn.apply(*dummy_outputs)
        self.output_shape = dummy_result.shape[1:]

    def _forward(self, x_in: mx.array) -> mx.array:
        # Execute all layers in parallel
        outputs = [layer.forward(x_in) for layer in self.layers]

        # Aggregate results
        result = self.aggregation_fn.apply(*outputs)

        # Save outputs for backward
        self.ctx['outputs'] = outputs

        return result

    def _backward(self, dx_out: mx.array) -> mx.array:
        outputs = self.ctx['outputs']

        # Compute gradient through aggregation
        # This gives us the gradient w.r.t. each branch output
        d_outputs = self.aggregation_fn.derivative(dx_out, *outputs)

        # Backprop through each layer
        dx_ins = []
        for i, layer in enumerate(self.layers):
            dx_in = layer.backward(d_outputs[i])
            dx_ins.append(dx_in)

        # Sum gradients from all branches (they all came from the same input)
        dx_in_total = dx_ins[0]
        for dx_in in dx_ins[1:]:
            dx_in_total = dx_in_total + dx_in

        return dx_in_total
