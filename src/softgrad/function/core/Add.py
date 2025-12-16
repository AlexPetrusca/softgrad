from softgrad.function.Function import Function

class Add(Function):
    @staticmethod
    def apply(*inputs):
        result = inputs[0]
        for inp in inputs[1:]:
            result = result + inp
        return result

    @staticmethod
    def derivative(dx_out, *inputs):
        # Gradient of addition: flows equally to all inputs
        return [dx_out for _ in inputs]