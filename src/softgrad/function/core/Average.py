from softgrad.function.Function import Function

class Average(Function):
    @staticmethod
    def apply(*inputs):
        result = inputs[0]
        for inp in inputs[1:]:
            result = result + inp
        return result / len(inputs)

    @staticmethod
    def derivative(dx_out, *inputs):
        # Gradient of average: divided equally among inputs
        return [dx_out / len(inputs) for _ in inputs]


average = Average()