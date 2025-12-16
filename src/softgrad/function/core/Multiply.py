from softgrad.function.Function import Function

class Multiply(Function):
    @staticmethod
    def apply(*inputs):
        result = inputs[0]
        for inp in inputs[1:]:
            result = result * inp
        return result

    @staticmethod
    def derivative(dx_out, *inputs):
        # Gradient of multiplication: product of all other inputs
        n = len(inputs)
        gradients = []
        for i in range(n):
            grad = dx_out
            for j in range(n):
                if i != j:
                    grad = grad * inputs[j]
            gradients.append(grad)
        return gradients