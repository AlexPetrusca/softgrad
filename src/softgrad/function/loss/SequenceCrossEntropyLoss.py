from softgrad.function.Function import Function
from softgrad.function.activation import softmax
import mlx.core as mx

class SequenceCrossEntropyLoss(Function):
    @staticmethod
    def apply(y_pred, y_true):
        B, T, C = y_pred.shape
        y_pred_flat = y_pred.reshape(B * T, C)
        y_true_flat = y_true.reshape(B * T)

        log_probs = y_pred_flat - mx.logsumexp(y_pred_flat, axis=-1, keepdims=True)
        losses = -mx.take_along_axis(log_probs, y_true_flat[:, None], axis=1).squeeze()
        return losses.reshape(B, T)

    @staticmethod
    def derivative(y_pred, y_true):
        B, T, C = y_pred.shape  # (B, T, C)
        y_pred_flat = y_pred.reshape(B * T, C)  # (BT, C)
        y_true_flat = y_true.reshape(B * T)  # (BT, C)

        probs = mx.softmax(y_pred_flat, axis=-1)

        targets_one_hot = mx.zeros((B * T, C))  # (BT, C)
        indices = mx.arange(B * T)
        targets_one_hot[indices, y_true_flat] = 1.0

        grad_flat = probs - targets_one_hot
        grad = grad_flat.reshape(B, T, C) / (B * T)  # (B, T, C)
        return grad


sequence_ce_loss = SequenceCrossEntropyLoss()