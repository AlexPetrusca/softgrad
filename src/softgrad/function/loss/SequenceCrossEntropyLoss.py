from softgrad.function.Function import Function
from softgrad.function.activation import softmax
import mlx.core as mx

class SequenceCrossEntropyLoss(Function):
    """Cross entropy loss for sequence prediction (B, T, vocab_size)"""

    @staticmethod
    def apply(y_pred, y_true):
        """
        y_pred: (B, T, vocab_size) logits
        y_true: (B, T) token indices
        Returns: (B, T) losses
        """
        B, T, C = y_pred.shape
        y_pred_flat = y_pred.reshape(B * T, C)
        y_true_flat = y_true.reshape(B * T)

        # Compute log softmax
        log_probs = y_pred_flat - mx.logsumexp(y_pred_flat, axis=-1, keepdims=True)

        # Gather log probs for true tokens
        losses = -mx.take_along_axis(log_probs, y_true_flat[:, None], axis=1).squeeze()

        return losses.reshape(B, T)

    @staticmethod
    def derivative(y_pred, y_true):
        """
        Gradient of cross entropy w.r.t. logits
        Returns: (B, T, vocab_size)
        """
        B, T, C = y_pred.shape
        y_pred_flat = y_pred.reshape(B * T, C)
        y_true_flat = y_true.reshape(B * T)

        # Softmax probabilities
        probs = mx.softmax(y_pred_flat, axis=-1)

        # Create one-hot encoding of targets
        targets_one_hot = mx.zeros((B * T, C))
        indices = mx.arange(B * T)
        targets_one_hot[indices, y_true_flat] = 1.0

        # Gradient: probs - targets
        grad_flat = probs - targets_one_hot

        # Average over all tokens
        grad = grad_flat.reshape(B, T, C) / (B * T)

        return grad


sequence_ce_loss = SequenceCrossEntropyLoss()