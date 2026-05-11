import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        # small epsilon to avoid log(0)
        eps = 1e-7
        # Clip predictions to be strictly inside (0,1)
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        # Apply the binary cross-entropy formula elementwise
        losses = -(
            y_true * np.log(y_pred_clipped) +
            (1.0 - y_true) * np.log(1.0 - y_pred_clipped)
        )

        # Average over all samples
        loss_mean = np.mean(losses)
        # round
        return round(float(loss_mean), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        # small epsilon to avoid log(0)
        eps = 1e-7
        # Clip predictions to be strictly inside (0,1)
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        # only true class c has y_true[i,c]
        per_sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        # average across samples
        loss_mean = np.mean(per_sample_losses)
        # round and return
        return round(float(loss_mean), 4)        