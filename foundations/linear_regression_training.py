import numpy as np
from numpy.typing import NDArray


class Solution:
   # Only for this problem. Derivates handled using PyTorch 
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights = initial_weights.astype(np.float64).copy()
        N = X.shape[0]
        for _ in range(num_iterations):
            # Forward pass: prediction for all samples
            preds = self.get_model_prediction(X, weights)
            # for each wt in index j, compute gradient
            for j in range(len(weights)):
                grad = self.get_derivative(preds, Y, N, X, j)
                # Gradient descent Update
                weights[j] -= self.learning_rate * grad

        return np.round(weights, 5)
