import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x,w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))
        delta = (y_hat - y_true) * y_hat * (1.0 - y_hat)
        # Gradients
        dL_dw = delta * x          # element-wise for each weight
        dL_db = delta              # scalar

        # Round to 5 decimal places
        dL_dw = np.round(dL_dw, 5)
        dL_db = round(float(dL_db), 5)

        return dL_dw, dL_db
