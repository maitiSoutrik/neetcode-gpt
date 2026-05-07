class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        # algo

        # iterate over iterations
        # every iteration, do the following
        # calculate x and update for next step
        # return final x round to 5 decimal places
        x = init
        
        # Perform gradient descent
        for _ in range(iterations):
            gradient = 2 * x
            x = x - learning_rate * gradient

        # Round to 5 decimal places as required
        return round(x,5)
        
