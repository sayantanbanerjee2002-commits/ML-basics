import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional

class GradientDescent1D:
    """
    Simple  1D Gradient Descent tool

    Parameters
    ----------
    func : callable
        The cost/loss function f(x).
    dfunc : callable
        The derivative of the cost function df(x) means compute Graident
    start_point : float
        Starting x value chosen by the user.
    learning_rate : Optional[float]
        Step size (alpha). If None, defaults to 0.01.
    max_iterations : Optional[int]
        Number of iterations to run. If None, defaults to 100.
    verbose : bool
        If True, print progress each iteration.
    plot_interval : int
        Plot progress every `plot_interval` iterations (default 10).
    """

    def __init__(
        self,
        func: Callable[[float], float],
        dfunc: Callable[[float], float],
        start_point: float,
        learning_rate: Optional[float] = None,
        max_iterations: Optional[int] = None,
        verbose: bool = True,
        plot_interval: int = 10,
    ):
        # Save functions
        self.func = func
        self.dfunc = dfunc

        # defaults Value of the parameter
        if learning_rate is None:
            learning_rate = 0.01
        if max_iterations is None:
            max_iterations = 100

        # Validate inputs
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iterations <= 0 or not isinstance(max_iterations, int):
            raise ValueError("max_iterations must be a positive integer.")
        if plot_interval <= 0 or not isinstance(plot_interval, int):
            raise ValueError("plot_interval must be a positive integer.")

        self.start_point = float(start_point)
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.verbose = bool(verbose)
        self.plot_interval = int(plot_interval)

        # Tracking History of the journey:-
        self.x_history = [self.start_point]
        self.cost_history = [self.func(self.start_point)]

    def step(self):
        """One gradient descent step: compute gradient and update x."""
        current_x = self.x_history[-1]
        gradient = self.dfunc(current_x)
        next_x = current_x - self.learning_rate * gradient

        # store history
        self.x_history.append(next_x)
        self.cost_history.append(self.func(next_x))

        return next_x, gradient, self.func(next_x)

    def fit(self):
        """Run gradient descent and return final x and cost history."""
        for it in range(1, self.max_iterations + 1):
            x_val, grad, cost = self.step()

            # print progress
            if self.verbose:
                print(f"Iter {it:3d} | x = {x_val:.8f} | cost = {cost:.8f} | grad = {grad:.8f}")

            # Plot progress every plot_interval iterations and on final iteration
            if (it % self.plot_interval == 0) or (it == self.max_iterations):
                self._plot_progress(it)

            # early stopping if gradient very small
            if abs(grad) < 1e-12:
                if self.verbose:
                    print(f"Stopped early at iteration {it} (tiny gradient: {grad:.2e}).")
                break

        return self.x_history[-1], self.cost_history

    def _plot_progress(self, iteration: int):
        """Plot the cost function and points visited so far."""
        xs = np.array(self.x_history)
        # choose a plotting range around visited xs
        x_min, x_max = xs.min(), xs.max()
        span = max(1.0, (x_max - x_min) * 1.5)  # ensure visible width
        center = (x_min + x_max) / 2.0
        plot_x = np.linspace(center - span, center + span, 400)
        plot_y = np.vectorize(self.func)(plot_x)

        plt.figure(figsize=(8, 5))
        plt.plot(plot_x, plot_y, label='f(x) (cost)')
        visited_y = np.vectorize(self.func)(xs)
        plt.plot(xs, visited_y, '-o', label='Visited points', markersize=6)
        plt.scatter([xs[-1]], [visited_y[-1]], color='black', s=80, label=f'Current (iter {iteration})')

        plt.title(f'Gradient Descent Progress (iter {iteration})\n'
                  f'start={self.start_point}, lr={self.learning_rate}, steps={len(xs)-1}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

# Demostration:-
if __name__ == "__main__":
    def f(x): return x**2
    def df(x): return 2*x

    gd1 = GradientDescent1D(
        func=f,
        dfunc=df,
        start_point=5.0,
        learning_rate=0.1,     # user chooses learning rate
        max_iterations=50,     # user chooses iterations
        verbose=True,
        plot_interval=10
    )
    final_x1, costs1 = gd1.fit()
    print("Final (user-chosen):", final_x1, costs1[-1])

    gd2 = GradientDescent1D(
        func=f,
        dfunc=df,
        start_point=5.0,       # user chooses only start point
        learning_rate=None,    # will default to 0.01
        max_iterations=None,   # will default to 100
        verbose=False,         # quieter run
        plot_interval=20
    )
    final_x2, costs2 = gd2.fit()
    print("Final (with defaults):", final_x2, costs2[-1])
