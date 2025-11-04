import numpy as np
import matplotlib.pyplot as plt


class GradientDescent1D:
    """
    Simple, user-friendly Gradient Descent for single-variable functions.

    Attributes:
        function: callable f(x)
        derivative: callable f'(x)
        current_x: current x value during the descent
        learning_rate: step size (alpha)
        max_iterations: how many iterations to run
        verbose: whether to print iteration info
        plot_interval: plot progress every this many iterations
    """

    def __init__(
        self,
        function,
        derivative,
        starting_point,
        learning_rate=0.01,
        max_iterations=100,
        verbose=True,
        plot_interval=10,
    ):
        self.function = function
        self.derivative = derivative
        self.current_x = float(starting_point)
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.verbose = bool(verbose)
        self.plot_interval = int(plot_interval)

        # history lists
        self.x_history = [self.current_x]
        self.cost_history = [self.function(self.current_x)]

    def step(self):
        """Do a single gradient descent update and return gradient and cost."""
        grad = self.derivative(self.current_x)
        # update rule (move opposite to gradient)
        self.current_x = self.current_x - self.learning_rate * grad
        cost = self.function(self.current_x)

        # save history
        self.x_history.append(self.current_x)
        self.cost_history.append(cost)

        return grad, cost

    def fit(self):
        """Run gradient descent for max_iterations, printing and plotting progress."""
        for iteration in range(1, self.max_iterations + 1):
            grad, cost = self.step()

            if self.verbose:
                print(
                    f"Iter {iteration:3d} | x = {self.current_x:.8f} | "
                    f"cost = {cost:.8f} | grad = {grad:.8f}"
                )

            # plot progress every plot_interval iterations and at the end
            if (iteration % self.plot_interval == 0) or (iteration == self.max_iterations):
                self._plot_progress(iteration)

            # small convergence check
            if abs(grad) < 1e-12:
                if self.verbose:
                    print(f"Converged (tiny gradient) at iteration {iteration}.")
                break

        return self.current_x, self.cost_history

    def _plot_progress(self, iteration):
        """Plot the cost function and visited points so far."""
        xs = np.array(self.x_history)
        ys = np.array(self.cost_history)

        # pick plotting window around visited x values
        x_min, x_max = xs.min(), xs.max()
        center = 0.5 * (x_min + x_max)
        span = max(1.0, (x_max - x_min) * 1.5)  # ensure a minimum width
        plot_x = np.linspace(center - span, center + span, 400)
        plot_y = np.vectorize(self.function)(plot_x)

        plt.figure(figsize=(8, 5))
        plt.plot(plot_x, plot_y, label="cost function f(x)")
        plt.plot(xs, ys, '-o', label="visited points", markersize=6, linewidth=1.2)
        plt.scatter([xs[-1]], [ys[-1]], color="black", s=80, label=f"current x (iter {iteration})")

        plt.title(f"Gradient Descent Progress (iter {iteration})\nlearning_rate={self.learning_rate}, start={self.x_history[0]:.4f}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


def ask_float(prompt, default):
    """
    Prompt user to enter a float. If they press Enter, return default.
    Keeps asking until a valid float or empty input is provided.
    """
    while True:
        user_input = input(f"{prompt} [default: {default}]: ").strip()
        if user_input == "":
            return float(default)
        try:
            return float(user_input)
        except ValueError:
            print("Please enter a valid number or press Enter for default.")


def ask_int(prompt, default):
    """Same as ask_float but returns int."""
    while True:
        user_input = input(f"{prompt} [default: {default}]: ").strip()
        if user_input == "":
            return int(default)
        try:
            return int(user_input)
        except ValueError:
            print("Please enter a valid integer or press Enter for default.")


if __name__ == "__main__":   
    def f(x):
        return x ** 2

    def df(x):
        return 2 * x

    print("\n=== Gradient Descent Interactive ===")
    start_point = ask_float("Enter starting point (x0)", 5.0)

    # learning rate input (default 0.01)
    learning_rate = ask_float("Enter learning rate (alpha)", 0.01)

    # iterations input (default 100)
    num_iterations = ask_int("Enter number of iterations", 100)

    # optional: allow user to control verbosity and plot interval
    verbose_choice = input("Show iteration details? (y/n) [default: y]: ").strip().lower()
    verbose_flag = (verbose_choice != "n")

    plot_interval = ask_int("Plot every N iterations", 10)

    # Create new object and run the gradient descent
    gd = GradientDescent1D(
        function=f,
        derivative=df,
        starting_point=start_point,
        learning_rate=learning_rate,
        max_iterations=num_iterations,
        verbose=verbose_flag,
        plot_interval=plot_interval,
    )

    final_x, cost_history = gd.fit()

    print("\n=== Final Result ===")
    print(f"Final x ≈ {final_x:.8f}")
    print(f"Final cost ≈ {cost_history[-1]:.8f}")
