import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# GradientDescent1D class
# -------------------------
class GradientDescent1D:
    """
    Simple gradient descent for single-variable functions.
    Stores history and plots progress.

    Attributes:
      function: callable f(x)
      derivative: callable df(x)
      current_x: float, current x value
      learning_rate: float, step size (alpha)
      max_iterations: int
      verbose: bool, whether to print each iteration
      plot_interval: int, plot every N iterations
    """

    def __init__(self, function, derivative, starting_x,
                 learning_rate=0.01, max_iterations=100,
                 verbose=True, plot_interval=10):
        # Store the function and its derivative
        self.function = function
        self.derivative = derivative

        # Initialize values from user inputs
        self.current_x = float(starting_x)
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.verbose = bool(verbose)
        self.plot_interval = int(plot_interval)

        # Keep history of visited x values and their costs
        self.x_history = [self.current_x]
        self.cost_history = [self.function(self.current_x)]

    def step(self):
        """
        Do one gradient descent update:
          grad = df(current_x)
          current_x = current_x - learning_rate * grad
        Save and return gradient and new cost.
        """
        grad = self.derivative(self.current_x)            # gradient at current_x
        self.current_x = self.current_x - self.learning_rate * grad  # move downhill
        cost = self.function(self.current_x)              # compute new cost

        # Save history for plotting and analysis
        self.x_history.append(self.current_x)
        self.cost_history.append(cost)

        return grad, cost

    def fit(self):
        """
        Run the gradient descent loop for max_iterations.
        Print progress if verbose. Plot progress every plot_interval iterations.
        """
        for iteration in range(1, self.max_iterations + 1):
            grad, cost = self.step()

            # Print details each iteration if verbose is True
            if self.verbose:
                print(f"Iter {iteration:3d} | x = {self.current_x:.8f} | "
                      f"cost = {cost:.8f} | grad = {grad:.8f}")

            # Plot progress every plot_interval iterations and at the final iteration
            if (iteration % self.plot_interval == 0) or (iteration == self.max_iterations):
                self._plot_progress(iteration)

            # Optional early stop if gradient is extremely small (flat slope)
            if abs(grad) < 1e-12:
                if self.verbose:
                    print(f"Converged (tiny gradient) at iteration {iteration}.")
                break

        return self.current_x, self.cost_history

    def _plot_progress(self, iteration):
        """
        Plot the cost function curve and the points visited so far.
        The plotting window is chosen around visited x values.
        """
        xs = np.array(self.x_history)
        ys = np.array(self.cost_history)

        # Define a plotting range centered on visited points
        x_min, x_max = xs.min(), xs.max()
        center = 0.5 * (x_min + x_max)
        span = max(1.0, (x_max - x_min) * 1.5)  # ensure a minimum width so plot isn't tiny
        plot_x = np.linspace(center - span, center + span, 400)
        plot_y = np.vectorize(self.function)(plot_x)

        plt.figure(figsize=(8, 5))
        plt.plot(plot_x, plot_y, label="Cost function f(x)")
        plt.plot(xs, ys, '-o', label="Visited points", markersize=6, linewidth=1.2)
        plt.scatter([xs[-1]], [ys[-1]], color="black", s=80,
                    label=f"Current x (iter {iteration})")

        plt.title(f"Gradient Descent Progress (iter {iteration})\n"
                  f"learning_rate={self.learning_rate}, start={self.x_history[0]:.4f}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()


# -------------------------
# Helper input functions
# -------------------------
def ask_float(prompt, default):
    """Ask the user for a float; use default if they press Enter."""
    while True:
        user_input = input(f"{prompt} [default: {default}]: ").strip()
        if user_input == "":
            return float(default)
        try:
            return float(user_input)
        except ValueError:
            print("Please enter a valid number or press Enter for default.")


def ask_int(prompt, default):
    """Ask the user for an int; use default if they press Enter."""
    while True:
        user_input = input(f"{prompt} [default: {default}]: ").strip()
        if user_input == "":
            return int(default)
        try:
            return int(user_input)
        except ValueError:
            print("Please enter a valid integer or press Enter for default.")


# -------------------------
# Preset functions for menu
# -------------------------
def make_quadratic(a, b, c):
    """
    Return a quadratic function f(x) = a*x^2 + b*x + c and its derivative df(x).
    This allows the user to create a custom quadratic easily.
    """
    def f(x):
        return a * x**2 + b * x + c
    def df(x):
        return 2 * a * x + b
    return f, df


def preset_cubic():
    """Example of a cubic function and derivative."""
    def f(x):
        return x**3 - 3 * x**2 + 2
    def df(x):
        return 3 * x**2 - 6 * x
    return f, df


def preset_square():
    """Simple f(x) = x^2 example."""
    def f(x): return x**2
    def df(x): return 2 * x
    return f, df


# -------------------------
# Interactive menu
# -------------------------
def interactive_menu():
    """
    Show a menu to the user. The user can:
      1) Run gradient descent on f(x) = x^2
      2) Run gradient descent on a preset cubic
      3) Create a custom quadratic (a*x^2 + b*x + c)
      4) Exit
    After each run, the menu reappears so the user can experiment again.
    """
    print("\n=== Gradient Descent Interactive Menu ===")
    while True:
        print("\nChoose an option:")
        print("1) Run default: f(x) = x^2")
        print("2) Run preset cubic: f(x) = x^3 - 3x^2 + 2")
        print("3) Create custom quadratic: f(x) = a*x^2 + b*x + c")
        print("4) Exit")

        choice = input("Enter your choice (1-4): ").strip()
        if choice == "4":
            print("Goodbye! 👋")
            break

        if choice == "1":
            f, df = preset_square()
            desc = "f(x) = x^2"
        elif choice == "2":
            f, df = preset_cubic()
            desc = "f(x) = x^3 - 3x^2 + 2"
        elif choice == "3":
            print("Enter coefficients for quadratic f(x) = a*x^2 + b*x + c")
            a = ask_float("Enter a", 1.0)
            b = ask_float("Enter b", 0.0)
            c = ask_float("Enter c", 0.0)
            f, df = make_quadratic(a, b, c)
            desc = f"f(x) = {a}*x^2 + {b}*x + {c}"
        else:
            print("Invalid choice — please enter 1, 2, 3 or 4.")
            continue

        # Ask user for GD settings, with defaults
        print(f"\nSelected: {desc}")
        start_x = ask_float("Enter starting point (x0)", 5.0)
        learning_rate = ask_float("Enter learning rate (alpha)", 0.01)
        num_iters = ask_int("Enter number of iterations", 100)
        verbose_choice = input("Show iteration details? (y/n) [default: y]: ").strip().lower()
        verbose_flag = (verbose_choice != "n")
        plot_interval = ask_int("Plot every N iterations", 10)

        # Create the gradient descent object and run
        gd = GradientDescent1D(function=f, derivative=df,
                               starting_x=start_x,
                               learning_rate=learning_rate,
                               max_iterations=num_iters,
                               verbose=verbose_flag,
                               plot_interval=plot_interval)

        final_x, costs = gd.fit()
        print("\n=== Run finished ===")
        print(f"Final x ≈ {final_x:.8f}")
        print(f"Final cost ≈ {costs[-1]:.8f}")
        print("Returning to menu...")


# -------------------------
# Run the interactive menu if this file is executed
# -------------------------
if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")

