from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt

def plot_contours(f, x_bounds, y_bounds, title, path=None):
    # Generate meshgrid for contour plotting
    x = np.linspace(x_bounds[0], x_bounds[1], 400)
    y = np.linspace(y_bounds[0], y_bounds[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([X[i, j], Y[i, j]])) for j in range(X.shape[1])] for i in range(X.shape[0])])

    plt.figure(figsize=(8, 6))
    cp = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 50), cmap='jet')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'w.-')  # White line for the path
        plt.plot(path_x[-1], path_y[-1], 'ro')  # Red dot for the last point

    plt.show()


    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'w.-')  # White line for the path
        plt.plot(path_x[-1], path_y[-1], 'ro')  # Red dot for the last point

    plt.show()


    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'w.-')  # White line for the path
        plt.plot(path_x[-1], path_y[-1], 'ro')  # Red dot for the last point

    plt.show()


    if path is not None:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'w.-')  # White line for the path
        plt.plot(path_x[-1], path_y[-1], 'ro')  # Red dot for the last point

    plt.show()

def plot_function_values(method_data, title):
    """
    Plots the function values from different methods over iterations.

    :param method_data: Dictionary with method names as keys and lists of function values as values
    :param title: Title of the plot
    """
    plt.figure(figsize=(8, 6))
    for method, values in method_data.items():
        plt.plot(values, label=method)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Function value')
    plt.legend()
    plt.show()

""" # Example usage within the file (to be removed or commented out when deploying)
if __name__ == "__main__":
    # Define a simple quadratic function for demonstration purposes
    def demo_function(v):
        x, y = v[0], v[1]
        return (x - 1)**2 + (y - 2)**2

    # Example path data
    example_path = [(1, 2), (1.5, 2.5), (0.8, 1.8), (1.1, 2.1)]

    # Plotting function contours and a path
    plot_contours(demo_function, (0, 3), (0, 3), "Demo Function Contour", path=example_path)

    # Example function values for plotting
    example_values = {
        'Gradient Descent': [5, 4, 3, 2, 1.5, 1.1, 1.05, 1.01],
        'Newton Method': [5, 3, 2.1, 2.01, 2.001, 2.0001]
    }

    # Plotting function values
    plot_function_values(example_values, "Demo Function Value Decrease")
 """