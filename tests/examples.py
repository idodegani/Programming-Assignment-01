import numpy as np

def quadratic_function(x, need_hessian=False):
    Q = np.array([[1, 0], [0, 100]])  # Change Q based on the required example
    f = x.T @ Q @ x
    g = 2 * Q @ x
    h = 2 * Q if need_hessian else None
    return f, g, h

def rosenbrock_function(x, need_hessian=False):
    x1, x2 = x[0], x[1]
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    g = np.array([-400 * x1 * (x2 - x1**2) - 2 * (1 - x1), 200 * (x2 - x1**2)])
    if need_hessian:
        h = np.array([[1200 * x1**2 - 400 * x2 + 2, -400 * x1],
                      [-400 * x1, 200]])
    else:
        h = None
    return f, g, h

def linear_function(x, need_hessian=False):
    a = np.array([2, 3])  # Example vector a
    f = a.T @ x
    g = a
    h = np.zeros((2, 2)) if need_hessian else None
    return f, g, h

def exponential_function(x, need_hessian=False):
    x1, x2 = x[0], x[1]
    f = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)
    g = np.array([
        np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1),
        3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)
    ])
    if need_hessian:
        h = np.array([
            [np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1), 
             3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)],
            [3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1), 
             9 * np.exp(x1 + 3 * x2 - 0.1) + 9 * np.exp(x1 - 3 * x2 - 0.1)]
        ])
    else:
        h = None
    return f, g, h
