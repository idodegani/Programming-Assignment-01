import numpy as np

def wolfe_condition_backtracking(f, grad, xk, pk, c1=0.01, alpha=1, rho=0.5):
    while f(xk + alpha * pk) > f(xk) + c1 * alpha * np.dot(grad(xk).T, pk):
        alpha *= rho
    return alpha

class LineSearchOptimizer:
    def __init__(self, f, grad, hess=None):
        self.f = f
        self.grad = grad
        self.hess = hess

    def gradient_descent(self, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
        x = x0
        for i in range(max_iter):
            gradient = self.grad(x)
            if np.linalg.norm(gradient) < param_tol:
                return x, self.f(x), True
            step_size = wolfe_condition_backtracking(self.f, self.grad, x, -gradient)
            x = x - step_size * gradient
            if abs(self.f(x) - self.f(x - step_size * gradient)) < obj_tol:
                return x, self.f(x), True
        return x, self.f(x), False

    def newton_method(self, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
        x = x0
        for i in range(max_iter):
            gradient = self.grad(x)
            hessian = self.hess(x)
            if np.linalg.norm(gradient) < param_tol:
                return x, self.f(x), True
            p_k = np.linalg.solve(hessian, -gradient)
            step_size = wolfe_condition_backtracking(self.f, self.grad, x, p_k)
            x = x + step_size * p_k
            if abs(self.f(x) - self.f(x + step_size * p_k)) < obj_tol:
                return x, self.f(x), True
        return x, self.f(x), False
