import unittest
import numpy as np
from src.unconstrained_min import LineSearchOptimizer
from tests.examples import quadratic_function, rosenbrock_function, linear_function, exponential_function
from src.utils import plot_contours, plot_function_values

class TestLineSearchMinimization(unittest.TestCase):
    def setUp(self):
        self.initial_points = {
            'quadratic': np.array([1, 1]),
            'rosenbrock': np.array([-1, 2]),
            'linear': np.array([1, 1]),
            'exponential': np.array([1, 1])
        }
        self.max_iter = {
            'gradient_descent_rosenbrock': 10000,
            'default': 100
        }
        self.obj_tol = 1e-12
        self.param_tol = 1e-8

    def test_quadratic(self):
        x0 = self.initial_points['quadratic']
        optimizer = LineSearchOptimizer(lambda x: quadratic_function(x)[0], lambda x: quadratic_function(x)[1])
        result, fval, success = optimizer.gradient_descent(x0, self.obj_tol, self.param_tol, self.max_iter['default'])
        self.assertTrue(success)
        print("Quadratic: Final location:", result, "Final value:", fval, "Success:", success)

    def test_rosenbrock(self):
        x0 = self.initial_points['rosenbrock']
        optimizer = LineSearchOptimizer(lambda x: rosenbrock_function(x)[0], lambda x: rosenbrock_function(x)[1])
        result, fval, success = optimizer.gradient_descent(x0, self.obj_tol, self.param_tol, self.max_iter['gradient_descent_rosenbrock'])
        self.assertTrue(success)
        print("Rosenbrock: Final location:", result, "Final value:", fval, "Success:", success)

    def test_linear(self):
        x0 = self.initial_points['linear']
        optimizer = LineSearchOptimizer(lambda x: linear_function(x)[0], lambda x: linear_function(x)[1])
        result, fval, success = optimizer.gradient_descent(x0, self.obj_tol, self.param_tol, self.max_iter['default'])
        self.assertTrue(success)
        print("Linear: Final location:", result, "Final value:", fval, "Success:", success)

    def test_exponential(self):
        x0 = self.initial_points['exponential']
        optimizer = LineSearchOptimizer(lambda x: exponential_function(x)[0], lambda x: exponential_function(x)[1])
        result, fval, success = optimizer.gradient_descent(x0, self.obj_tol, self.param_tol, self.max_iter['default'])
        self.assertTrue(success)
        print("Exponential: Final location:", result, "Final value:", fval, "Success:", success)

    def test_all_examples(self):
        # Runs all test methods
        self.test_quadratic()
        self.test_rosenbrock()
        self.test_linear()
        self.test_exponential()

# To execute the tests when this script is run directly
if __name__ == "__main__":
    unittest.main()
