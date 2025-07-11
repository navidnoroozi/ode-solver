import math
import numpy as np
import sys


class EulerSolver:
    def __init__(self, func, step_size=1e-2,max_iter=1000,tol=1e-6):
        """
        Initializes the Euler solver.

        Parameters:
        - func: Function f(.) describng the dynamics x_dot = f(x)
        - tol: Convergence tolerance.
        - max_iter: Maximum number of iterations.
        - h: Step size for numerical derivative.
        """
        self.func = func
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, initial_cond):
        """
        Solves the nonlinear equation using Newton's method.

        Parameters:
        - initial_cond: Starting value for the iteration.

        Returns:
        - The root if found within the allowed iterations.
        """
        x = initial_cond
        F = x
        try:
            for i in range(self.max_iter):
                f_val = self.func(x)
                f_new = x + self.step_size*f_val
                x = f_new
                F = np.vstack([F, f_new])
                if np.linalg.norm(f_new - f_val) < self.tol:
                  print(f"Tolerence in {i+1} iterations reached.")
                  break
            return f_new,i+1,F
        except ValueError:
              print("Dimension of funcion and or initial conditions does not match")
              sys.exit(1)

        # raise RuntimeError("Newton method did not converge within the maximum number of iterations.")
