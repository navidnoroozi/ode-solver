import math
import numpy as np
import matplotlib.pyplot as plt
import sys
from solver_method import euler_solver

def my_function(x):
    return np.array([-x[0] , -5*x[1]*x[0]])

# === Create solver instance and solve ===
Ts = 0.01
solver = euler_solver.EulerSolver(func=my_function,step_size=0.01,max_iter=1000)
out=solver.solve(np.array([1,1]))
print(f"The final value is {out[0]} is reached at iteration {out[1]}")
#print(vars(solver)['func'])
print(f"Dimension of {np.shape(out[2])}")

# === Plot the result for the first component vs. time in seconds ===

time = np.arange(out[1]+1)*Ts  # shape: (100,)
plt.figure(figsize=(10, 5))
plt.plot(time, out[2][:, 0], label='Dimension 1')
plt.xlabel("time(s)")
plt.ylabel("x_1")
plt.grid(True)
plt.show()