import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
 
# 1. Define system parameters
m = 1.0      # mass (kg)
k = 20.0     # spring constant (N/m)
c = 5.0      # damping coefficient (Ns/m)
 
# 2. Define state matrices
A = [[0, 1],[-k/m, -c/m]]
 
B = [[0],[1/m]]
 
C = [[1, 0]]     # Output = displacement
D = [[0]]
 
system = signal.StateSpace(A, B, C, D)
 
# 3. Define time vector
dt = 0.01
t = np.arange(0, 10, dt)
 
# 4. Define input (Unit Step)
u = np.ones_like(t)
 
 
# 5. Simulate using lsim equivalent
t_out, y_out, x_out = signal.lsim(system, U=u, T=t)
 
# 6. Plot result
plt.figure()
plt.plot(t_out, y_out)
plt.xlabel("Time (seconds)")
plt.ylabel("Displacement (m)")
plt.title("Step Response of Mass-Spring-Damper System")
plt.grid()
plt.show()
