import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ══════════════════════════════════════════════════════════════════
# SECTION A: System Parameters and State-Space Matrices
# ══════════════════════════════════════════════════════════════════

# ── System Parameters ─────────────────────────────────────────────
m = 1.0       # Mass (kg)
k = 4.0       # Spring constant (N/m)
c = 1.0       # Damping coefficient (N*s/m)

# ── Derived Parameters ────────────────────────────────────────────
omega_n = np.sqrt(k / m)          # Natural angular frequency (rad/s)
zeta    = c / (2 * m * omega_n)   # Damping ratio

print("=" * 55)
print("        MASS-SPRING-DAMPER SYSTEM SIMULATION")
print("=" * 55)
print(f"  Mass (m)                : {m} kg")
print(f"  Spring Constant (k)     : {k} N/m")
print(f"  Damping Coefficient (c) : {c} N·s/m")
print(f"  Natural Frequency (wn)  : {omega_n:.4f} rad/s")
print(f"  Damping Ratio (zeta)    : {zeta:.4f}")

if   zeta < 1:  print(f"  System Type             : UNDERDAMPED")
elif zeta == 1: print(f"  System Type             : CRITICALLY DAMPED")
else:           print(f"  System Type             : OVERDAMPED")
print("=" * 55)

# ── State-Space Matrices ──────────────────────────────────────────
#
#   State vector: x = [x1, x2] = [displacement, velocity]
#
#   A = [  0      1  ]     B = [  0  ]
#       [ -k/m  -c/m ]         [ 1/m ]
#
#   C = [ 1  0 ]             D = [ 0 ]
#
A = np.array([[0,      1    ],
              [-k/m,  -c/m  ]])

B = np.array([0, 1/m])      # Flattened for use in ODE
C = np.array([[0, 1]])
D = np.array([[0]])

print("\nSystem Matrix A:")
print(A)
print("Input Matrix B:")
print(B)


# ══════════════════════════════════════════════════════════════════
# SECTION B: ODE Definition and Simulation
# ══════════════════════════════════════════════════════════════════

# ── Time Setup ────────────────────────────────────────────────────
t_start = 0.0
t_end   = 10.0
dt      = 0.01
t_span  = (t_start, t_end)
t_eval  = np.arange(t_start, t_end + dt, dt)

def external_force(t):
    """Unit step input: F(t) = 1 N for all t >= 0"""
    return 1.0

def state_equation(t, x):
    """
    State-space ODE:  dx/dt = A*x + B*u
    x[0] = x1 = displacement
    x[1] = x2 = velocity
    """
    return A @ x + B * external_force(t)

# ── Initial Conditions ────────────────────────────────────────────
x0 = [0.0, 0.0]    # [initial displacement (m), initial velocity (m/s)]

# ── Solve ODE (RK45) ──────────────────────────────────────────────
solution = solve_ivp(
    fun    = state_equation,
    t_span = t_span,
    y0     = x0,
    t_eval = t_eval,
    method = 'RK45',
    rtol   = 1e-8,
    atol   = 1e-10
)

t  = solution.t
x1 = solution.y[0]   # Displacement x(t)
x2 = solution.y[1]   # Velocity dx/dt

steady_state = 1 / k  # Theoretical: x_ss = F/k = 1/k
print(f"\nTheoretical Steady-State Displacement : {steady_state:.4f} m")
print(f"Simulated  Final Displacement         : {x1[-1]:.4f} m")


# ══════════════════════════════════════════════════════════════════
# SECTION C: Plot Base System Response
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle(f'Mass-Spring-Damper System Response\n'
             f'(m={m} kg, k={k} N/m, c={c} N·s/m, ζ={zeta:.3f})',
             fontsize=13, fontweight='bold')

# Displacement
axes[0].plot(t, x1, color='steelblue', linewidth=2,
             label='Displacement x(t)')
axes[0].axhline(y=steady_state, color='red', linestyle='--',
                linewidth=1.4,
                label=f'Steady-State = {steady_state:.4f} m')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Displacement (m)')
axes[0].set_title('Displacement Response')
axes[0].legend()
axes[0].grid(True, alpha=0.4)

# Velocity
axes[1].plot(t, x2, color='darkorange', linewidth=2,
             label='Velocity ẋ(t)')
axes[1].axhline(y=0, color='gray', linestyle='--',
                linewidth=1, label='Steady-State = 0 m/s')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].set_title('Velocity Response')
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('msd_response.png', dpi=150)
plt.show()
print("\n[Saved] msd_response.png")


# ══════════════════════════════════════════════════════════════════
# SECTION D: Parameter Analysis — Effect of Damping Coefficient
# ══════════════════════════════════════════════════════════════════

c_critical = 2 * m * omega_n    # Critical damping coefficient

damping_cases = {
    f'Underdamped   (c=0.5,  ζ={0.5/(2*m*omega_n):.2f})':     0.5,
    f'Underdamped   (c=1.0,  ζ={1.0/(2*m*omega_n):.2f})':     1.0,
    f'Critically Damped (c={c_critical:.2f}, ζ=1.00)':        c_critical,
    f'Overdamped   (c=8.0,  ζ={8.0/(2*m*omega_n):.2f})':      8.0,
}

plt.figure(figsize=(10, 5))

for label, c_val in damping_cases.items():
    A_temp = np.array([[0, 1], [-k/m, -c_val/m]])
    B_temp = np.array([0, 1/m])

    def ode_damping(t, x, A=A_temp, B=B_temp):
        return A @ x + B * external_force(t)

    sol = solve_ivp(ode_damping, t_span, x0,
                    t_eval=t_eval, method='RK45')
    plt.plot(sol.t, sol.y[0], linewidth=2, label=label)

plt.axhline(y=steady_state, color='gray', linestyle=':',
            linewidth=1.4, label=f'Steady-State = {steady_state:.4f} m')
plt.xlabel('Time (s)')
plt.ylabel('Displacement x(t) (m)')
plt.title('Effect of Damping Coefficient on System Response')
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('msd_damping_comparison.png', dpi=150)
plt.show()
print("[Saved] msd_damping_comparison.png")


# ══════════════════════════════════════════════════════════════════
# SECTION E: Parameter Analysis — Effect of Spring Constant
# ══════════════════════════════════════════════════════════════════

spring_cases = {
    'k = 1.0 N/m':   1.0,
    'k = 4.0 N/m':   4.0,
    'k = 9.0 N/m':   9.0,
    'k = 16.0 N/m':  16.0,
}

plt.figure(figsize=(10, 5))

for label, k_val in spring_cases.items():
    A_temp  = np.array([[0, 1], [-k_val/m, -c/m]])
    B_temp  = np.array([0, 1/m])
    wn_temp = np.sqrt(k_val / m)
    ss_temp = 1 / k_val

    def ode_spring(t, x, A=A_temp, B=B_temp):
        return A @ x + B * external_force(t)

    sol = solve_ivp(ode_spring, t_span, x0,
                    t_eval=t_eval, method='RK45')
    plt.plot(sol.t, sol.y[0], linewidth=2,
             label=f'{label}  →  ωn={wn_temp:.2f} rad/s,  x_ss={ss_temp:.3f} m')

plt.xlabel('Time (s)')
plt.ylabel('Displacement x(t) (m)')
plt.title('Effect of Spring Constant on System Response')
plt.legend(loc='upper right', fontsize=9)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('msd_spring_comparison.png', dpi=150)
plt.show()
print("[Saved] msd_spring_comparison.png")

print("\n[Done] All plots generated successfully.")
print("  msd_response.png")
print("  msd_damping_comparison.png")
print("  msd_spring_comparison.png")