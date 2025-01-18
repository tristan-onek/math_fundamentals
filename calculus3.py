# Calculus III Concepts in Python

# Imports:
import numpy as np
from sympy import symbols, diff, Matrix, sqrt, Eq, solve
import matplotlib.pyplot as plt
# End imports

# 1) Partial Derivatives:
x, y = symbols('x y')
f = x**2 * y + y**3

# Partial derivatives
partial_x = diff(f, x)
partial_y = diff(f, y)
print(f"∂f/∂x: {partial_x}")
print(f"∂f/∂y: {partial_y}")

# 2) Gradient computation:
z = symbols('z')
f = x**2 + y**2 + z**2
gradient = Matrix([diff(f, var) for var in (x, y, z)])
print(f"Gradient: {gradient}")

# 3) Double Integrals:
double_integral = integrate(integrate(x + y, (y, 0, x)), (x, 0, 1))
print(f"Double integral result: {double_integral}")

# 4) Triple Integrals:
triple_integral = integrate(integrate(integrate(x + y + z, (z, 0, 1)), (y, 0, 1)), (x, 0, 1))
print(f"Triple integral result: {triple_integral}")

# 5) Directional derivatives:
v = Matrix([3, 4])
v_unit = v / sqrt(v.dot(v))
grad_f = Matrix([diff(f, var) for var in (x, y)])
point = {x: 1, y: 1}
directional_derivative = grad_f.subs(point).dot(v_unit)
print(f"Directional derivative: {directional_derivative}")

# 6) Vector Fields:
X, Y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
U = Y
V = -X
plt.quiver(X, Y, U, V)
plt.title("Vector Field: F(x, y) = (y, -x)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# 7) Surface Integrals:
z = x**2 + y**2
dS = sqrt(1 + diff(z, x)**2 + diff(z, y)**2)
surface_integral = integrate(integrate(z * dS, (y, -1, 1)), (x, -1, 1))
print(f"Surface integral result: {surface_integral}")

# 8) Stokes' Theorem:
boundary_r = Matrix([cos(t), sin(t), 0])  # Circle parameterized
boundary_dr = boundary_r.diff(t)
boundary_F = F.subs({x: boundary_r[0], y: boundary_r[1], z: boundary_r[2]})
boundary_integral = integrate(boundary_F.dot(boundary_dr), (t, 0, 2 * pi))
surface_integral = integrate(integrate(curl.dot(Matrix([0, 0, 1])), (y, -1, 1)), (x, -1, 1))
print(f"Boundary integral: {boundary_integral}")
print(f"Surface integral: {surface_integral}")

# 9) Lagrange Multipliers:
g = x + y - 1
λ = symbols('lambda')
lagrange_eqs = [
    diff(f, x) - λ * diff(g, x),
    diff(f, y) - λ * diff(g, y),
    g
]
solution = solve(lagrange_eqs, (x, y, λ))
print(f"Lagrange multipliers solution: {solution}")
