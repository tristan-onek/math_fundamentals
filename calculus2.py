# Imports
import numpy as np
from sympy import exp, series, oo, symbols, integrate, sin, diff, sqrt
# End imports

# Part 1) Sequences and Series:
# In the following example, approximate sum of 1/n^2...
n = np.arange(1, 10000)
series_sum = np.sum(1 / n**2)
print(f"Sum of series (approximated): {series_sum}")

# Part 2) Taylor Series:
taylor_exp = series(exp(x), x, 0, 5)  # 5 terms of the expansion
print(f"Taylor Series for e^x: {taylor_exp}")

# Part 3) Improper Integrals:
improperIntegral = integrate(1 / x**2, (x, 1, oo))
print(f"Improper integral result: {improperIntegral}")

# Part 4) Integration by Parts:
x = symbols('x')
u = x
v_prime = x**2
v = integrate(v_prime, x)
# Note: Integration by parts: uv - ∫v * du
result = u * v - integrate(v * 1, x)
print(f"Integration by Parts Result: {result}")

# Part 5) Partial Derivatives:
x, y = symbols('x y')
g = x**2 + y**2
# Partial derivative with respect to x
partial_x = diff(g, x)
# Partial derivative with respect to y
partial_y = diff(g, y)
print(f"Partial Derivatives: ∂g/∂x = {partial_x}, ∂g/∂y = {partial_y}")

# Part 6) Trigonometric Integration:
trig_integral = integrate(sin(x)**2, x)
print(f"Integral of sin^2(x): {trig_integral}")

# Part 7) Power Series:
taylor_exp = series(exp(x), x, 0, 5)  # 5 trms of expansion
print(f"Taylor Series of e^x: {taylor_exp}")

# Part 8) Parametric Equations:
t = symbols('t')
x = t**2
y = t**3
# Note: dy/dx = (dy/dt) / (dx/dt)
dy_dx = diff(y, t) / diff(x, t)
print(f"dy/dx: {dy_dx}")

# Part 9) Arc Length:
y = x**2
dy_dx = diff(y, x)
arc_length = integrate(sqrt(1 + dy_dx**2), (x, 0, 1))
print(f"Arc Length Calculation: {arc_length}")

# Part 10) Disk Integration in Volume:
volume_disk = integrate(pi * (x**2)**2, (x, 0, 1))
print(f"Volume using the Disk Integration Method: {volume_disk}")

# Part 11) Shell Method of Integration in Volume:
volume_shell = integrate(2 * pi * x * x, (x, 0, 1))
print(f"Volume with Shell Method: {volume_shell}")
