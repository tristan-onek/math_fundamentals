# Imports:

from sympy import diff, Symbol, limit, integrate
from scipy.optimize import newton
from scipy.integrate import quad
# End imports

# 1) Derivatives:

f = x**3 + 2*x**2 - x + 5
# First derivative...
f_prime = diff(f, x)
print(f"First derivative: {f_prime}")

# 2) Limits:

x = Symbol('x')
f = (x**2 - 1) / (x - 1)
# Compute limit as x approaches 1...
limit_value = limit(f, x, 1)
print(f"Limit as x -> 1: {limit_value}")

# 3) Integrals:

# Indefinite integral...
indef_integral = integrate(f, x)
print(f"Indefinite integral: {indef_integral}")

# Definite integral from 0 to 2...
def_integral = integrate(f, (x, 0, 2))
print(f"Definite integral from 0 to 2: {def_integral}")

# 4) Newton's Method:

import numpy as np
def f(x):
    return x**3 - 6*x**2 + 11*x - 6

def f_prime(x):
    return 3*x**2 - 12*x + 11

# Newton's method for finding roots...
root = newton(f, x0=2, fprime=f_prime)
print(f"Root: {root}")

# 5) Definite Integrals:
# Integrate f(x) = x^2 from 0 to 2
result, error = quad(lambda x: x**2, 0, 2)
print(f"Numerical integration result: {result}")
