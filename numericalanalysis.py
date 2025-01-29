# imports:
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.integrate import quad
# end imports


# Root Finding - Solve for f(x) = x^2 - 4:
f = lambda x: x**2 - 4
root = fsolve(f, x0=1)  # initial guess at x0 = 1
print(f"Root of x^2 - 4: {root}")

# Cubic Root Finding Example - Solve f(x) = x^3 - 6x^2 + 11x - 6:
f2 = lambda x: x**3 - 6*x**2 + 11*x - 6
root2 = fsolve(f2, x0=[1, 2, 3])  # Multiple initial guesses
print(f"Roots of x^3 - 6x^2 + 11x - 6: {root2}")

# Trigonometric Root Finding - Solve f(x) = sin(x) - 0.5
f3 = lambda x: np.sin(x) - 0.5
root3 = fsolve(f3, x0=[0, 3, 6])  # Multiple guesses for periodic function
print(f"Roots of sin(x) - 0.5: {root3}")

# Interpolation:
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 2.7, 5.8, 6.6, 7.5])
interp_func = interp1d(x, y, kind='cubic')
x_new = np.linspace(0, 4, 100)
y_new = interp_func(x_new)
plt.plot(x, y, 'o', label='Data Points')
plt.plot(x_new, y_new, '-', label='Cubic Interpolation')
plt.legend()
plt.show()

# Numerical Differentiation:
f_prime = lambda x: np.sin(x)  # Derivative of sin(x) is cos(x)
deriv_at_pi_4 = derivative(f_prime, np.pi/4, dx=1e-6)
print(f"The numerical derivative of sin(x) at pi/4: {deriv_at_pi_4}")

# Numerical Integration:
integrand = lambda x: np.exp(-x**2)
result, error = quad(integrand, 0, 1)
print(f"Integral of exp(-x^2) from 0 to 1: {result}")
