import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize
from numpy.linalg import eig, solve
from scipy.fft import fft, ifft

# 1. Linear Algebra: Eigenvalues and Eigenvectors
def eigenvalues_eigenvectors():
    print("1. Linear Algebra: Eigenvalues and Eigenvectors")
    A = np.array([[4, -2], [1, 1]])  # Example matrix
    values, vectors = eig(A)
    print("Matrix A:\n", A)
    print("Eigenvalues:", values)
    print("Eigenvectors:\n", vectors)
    print()

# 2. Numerical Integration
def numerical_integration():
    print("2. Numerical Integration")
    # Integrate f(x) = x^2 from 0 to 1
    def f(x):
        return x ** 2

    result, _ = quad(f, 0, 1)
    print("Integral of x^2 from 0 to 1:", result)
    print()

# 3. Ordinary Differential Equations (ODEs)
def solve_ode():
    print("3. Ordinary Differential Equations (ODEs)")
    # Solve dy/dx = -2y, y(0) = 1
    def dydx(t, y):
        return -2 * y

    t_span = (0, 5)
    y0 = [1]
    sol = solve_ivp(dydx, t_span, y0, t_eval=np.linspace(0, 5, 100))

    # Plot solution
    plt.plot(sol.t, sol.y[0], label="y(t)")
    plt.title("Solution of dy/dx = -2y")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# 4. Optimization
def optimization():
    print("4. Optimization")
    # Minimize f(x) = (x-3)^2
    def f(x):
        return (x - 3) ** 2

    result = minimize(f, x0=0)
    print("Optimal value of x:", result.x[0])
    print("Minimum value of f(x):", result.fun)
    print()

# 5. Fourier Analysis
def fourier_analysis():
    print("5. Fourier Analysis")
    # Generate a sine wave and compute its Fourier transform
    t = np.linspace(0, 1, 500, endpoint=False)
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    freq = fft(signal)

    # Plot original signal and its frequency spectrum
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, signal)
    plt.title("Original Signal")
    plt.subplot(1, 2, 2)
    plt.plot(np.abs(freq))
    plt.title("Frequency Spectrum")
    plt.show()

# 6. Partial Differential Equations (Heat Equation)
def solve_heat_equation():
    print("6. Partial Differential Equations (Heat Equation)")
    # Solve u_t = alpha*u_xx on [0,1] with u(0,t)=u(1,t)=0
    nx, nt = 50, 100  # Grid points in space and time
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, 0.2, nt)
    alpha = 0.01
    dx, dt = x[1] - x[0], t[1] - t[0]

    u = np.zeros((nx, nt))
    u[:, 0] = np.sin(np.pi * x)  # Initial condition

    for n in range(0, nt - 1):
        for i in range(1, nx - 1):
            u[i, n + 1] = u[i, n] + alpha * dt / dx**2 * (u[i + 1, n] - 2 * u[i, n] + u[i - 1, n])

    plt.imshow(u, extent=[0, 0.2, 0, 1], origin="lower", aspect="auto", cmap="hot")
    plt.colorbar(label="Temperature")
    plt.title("Heat Equation Solution")
    plt.xlabel("Time")
    plt.ylabel("Space")
    plt.show()

# 7. Stochastic Processes (Monte Carlo Simulation)
def monte_carlo_simulation():
    print("7. Stochastic Processes: Monte Carlo Simulation")
    # Estimate pi using random sampling
    np.random.seed(42)
    num_points = 10000
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    inside_circle = x**2 + y**2 <= 1

    pi_estimate = 4 * np.sum(inside_circle) / num_points
    print("Estimated value of pi:", pi_estimate)

    # Plot points
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=inside_circle, cmap="coolwarm", s=1)
    plt.title("Monte Carlo Simulation of Pi")
    plt.show()

# 8. Regression Analysis (Linear Regression)
def regression_analysis():
    print("8. Regression Analysis")
    # Generate synthetic data
    np.random.seed(42)
    X = np.linspace(0, 10, 100)
    Y = 3 * X + 5 + np.random.normal(0, 2, size=100)

    # Perform linear regression
    A = np.vstack([X, np.ones(len(X))]).T
    coef, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    print("Linear regression coefficients (slope, intercept):", coef)

    # Plot
    plt.scatter(X, Y, label="Data")
    plt.plot(X, coef[0] * X + coef[1], color="red", label="Fitted Line")
    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Run the demonstrations
eigenvalues_eigenvectors()
numerical_integration()
solve_ode()
optimization()
fourier_analysis()
solve_heat_equation()
monte_carlo_simulation()
regression_analysis()
