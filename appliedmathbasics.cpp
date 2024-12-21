#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <Eigen/Dense> // For linear algebra
#include <gsl/gsl_integration.h> // GNU Scientific Library for integration
#include <matplot/matplot.h> // For plotting (install matplot++ for this to work)

// 1. Linear Algebra: Eigenvalues and Eigenvectors
void eigenvalues_eigenvectors() {
    std::cout << "1. Linear Algebra: Eigenvalues and Eigenvectors\n";
    Eigen::Matrix2d A;
    A << 4, -2, 1, 1;

    Eigen::EigenSolver<Eigen::Matrix2d> solver(A);
    std::cout << "Matrix A:\n" << A << "\n";
    std::cout << "Eigenvalues:\n" << solver.eigenvalues() << "\n";
    std::cout << "Eigenvectors:\n" << solver.eigenvectors() << "\n\n";
}

// 2. Numerical Integration
double f(double x, void* params) {
    return x * x; // Example: Integrand is f(x) = x^2
}

void numerical_integration() {
    std::cout << "2. Numerical Integration\n";
    gsl_integration_workspace* w = gsl_integration_workspace_alloc(1000);
    gsl_function F;
    F.function = &f;

    double result, error;
    gsl_integration_qags(&F, 0, 1, 1e-7, 1e-7, 1000, w, &result, &error);

    std::cout << "Integral of x^2 from 0 to 1: " << result << "\n\n";
    gsl_integration_workspace_free(w);
}

// 3. Ordinary Differential Equations (ODEs)
void solve_ode() {
    std::cout << "3. Ordinary Differential Equations (ODEs)\n";
    // ODE solver logic here (similar to solving dy/dx = -2y)
    // Can use Boost libraries or custom implementation
    std::cout << "ODE solved (not shown, use Boost Odeint for detailed implementation).\n\n";
}

// 4. Optimization
void optimization() {
    std::cout << "4. Optimization\n";
    auto f = [](double x) { return (x - 3) * (x - 3); };
    double x = 0.0, step = 0.01, grad;

    // Gradient descent example
    for (int i = 0; i < 1000; ++i) {
        grad = 2 * (x - 3);
        x -= step * grad;
        if (std::fabs(grad) < 1e-6) break;
    }

    std::cout << "Optimal value of x: " << x << "\nMinimum value of f(x): " << f(x) << "\n\n";
}

// 5. Fourier Analysis
void fourier_analysis() {
    std::cout << "5. Fourier Analysis\n";
    const int N = 500;
    std::vector<double> signal(N), freq(N);

    for (int i = 0; i < N; ++i) {
        signal[i] = sin(2 * M_PI * 50 * i / N) + sin(2 * M_PI * 120 * i / N);
    }

    matplot::plot(signal);
    matplot::title("Original Signal");
    matplot::show();
}

// 6. Partial Differential Equations (Heat Equation)
void solve_heat_equation() {
    std::cout << "6. Partial Differential Equations (Heat Equation)\n";
    const int nx = 50, nt = 100;
    double alpha = 0.01, dx = 1.0 / nx, dt = 0.2 / nt;

    std::vector<std::vector<double>> u(nx, std::vector<double>(nt, 0.0));
    for (int i = 0; i < nx; ++i) u[i][0] = sin(M_PI * i * dx); // Initial condition

    for (int n = 0; n < nt - 1; ++n) {
        for (int i = 1; i < nx - 1; ++i) {
            u[i][n + 1] = u[i][n] + alpha * dt / (dx * dx) * (u[i + 1][n] - 2 * u[i][n] + u[i - 1][n]);
        }
    }

    matplot::imshow(u);
    matplot::colorbar();
    matplot::title("Heat Equation Solution");
    matplot::show();
}

// 7. Stochastic Processes (Monte Carlo Simulation)
void monte_carlo_simulation() {
    std::cout << "7. Stochastic Processes: Monte Carlo Simulation\n";
    const int num_points = 10000;
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    int inside_circle = 0;
    for (int i = 0; i < num_points; ++i) {
        double x = distribution(generator);
        double y = distribution(generator);
        if (x * x + y * y <= 1.0) ++inside_circle;
    }

    double pi_estimate = 4.0 * inside_circle / num_points;
    std::cout << "Estimated value of pi: " << pi_estimate << "\n\n";
}

// 8. Regression Analysis (Linear Regression)
void regression_analysis() {
    std::cout << "8. Regression Analysis\n";
    const int n = 100;
    std::vector<double> X(n), Y(n);
    double slope = 3.0, intercept = 5.0;

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 2.0);

    for (int i = 0; i < n; ++i) {
        X[i] = i;
        Y[i] = slope * X[i] + intercept + noise(generator);
    }

    double sum_x = std::accumulate(X.begin(), X.end(), 0.0);
    double sum_y = std::accumulate(Y.begin(), Y.end(), 0.0);
    double sum_xy = std::inner_product(X.begin(), X.end(), Y.begin(), 0.0);
    double sum_xx = std::inner_product(X.begin(), X.end(), X.begin(), 0.0);

    double a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double b = (sum_y - a * sum_x) / n;

    std::cout << "Linear regression coefficients:\nSlope: " << a << ", Intercept: " << b << "\n";
}

// Main function
int main() {
    eigenvalues_eigenvectors();
    numerical_integration();
    solve_ode();
    optimization();
    fourier_analysis();
    solve_heat_equation();
    monte_carlo_simulation();
    regression_analysis();

    return 0;
}
