#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>       // For matrix operations and spectral methods
#include <Eigen/Sparse>      // For sparse matrices (FEM)
#include <matplot/matplot.h> // For plotting (install matplot++ for this)

// Parameters for numerical solvers
const double PI = 3.141592653589793;

// Function to display a 2D vector as a matrix (helper function)
void display_matrix(const std::vector<std::vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            std::cout << val << "\t";
        }
        std::cout << "\n";
    }
}

// 1. Heat Equation: Explicit Finite Difference Method
void heat_equation_fdm() {
    std::cout << "1. Heat Equation: Finite Difference Method\n";

    const int nx = 50, nt = 200; // Number of spatial and temporal points
    const double alpha = 0.01, dx = 1.0 / nx, dt = 0.1 * dx * dx / alpha;

    std::vector<std::vector<double>> u(nx, std::vector<double>(nt, 0.0));

    // Initial condition: u(x, 0) = sin(pi * x)
    for (int i = 0; i < nx; ++i) {
        u[i][0] = sin(PI * i * dx);
    }

    // Time-stepping
    for (int n = 0; n < nt - 1; ++n) {
        for (int i = 1; i < nx - 1; ++i) {
            u[i][n + 1] = u[i][n] + alpha * dt / (dx * dx) * (u[i + 1][n] - 2 * u[i][n] + u[i - 1][n]);
        }
    }

    // Display and plot solution
    display_matrix(u);
    matplot::imshow(u);
    matplot::colorbar();
    matplot::title("Heat Equation Solution (FDM)");
    matplot::show();
}

// 2. Wave Equation: Spectral Method
void wave_equation_spectral() {
    std::cout << "2. Wave Equation: Spectral Method\n";

    const int N = 64; // Number of Fourier modes
    const double T = 2.0, dt = 0.01;
    int nt = T / dt;

    Eigen::VectorXd u = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd u_prev = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd u_next = Eigen::VectorXd::Zero(N);

    // Initial conditions: u(x, 0) = sin(pi * x)
    for (int i = 0; i < N; ++i) {
        u[i] = sin(PI * i / N);
        u_prev[i] = u[i];
    }

    // Time stepping using spectral method
    for (int t = 0; t < nt; ++t) {
        Eigen::FFT<double> fft;
        Eigen::VectorXd u_fft = fft.fwd(u);
        u_fft = -u_fft.array().square(); // Apply Laplace operator in Fourier space

        // Update
        u_next = 2 * u - u_prev + dt * dt * u_fft;
        u_prev = u;
        u = u_next;
    }

    std::cout << "Wave Equation Solution (Spectral):\n" << u << "\n";
}

// 3. Finite Element Method: Poisson Equation
void poisson_fem() {
    std::cout << "3. Finite Element Method: Poisson Equation\n";

    const int nx = 10; // Number of nodes
    const double L = 1.0, dx = L / (nx - 1);

    Eigen::SparseMatrix<double> A(nx, nx);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(nx);
    Eigen::VectorXd u = Eigen::VectorXd::Zero(nx);

    // Assembly of stiffness matrix and load vector
    for (int i = 0; i < nx - 1; ++i) {
        A.insert(i, i) += 2.0 / dx;
        A.insert(i, i + 1) += -1.0 / dx;
        A.insert(i + 1, i) += -1.0 / dx;
        A.insert(i + 1, i + 1) += 2.0 / dx;

        b[i] += 1.0 * dx; // Example load function
    }

    // Solve linear system
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    u = solver.solve(b);

    std::cout << "FEM solution for Poisson Equation:\n" << u << "\n";

    // Visualization
    std::vector<double> x(nx), y(nx);
    for (int i = 0; i < nx; ++i) {
        x[i] = i * dx;
        y[i] = u[i];
    }

    matplot::plot(x, y);
    matplot::title("FEM Solution to Poisson Equation");
    matplot::show();
}

// Main Function
int main() {
    heat_equation_fdm();
    wave_equation_spectral();
    poisson_fem();
    return 0;
}
