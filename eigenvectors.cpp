#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

typedef vector<vector<double>> Matrix;
typedef vector<double> Vector;

Vector multiply(const Matrix &matrix, const Vector &vec) 
{
    Vector result(matrix.size(), 0.0);
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}

double norm(const Vector &vec) 
{
    double sum = 0.0;
    for (double v : vec) 
    {
        sum += v * v;
    }
    return sqrt(sum);
}

void normalize(Vector &vec) 
{
    double vecNorm = norm(vec);
    for (double &v : vec) 
    {
        v /= vecNorm;
    }
}

// Power Iteration Method:
Vector powerIteration(const Matrix &matrix, int maxIterations = 1000, double tolerance = 1e-6) 
{
    size_t n = matrix.size();
    Vector eigenvector(n, 1.0); // Start with an initial guess (non-zero vector)

    for (int iter = 0; iter < maxIterations; ++iter) {
        Vector nextVec = multiply(matrix, eigenvector);
        normalize(nextVec);

        // Check for convergence
        double diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            diff += abs(nextVec[i] - eigenvector[i]);
        }

        if (diff < tolerance) {
            cout << "Converged in " << iter + 1 << " iterations." << endl;
            return nextVec;
        }

        eigenvector = nextVec;
    }

    cerr << "Power iteration did not converge within the maximum number of iterations." << endl;
    return eigenvector;
}

int main() 
{
    // Example matrix
    Matrix matrix = {
        {4, 1},
        {2, 3}
    };

    // Find the dominant eigenvector
    Vector dominantEigenvector = powerIteration(matrix);

    // Print the result
    cout << "Dominant Eigenvector: ";
    for (double val : dominantEigenvector) 
    {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
