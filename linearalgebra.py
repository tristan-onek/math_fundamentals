# Linear algebra basic in Python
import numpy as np

# Create two vectors:
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition:
print("Vector Addition:", v1 + v2)

# Dot product:
dot_product = np.dot(v1, v2)
print("Dot Product:", dot_product)

# Cross product:
cross_product = np.cross(v1, v2)
print("Cross Product:", cross_product)

# Create matrices:
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition:
print("Matrix Addition:\n", A + B)

# Matrix multiplication:
print("Matrix Multiplication:\n", np.dot(A, B))

# Transpose of a matrix:
print("Transpose of Matrix A:\n", A.T)

# Determinant of a matrix:
det_A = np.linalg.det(A)
print("Determinant of Matrix A:", det_A)

# Eigenvalues and Eigenvectors:
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues of A:", eigenvalues)
print("Eigenvectors of A:\n", eigenvectors)
