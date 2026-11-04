import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

def task1(matrix: np.ndarray ):
    if matrix.shape[0] != matrix.shape[1] or matrix.ndim != 2:
       return "Eigenvalues and eigenvectors cant be computed"

    vals, vects = np.linalg.eig(matrix)
    vals = np.real(vals)
    vects = np.real(vects)
    for i in range(0, len(vals)):
        if np.allclose(np.dot(matrix, vects[:, i]), vals[i] * vects[:, i]):
            print(f"{vals[i]}  - {vects[:, i]}")



test_matrix = np.random.randint(-10, 10, (3, 3)).astype(float)
print(test_matrix)
task1(test_matrix)