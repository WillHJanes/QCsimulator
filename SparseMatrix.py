import numpy as np
from typing import List


def SparseMatrix():
    def __init__(self, values: List):
        self.inner_array = np.zeros((3, len(values)))

        rows = set()
        cols = set()
        # Sort by row and then by column
        for i, (r, c, val) in enumerate(sorted(values, key=lambda x: (x[0], x[1]))):
            # Check that there are no duplicate entries for rth row and cth col
            assert not (r in rows and c in cols)

            rows.add(r)
            cols.add(c)

            self.inner_array[0][i] = r
            self.inner_array[1][i] = c
            self.inner_array[2][i] = val

    def sparsify(self, matrix: np.array) -> SparseMatrix:
        values = []
        for idx in np.nonzero(matrix):
            r, c = idx[0], idx[1]
            values.append((r, c, matrix[r][c]))
        return SparseMatrix(values)

    def tensor_product(self, matrix: SparseMatrix) -> SparseMatrix:
        return np.tensordot(self.numpy(), matrix)

    def numpy(self) -> np.array:
        rows = self.inner_array[0][-1]
        cols = self.inner_array[1][-1]
        numpy_matrix = np.zeros((rows, cols))
        for i in range(len(self.inner_array.shape[1])):
            r = self.inner_array[0][i]
            c = self.inner_array[1][i]
            val = self.inner_array[2][i]
            numpy_matrix[r][c] = val

        return numpy_matrix

    def dot(self, matrix: SparseMatrix) -> SparseMatrix:
        return np.dot(self.numpy(), matrix.numpy())