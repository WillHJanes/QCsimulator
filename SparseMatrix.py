from __future__ import annotations
import numpy as np
from typing import List


class SparseMatrix:
    def __init__(self, values: List, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.inner_array = np.zeros((3, len(values)))
        indexes = set()
        # Sort by row and then by column
        for i, (r, c, val) in enumerate(sorted(values, key=lambda x: (x[0], x[1]))):
            # Check that there are no duplicate entries for rth row and cth col
            assert (r, c) not in indexes, "Found duplicate entry for row {} and column {}".format(r, c)

            indexes.add((r, c))
            self.inner_array[0][i] = r
            self.inner_array[1][i] = c
            self.inner_array[2][i] = val

    @staticmethod
    def sparsify(matrix: np.array) -> SparseMatrix:
        values = []
        rs, cs = np.nonzero(matrix)
        for i in range(len(rs)):
            r, c = int(rs[i]), int(cs[i])
            values.append((r, c, matrix[r][c]))

        rows, cols = matrix.shape
        return SparseMatrix(values, rows, cols)

    def tensordot(self, matrix: SparseMatrix) -> SparseMatrix:
        self_numpy = self.numpy()
        other_numpy = matrix.numpy()
        tensor_product = np.kron(self_numpy, other_numpy)
        return SparseMatrix.sparsify(tensor_product)

    def numpy(self) -> np.array:
        numpy_matrix = np.zeros((self.rows, self.cols))
        for i in range(self.inner_array.shape[1]):
            r = int(self.inner_array[0][i])
            c = int(self.inner_array[1][i])
            val = self.inner_array[2][i]
            numpy_matrix[r][c] = val
        return numpy_matrix

    def dot(self, matrix: SparseMatrix) -> SparseMatrix:
        return SparseMatrix.sparsify(np.dot(self.numpy(), matrix.numpy()))

    def get_value(self, r: int, c: int):
        cols = self.inner_array.shape[1]
        for i in range(cols):
            if r == self.inner_array[0][i]:
                if c == self.inner_array[1][i]:
                    return self.inner_array[2][i]
        return 0


m1 = np.random.randint(0, 2, (3, 3))
s1 = SparseMatrix.sparsify(m1)
assert np.all(m1 == s1.numpy())

m2 = np.random.randint(0, 2, (3, 3))
s2 = SparseMatrix.sparsify(m2)
assert np.all(np.kron(s2.numpy(), s1.numpy()) == s2.tensordot(s1).numpy())

dim = 4
m3 = SparseMatrix([(0, 0, 1), (2, 1, 1), (1, 1, 2)], dim, dim)
m4 = np.zeros((dim, dim))
m4[0][0] = 1
m4[2][1] = 1
m4[1][1] = 2

assert np.all(m3.numpy() == m4)




