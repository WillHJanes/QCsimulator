from __future__ import annotations
import numpy as np
from typing import List


class SparseMatrix:
    """
    This is a class for sparse matrix interface. It holds the values in an inner numpy array in the following format:

    r1 | r2 | r3 | ... | rn
    c1 | c2 | c3 | ... | cn
    v1 | v2 | v3 | ... | vn

    where r is the row index, c - column index and v is the values stored.
    The class implements the dot and tensordot products for sparse matrices, as well as provides conversion tools for
    conversion between numpy array and Sparse Matrix.
    """

    def __init__(self, values: List, rows: int, cols: int):
        """
        This is the constructor method for the Sparse Matrix
        @param values: List of tuples of (row, column, val) to specify the row and column where the val v is inserted
        @param rows: Number of rows in the matrix
        @param cols: Number of rows in the matrix
        """
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
        """
        Static method for generating Sparse Matrix for numpy matrix
        @param matrix: Numpy matrix
        @return: Sparse matrix generated
        """
        values = []
        rs, cs = np.nonzero(matrix)
        for i in range(len(rs)):
            r, c = int(rs[i]), int(cs[i])
            values.append((r, c, matrix[r][c]))

        rows, cols = matrix.shape
        return SparseMatrix(values, rows, cols)

    def tensordot(self, matrix: SparseMatrix) -> SparseMatrix:
        """
        Tensor product method for Sparse Matrices
        @param matrix: Sparse Matrix to be dotted with
        @return: Tensor product of the Sparse Matrices
        """
        return SparseMatrix.sparsify(np.kron(self.numpy(), matrix.numpy()))

    def numpy(self) -> np.array:
        """
        Convert Sparse Matrix to numpy array
        @return: Numpy array
        """
        numpy_matrix = np.zeros((self.rows, self.cols))
        for i in range(self.inner_array.shape[1]):
            r = int(self.inner_array[0][i])
            c = int(self.inner_array[1][i])
            val = self.inner_array[2][i]
            numpy_matrix[r][c] = val
        return numpy_matrix

    def dot(self, matrix: SparseMatrix) -> SparseMatrix:
        """
        Dot product between two Sparse Matrices
        @param matrix: Sparse Matrix to be dotted with
        @return: Result Sparse Matrix
        """
        assert self.cols == matrix.rows, \
            "Matrices dimensions do not match, {} != {}".format(self.cols, self.rows)

        values = []
        for i, row in enumerate(self.get_nonzero_rows()):
            row_vals = self.get_row(int(row))
            for j, col in enumerate(matrix.get_nonzero_cols()):
                col_vals = matrix.get_col(int(col))
                val = 0
                for c in row_vals.keys():
                    if c in col_vals.keys():
                        val += row_vals[c] * col_vals[c]
                values.append((int(row), int(col), val))

        return SparseMatrix(values, self.rows, matrix.cols)

    def get_row(self, r: int) -> dict:
        """
        Get all values in the row r
        @param r: Selected row
        @return: Dictionary of values in format vals[col] with value at row r and column col
        """
        vals = {}
        for i in range(self.inner_array.shape[1]):
            # The inner array is sorted by row and column. This reduces the number of iterations
            if r < self.inner_array[0][i]:
                break
            elif self.inner_array[0][i] == r:
                vals[self.inner_array[1][i]] = self.inner_array[2][i]
        return vals

    def get_col(self, c: int) -> dict:
        """
        Get all values in the column c
        @param c: Selected column
        @return: Dictionary of values in format vals[r] with value at row r and column c
        """
        vals = {}
        for i in range(self.inner_array.shape[1]):
            if self.inner_array[1][i] == c:
                vals[self.inner_array[0][i]] = self.inner_array[2][i]
        return vals

    def get_value(self, r: int, c: int) -> float:
        """
        @param r: Selected row
        @param c: Selected column
        @return: Value at selected row and column
        """
        for i in range(self.inner_array.shape[1]):
            # The inner array is sorted by row and column. This reduces the number of iterations
            if r < self.inner_array[0][i]:
                break
            elif r == self.inner_array[0][i]:
                if c == self.inner_array[1][i]:
                    return self.inner_array[2][i]
        return 0

    def get_nonzero_rows(self) -> List[float]:
        """
        Get all rows that have entries
        @return: List of rows
        """
        return np.unique(self.inner_array[0]).tolist()

    def get_nonzero_cols(self) -> List[float]:
        """
        Get all columns that have entries
        @return: List of all columns
        """
        return np.unique(self.inner_array[1]).tolist()


m1 = np.random.randint(0, 2, (3, 2))
s1 = SparseMatrix.sparsify(m1)
assert np.all(m1 == s1.numpy())

m2 = np.random.randint(0, 2, (2, 3))
s2 = SparseMatrix.sparsify(m2)

numpy_dot = m1.dot(m2)
sparse_dot = s1.dot(s2)
assert np.all(numpy_dot == sparse_dot.numpy())

assert np.all(np.kron(s2.numpy(), s1.numpy()) == s2.tensordot(s1).numpy())

dim = 4
m3 = SparseMatrix([(0, 0, 1), (2, 1, 1), (1, 1, 2)], dim, dim)
m4 = np.zeros((dim, dim))
m4[0][0] = 1
m4[2][1] = 1
m4[1][1] = 2

assert np.all(m3.numpy() == m4)




