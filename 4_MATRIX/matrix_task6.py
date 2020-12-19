import os

import numpy as np
from mrjob.job import MRJob
from mrjob.step import MRStep

# ****** CONSTANTS *******
A_NAME = 'A'
A_ROWS = 1000
A_COLUMNS = 50

B_NAME = 'B'
B_ROWS = 50
B_COLUMNS = 2000


class MatrixMatrixMultiplication(MRJob):

    def mapper_raw(self, input_path, input_uri):
        """
        This mapper is needed to be able to read the input file at once
        and give the result to the next mapper as a whole.
        This is overriding the default implementation from MRJob.
        :param input_path: path to the input file, will be received from command line
        :param input_uri: URI for HDFS, S3 etc., not relevant for us
        :return: (matrix_name, matrix) matrix name extracted from file name, and
                 matrix elements as a list of lists
        """
        # Using numpy we can directly construct the matrix from the txt files
        matrix = np.loadtxt(input_path)

        # For identifying purposes, get the file name as the name of the matrix
        # ex. .../A.txt -> 'A'
        matrix_name = os.path.splitext(os.path.basename(input_path))[0]

        yield matrix_name, matrix.tolist()

    def mapper_matrix_elements(self, matrix_name, matrix):
        """
        According to the algorithm from the course, this mapper splits the matrices onto elements
        and associates identifying values to them for later processing
        :param matrix_name: 'A' or 'B' from previous mapper (can be else if code is modified accordingly)
        :param matrix: matrix elements in list representation
        :return: ((i, k), (name, j, element))
                 i - row in resulting matrix on which the element will be relevant
                 k - column in resulting matrix on which the element will be relevant
                 name - matrix name as defined
                 j - iterating element for later processing
                 element - the matrix element value
        """
        if matrix_name == A_NAME:
            for k in range(B_COLUMNS):
                for i in range(A_ROWS):
                    for j in range(A_COLUMNS):
                        yield (i, k), (A_NAME, j, matrix[i][j])

        if matrix_name == B_NAME:
            for i in range(A_ROWS):
                for j in range(B_ROWS):
                    for k in range(B_COLUMNS):
                        yield (i, k), (B_NAME, j, matrix[j][k])

    def reducer_matrix_prod_elements(self, key_pair, matrix_element_combined):
        """
        This reducer creates a list from all element constructs with the same key
        :param key_pair: (i, k) the row and column of resulting matrix
        :param matrix_element_combined: (name, j, element), a construct composed by the matrix name,
               an integer and a matrix element value
        :return: ((i, k), [(name, j, element), ...])
        """
        result = []
        for matrix_element_tuple in matrix_element_combined:
            result.append(matrix_element_tuple)
        yield key_pair, result

    def reducer_multiply_elements(self, key_pair, matrix_elements):
        """
        This reducer calculates the resulting matrix element value by position
        :param key_pair: (i, k) the position of the result value in the resulting matrix
        :param matrix_elements: [(name, j, element), ...]
        :return: ((i, k), value) where (i, k) is the same as the parameter and value is the
                 resulting computed element
        """
        # The parameter matrix_elements is a generator in MRJob's definition and the sorted()
        # method returns us a list of a sorted list, so we take the first (0) value, which is
        # the desired list
        sorted_by_matrix_name = sorted(matrix_elements, key=lambda x: x[0])[0]

        # Separate the elements belonging to the two matrices
        list_A = []
        list_B = []
        for element in sorted_by_matrix_name:
            if element[0] == A_NAME:
                list_A.append(element)
            if element[0] == B_NAME:
                list_B.append(element)

        # Sort the elements by the iterating value j
        list_A.sort(key=lambda x: x[1])
        list_B.sort(key=lambda x: x[1])

        ret = 0
        for j in range(A_COLUMNS):
            ret += list_A[j][2] * list_B[j][2]

        yield key_pair, ret

    def steps(self):
        return [
            MRStep(mapper_raw=self.mapper_raw),
            MRStep(mapper=self.mapper_matrix_elements,
                   reducer=self.reducer_matrix_prod_elements),
            MRStep(reducer=self.reducer_multiply_elements),
        ]


if __name__ == '__main__':
    MatrixMatrixMultiplication.run()
