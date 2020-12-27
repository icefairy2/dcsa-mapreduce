import re

import numpy as np

EXPECTED_MATRIX_PATH = 'C.txt'

# Assuming the task was run by for example 'python matrix_task6.py A.txt B.txt >
RESULTED_MATRIX_PATH = 'C_computed.txt'

RESULTED_LINE_REGEX = re.compile(r'\[(\d+), (\d+)\]\t(.*)$')

# Expected matrix written by i.py has a matrix format within the input file
expected_matrix = np.loadtxt(EXPECTED_MATRIX_PATH)
print('Comparing matrices of shape ' + str(expected_matrix.shape))

resulted_matrix = np.zeros(expected_matrix.shape)

# Resulted matrix written by MRJob has lines of format '[row_index, column_index] element_value'
result_input_file = open(RESULTED_MATRIX_PATH, "r")
for line in result_input_file:
    row_index, column_index, element_value = RESULTED_LINE_REGEX.match(line).groups()
    resulted_matrix[int(row_index), int(column_index)] = float(element_value)

result_input_file.close()

# Print True if matrices are equal  ***see below the explanation for using 'allclose()'
print(np.allclose(expected_matrix, resulted_matrix, atol=1e-15, rtol=1e-15))

# *** The floating point representation in computer systems in imprecise because the binary system does not
# have a perfect representation for decimals ending in 3 or multiples of 3. From this reason, np.array_equal()
# gives false on some elements even though they are equal to the human eye.
# np.allclose() uses a tolerance margin for the number of decimals which should be equal in order for the
# compared numbers to be considered equal. For more info see the documentation at
# https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
