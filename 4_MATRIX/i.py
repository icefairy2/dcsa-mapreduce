import numpy as np

"""
Use this script to generate input matrices and their cross product and save these data into separate files
"""

A = np.random.rand(1000, 50)

B = np.random.rand(50, 2000)

np.savetxt('A.txt', A)
np.savetxt('B.txt', B)

C = np.dot(A, B)
np.savetxt('C.txt', C)
