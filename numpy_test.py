import numpy as np
from time import time

row_size = 1024
col_size = 1024

#mat_a = np.random.randn(row_size, col_size)
#mat_b = np.random.randn(row_size, col_size)

mat_a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) #np.ones([row_A_size, col_A_size])
mat_b = np.array([[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])

mat_a = mat_a.astype(np.float32)
mat_b = mat_b.astype(np.float32)
#mat_c = mat_c.astype(np.float32)

start_time = time()

mat_c = np.dot(mat_a, mat_b)

run_time = time()

print "RESULT: {0}s".format(run_time-start_time)
print mat_c

