import numpy as np
from time import time

mat_a = np.random.randn(2048, 2048)
mat_b = np.random.randn(2048, 2048)
#mat_c = np.empty([1024, 1024])

mat_a = mat_a.astype(np.float32)
mat_b = mat_b.astype(np.float32)
#mat_c = mat_c.astype(np.float32)

start_time = time()

mat_c = np.dot(mat_a, mat_b)

run_time = time()

print "RESULT: {0}s".format(run_time-start_time)