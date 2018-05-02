import numpy as np
import time

A = np.random.random_sample((1024, 1))
B = np.random.random_sample((1024, 1))

start = time.time()
C = A + B
print(time.time() - start)
