import numpy as np
import pyopencl as cl
import sys
from time import time

# Version 1
c_dot_product_kernel = '''
__kernel void dotProduct(
    const int rowA,
    const int rowB,
    const int comDim,
    __global float* inputA,
    __global float* inputB,
    __global float* outputC) {
        int g_col = get_global_id(0);
        int g_row = get_global_id(1);

        double sum=0.0f;

        for (int k = 0; k < comDim; k++) {
            sum += inputA[g_row * rowA + k] * inputB[k * rowB + g_col];
        }
        outputC[g_row * rowA + g_col] = sum;
    }
'''

row_A_size = 1024
row_B_size = 1024
row_C_size = 1024

mat_a = np.random.randn(row_A_size* row_A_size)
mat_b = np.random.randn(row_B_size* row_B_size)
mat_c = np.empty(row_C_size* row_C_size)

mat_a = mat_a.astype(np.float32)
mat_b = mat_b.astype(np.float32)
mat_c = mat_c.astype(np.float32)

# #############
# Set up OpenCL
# #############

context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Create Opencl Buffers
buffer_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = mat_a)
buffer_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = mat_b)
buffer_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, mat_c.nbytes)

# Program
start_time = time()

row_A_size = np.int32(row_A_size)
row_B_size = np.int32(row_B_size)
row_C_size = np.int32(row_C_size)

program = cl.Program(context, c_dot_product_kernel).build()
program.dotProduct.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])
program.dotProduct(queue, [1024, 1024], None, row_A_size, row_B_size, row_C_size, buffer_a, buffer_b, buffer_c)

queue.finish()

run_time = time()

## Move the kernel's output data to host memory.
#cl.enqueue_copy(queue, mat_c, buffer_c)
cl.enqueue_copy(queue, mat_c, buffer_c)

print "RESULT: {0}s".format(run_time-start_time)