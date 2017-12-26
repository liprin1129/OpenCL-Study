import numpy as np
import pyopencl as cl
import sys
from time import time

# Version 1
c_dot_product_kernel = '''
__kernel void dotProduct(
    const int rowA,
    const int comDim,
    __global float* inputA,
    __global float* inputB,
    __global float* outputC) {
        int g_row = get_global_id(0);
        int g_col = get_global_id(1);

        float output = 0.0f;
        int count = 0;
        for (int k = 0; k < comDim; k++){
            //output += inputA[k*rowA + g_row] * inputB[g_col*comDim + k];
            //printf("%f, ", inputA[k*rowA + g_row]);
            //printf("%f, ", inputB[k*comDim + g_col]);
            //printf("%d, ", k*rowA + g_row);
            count += 1;
            printf("%d, ", k);
        }
        outputC[g_col * rowA + g_row] += output;
    }
'''

row_A_size = 3
col_A_size = 3

row_B_size = 3
col_B_size = 2

row_C_size = row_A_size
col_C_size = col_B_size
'''
mat_a = np.random.randn(row_A_size* col_A_size)
mat_b = np.random.randn(row_B_size* col_B_size)
mat_c = np.empty(row_C_size* col_C_size)
'''

mat_a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]) #np.ones([row_A_size, col_A_size])
mat_a = mat_a.reshape(1, -1)
mat_b = np.array([[2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
mat_b = mat_b.reshape(1, -1)
mat_c = np.zeros(row_C_size * col_C_size)

print mat_a
print mat_b

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

row_A = np.int32(row_A_size)
row_B = np.int32(row_B_size)
row_C = np.int32(col_A_size)

program = cl.Program(context, c_dot_product_kernel).build()
program.dotProduct.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])
err = program.dotProduct(queue, [row_A_size, col_A_size], None, row_A, row_C, buffer_a, buffer_b, buffer_c)

queue.finish()

run_time = time()

## Move the kernel's output data to host memory.
#cl.enqueue_copy(queue, mat_c, buffer_c)
cl.enqueue_copy(queue, mat_c, buffer_c)

print "RESULT: {0}s".format(run_time-start_time)
print np.reshape(mat_c, (row_A_size, col_B_size))