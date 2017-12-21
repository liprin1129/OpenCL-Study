import numpy as np
import pyopencl as cl
import sys
from time import time

c_dot_product_kernel = '''
__kernel void dotProduct(
    const int widthA,
    const int heightA,
    const int widthB,
    const int heightB,
    __global float* inputA,
    __global float* inputB,
    __global float* outputC) {
        int row = get_global_id(1);
        int col = get_global_id(0);

        double sum=0.0f;

        for (int i = 0; i < widthA; i++) {
            sum += inputA[row * widthA + i] * inputB[i * widthB + col];
        }
        outputC[row * widthB + col] = sum;
    }
'''

mat_a = np.random.randn(100, 100)
mat_b = np.random.randn(100, 10)
mat_c = np.empty([100, 10])

mat_a = mat_a.astype(np.float32)
mat_b = mat_b.astype(np.float32)
mat_c = mat_c.astype(np.float32)
'''
mat_a = np.ones([10, 10], dtype=np.float32)*2
mat_b = np.ones([10, 10], dtype=np.float32)*5
mat_c = np.empty([10, 10], dtype=np.float32)
'''

print "INPUT MATRIX:", np.shape(np.dot(mat_a, mat_b))
print "OUTPUT MATRIX: ", np.shape(mat_c)

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

widthA = 10
widthA = np.int32(widthA)
heightA = 10
heightA = np.int32(heightA)
widthB = 10
widthB = np.int32(widthB)
heightB = 10
heightB = np.int32(heightB)

program = cl.Program(context, c_dot_product_kernel).build()
program.dotProduct.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, np.int32, None, None, None])
program.dotProduct(queue, mat_c.shape, None, widthA, heightA, widthB, heightB, buffer_a, buffer_b, buffer_c)

queue.finish()

run_time = time()

## Move the kernel's output data to host memory.
#cl.enqueue_copy(queue, mat_c, buffer_c)
cl.enqueue_copy(queue, mat_c, buffer_c)

print "RESULT: {0} in {1}s".format(mat_c, run_time-start_time)