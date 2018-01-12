import numpy as np
import pyopencl as cl
import sys
from time import time

# Version 1
c_dot_product_kernel = '''
__kernel void dotProduct(
const int N,
__global float *A,
__global float *B,
__global float *C)
{
    int j, k;
    int i = get_global_id(0);
    float tmp;
    for (int p = 0; p < 10; p++) {
        for (j = 0; j < N; j++) {
            tmp = 0.0f;
            for (k = 0; k < N; k++) {
                tmp += A[i*N+k]*B[k*N+j];
                }
        C[i*N+j] = tmp;
        }
    }
}
'''

row_size = 1024
col_size = 1024

mat_a = np.random.randn(row_size* col_size)
mat_b = np.random.randn(row_size* col_size)
mat_c = np.empty(row_size* col_size)

mat_a = mat_a.astype(np.float32)
mat_b = mat_b.astype(np.float32)
mat_c = mat_c.astype(np.float32)

# #############
# Set up OpenCL
# #############
platform = cl.get_platforms()
my_gpu_devices = [platform[0].get_devices(device_type=cl.device_type.GPU)[0]]
context = cl.Context(devices=my_gpu_devices)
#context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Create Opencl Buffers
buffer_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = mat_a)
buffer_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = mat_b)
buffer_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, mat_c.nbytes)

# Program
start_time = time()

widthA = row_size
widthA = np.int32(widthA)

widthB = row_size
widthB = np.int32(widthB)

program = cl.Program(context, c_dot_product_kernel).build()
program.dotProduct.set_scalar_arg_dtypes([np.int32, None, None, None])
program.dotProduct(queue, [1024], [1024/16], widthA, buffer_a, buffer_b, buffer_c)

queue.finish()

run_time = time()

## Move the kernel's output data to host memory.
#cl.enqueue_copy(queue, mat_c, buffer_c)
cl.enqueue_copy(queue, mat_c, buffer_c)

print "RESULT: {0}s".format(run_time-start_time)
