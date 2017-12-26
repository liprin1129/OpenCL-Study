import numpy as np
import pyopencl as cl
import sys
from time import time

# Version 1
c_dot_product_kernel = '''
__kernel void dotProduct(
    __global float* inputA,
    __global float* inputB,
    __global float* outputC) {
    
    int glob_x = get_global_id(0);
    int glob_y = get_global_id(1);
    
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);
    
    outputC[glob_y * sizeof(inputA) + glob_x] = dot(inputA[glob_y], inputB[glob_x]);
    }
'''

mat_size = 1024*2
mat_a = np.random.randn(mat_size* mat_size)
mat_b = np.random.randn(mat_size* mat_size)
mat_c = np.empty(mat_size* mat_size)

mat_a = mat_a.astype(np.float32)
mat_b = mat_b.astype(np.float32)
mat_c = mat_c.astype(np.float32)

#print "INPUT MATRIX:", np.shape(np.dot(mat_a, mat_b))
#print "OUTPUT MATRIX: ", np.shape(mat_c)

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

program = cl.Program(context, c_dot_product_kernel).build()
program.dotProduct.set_scalar_arg_dtypes([None, None, None])
program.dotProduct(queue, mat_c.shape, (mat_size/16,), buffer_a, buffer_b, buffer_c)

queue.finish()

run_time = time()-start_time

## Move the kernel's output data to host memory.
#cl.enqueue_copy(queue, mat_c, buffer_c)
cl.enqueue_copy(queue, mat_c, buffer_c)

mflops = 2.0 * mat_size * mat_size * mat_size/(1000000.0* run_time)

print "RESULT: {0}s".format(run_time), mflops
