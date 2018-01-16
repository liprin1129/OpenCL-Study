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
__global float *C,
__local float *Bwrk)
{
    int j, k;
    int i = get_global_id(0);
    int iloc =  get_local_id(0);
    int nloc =  get_local_size(0);
    
    float tmp;
    float Awrk[1024];
    
    if (i < N) {
        for (int p = 0; p < 1; p++) {
            for (k = 0; k < N; k++) {
                Awrk[k] = A[i*N+k];
            }
            
            for (j = 0; j < N; j++) {
                barrier(CLK_LOCAL_MEM_FENCE);
                for (k=iloc; k<N; k+=nloc) {
                    Bwrk[k] = B[k* N+j];
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                tmp = 0.0f;
                for (k = 0; k < N; k++) {
                    tmp += Awrk[k] * Bwrk[k];
                }
            C[i*N+j] = tmp;
            
            barrier(CLK_LOCAL_MEM_FENCE);
            }
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

widthA = row_size
widthA = np.int32(widthA)

widthB = row_size
widthB = np.int32(widthB)

# #############
# Set up OpenCL
# #############
platform = cl.get_platforms()
my_gpu_devices = [platform[0].get_devices(device_type=cl.device_type.GPU)[1]]
context = cl.Context(devices=my_gpu_devices)
#context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Create Opencl Buffers
buffer_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = mat_a)
buffer_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = mat_b)
buffer_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, mat_c.nbytes)

# Program
program = cl.Program(context, c_dot_product_kernel).build()
program.dotProduct.set_scalar_arg_dtypes([np.int32, None, None, None, None])

start_time = time()

local_memory = cl.LocalMemory(np.dtype(np.float32).itemsize * 1024)

program.dotProduct(queue, (1024,), (1024/16,), widthA, buffer_a, buffer_b, buffer_c, local_memory)

queue.finish()

run_time = time()

## Move the kernel's output data to host memory.
#cl.enqueue_copy(queue, mat_c, buffer_c)
cl.enqueue_copy(queue, mat_c, buffer_c)

print "RESULT: {0}s".format(run_time-start_time)
