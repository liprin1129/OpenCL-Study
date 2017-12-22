import numpy as np
import pyopencl as cl
import sys
from time import time

"""
# Version 1
c_dot_product_kernel = '''
__kernel void dotProduct(
    const int widthA,
    const int widthB,
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
"""

# Version 2
c_dot_product_kernel = '''
    // Tiled and coalesced version
    __kernel void dotProduct(const int M, const int N, const int K,
                          const __global float* A,
                          const __global float* B,
                          __global float* C)
        {
        const int TS = 32;
        // Thread identifiers
        const int row = get_local_id(0); // Local row ID (max: TS)
        const int col = get_local_id(1); // Local col ID (max: TS)
        const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
        const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

        // Local memory to fit a tile of TS*TS elements of A and B
        __local float Asub[32][32];
        __local float Bsub[32][32];
     
        // Initialise the accumulation register
        float acc = 0.0f;
        
        // Loop over all tiles
        const int numTiles = K/TS;
        for (int t=0; t<numTiles; t++) {
     
            // Load one tile of A and B into local memory
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[col][row] = A[tiledCol*M + globalRow];
            Bsub[col][row] = B[globalCol*K + tiledRow];
     
            // Synchronise to make sure the tile is loaded
            barrier(CLK_LOCAL_MEM_FENCE);
     
            // Perform the computation for a single tile
            for (int k=0; k<TS; k++) {
                acc += Asub[k][row] * Bsub[col][k];
            }
     
            // Synchronise before loading the next tile
            barrier(CLK_LOCAL_MEM_FENCE);
        }
     
        // Store the final result in C
        C[globalCol*M + globalRow] = acc;
    }
'''

mat_a = np.random.randn(2048* 2048)
mat_b = np.random.randn(2048* 2048)
mat_c = np.empty(2048* 2048)

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

M = 2048
M = np.int32(M)

N = 2048
N = np.int32(N)

K = 2048
K = np.int32(K)
program = cl.Program(context, c_dot_product_kernel).build()
program.dotProduct.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, None, None, None])
program.dotProduct(queue, mat_c.shape, None, M, N, K, buffer_a, buffer_b, buffer_c)

queue.finish()

run_time = time()

## Move the kernel's output data to host memory.
#cl.enqueue_copy(queue, mat_c, buffer_c)
cl.enqueue_copy(queue, mat_c, buffer_c)

print "RESULT: {0}s".format(run_time-start_time)