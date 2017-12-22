import numpy as np
import pyopencl as cl
import sys
from time import time

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
__kernel void dotProduct(
    const int widthA,
    const int widthB,
    __global float* inputA,
    __global float* inputB,
    __global float* outputC) {
        const int WI_SIZE = 16;

        // Work Group Index
        int wgX = get_global_id(0);
        int wgY = get_global_id(1);

        // Work Item Index
        int wiX = get_local_id(0);
        int wiY = get_local_id(1);

        // INDEX OF THE FIRST SUB-MATRIX OF A PROCESSED
        // BY THE WORKING ITEM
        int aBegin = widthA * WI_SIZE * wgY;

        // INDEX OF THE LAST SUB-MATRIX OF A PROCESSED
        // BY THE WORKING ITEM
        int aEnd = aBegin + widthA - 1;

        // STEP SIZE USED TO ITERATE THROUGH THE
        // SUB-MATRICES OF A
        int aStep = WI_SIZE;

        // INDEX OF THE FIRST SUB-MATRIX OF B PROCESSED
        // BY THE BLOCK
        int bBegin = WI_SIZE * wgX;

        // STEP SIZE USED TO ITERATE THROUGH THE
        // SUB-MATRICES OF B
        int bStep = WI_SIZE * widthB;

        float subC=0.0f; // for outputC's sub-matrices
        
        // LOOP OVER ALL THE SUB-MATRICES OF A AND B
        // REQUIRED TO COMPUTE THE WORKI ITEM SUB-MATRIX
        for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep)
        {
            // DECLARATION OF THE LOCAL MEMORY ARRAY NAMED As
            // USED TO STORE THE SUB-MATRIX OF A
            __local float As[16][16];

            // DECLARATION OF THE LOCAL MEMORY ARRAY NAMED Bs
            // USED TO STORE THE SUB-MATRIX OF b
            __local float Bs[16][16];

            // LOAD THE MATRICES FROM GLOBAL MEMORY
            // TO LOCAL MEMORY; EACH WORKING ITEM LOADS
            // ONE ELEMENT OF EACH MATRIX
            As[wiY][wiX] = inputA[a + widthA * wiY + wiX];
            Bs[wiY][wiX] = inputB[b + widthB * wiY + wiX];

            // SYNCHRONIZE TO MAKE SURE THE MATRICES ARE LOADED
            barrier(CLK_LOCAL_MEM_FENCE);

            // MUTIPLY THE TWO MATRICES TOGETHER; EACH WORK ITEM COMPUTES
            // ONE ELEMENT OF THE WORK GROUP SUB-MATRIX

            for (int k = 0; k < WI_SIZE; ++k){
                subC += As[wiY][k] * Bs[k][wiX];
            }

            // SYCHRONIZE TO MAKE SURE THAT THE PRECEDING COMPUTATION IS
            // DONE BEFORE LOADING TWO NEW SUB-MATRICES OF A AND B IN THE
            // NEXT ITERATION
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // WRITE THE BLOCK SUB-MATRIX TO DEVICE MEMORY; EACH WORK ITEM WRITES
        // ONE ELEMENT
        int c = widthB * WI_SIZE * wgY + WI_SIZE * wgX;
        outputC[c + widthB * wiY + wiX] = subC;
    }
'''
"""

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

widthA = 2048
widthA = np.int32(widthA)

widthB = 2048
widthB = np.int32(widthB)

program = cl.Program(context, c_dot_product_kernel).build()
program.dotProduct.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])
program.dotProduct(queue, mat_c.shape, None, widthA, widthB, buffer_a, buffer_b, buffer_c)

queue.finish()

run_time = time()

## Move the kernel's output data to host memory.
#cl.enqueue_copy(queue, mat_c, buffer_c)
cl.enqueue_copy(queue, mat_c, buffer_c)

print "RESULT: {0}s".format(run_time-start_time)