/*
    //
//  main.cpp
//  OpenCLWithC++
//
//  Created by SeongMuk Gang on 2018/04/14.
//  Copyright © 2018 SeongMuk Gang. All rights reserved.
//

#define __CL_ENABLE_EXCEPTIONS

//#include "../../Include/cl.hpp"
//#include "../../Include/util.hpp"
//#include "../../Include/err_code.h"

#include "cl.hpp"
#include "util.hpp"
#include "err_code.h"

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <iostream>
#include <fstream>

#include <Accelerate/Accelerate.h>

// pick up device type from compiler command line or from the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

int main()
{
    // Host memories
    std::vector<float> h_a(LENGTH);
    std::vector<float> h_b(LENGTH);
    std::vector<float> h_c(LENGTH, 0xdeadbeef);

    int count = LENGTH;
    for (int i=0; i<count; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Device memories
    cl::Buffer d_a;
    cl::Buffer d_b;
    cl::Buffer d_c;

    try {
        // Create a context
        cl::Context context(DEVICE);

        // Load in kernel source, creating a program object for the context
        cl::Program program(context, util::loadProgram("/Users/pure/Private_Local_Data/Study/OpenCL/OpenCL-Study/OpenCLWithC++/OpenCLWithC++/Kernel/vectorAdd.cl"), true);

        // Get the command queue
        cl::CommandQueue queue(context);

        // Create the kernel functor
        auto vadd = cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int>(program, "vadd");

        d_a = cl::Buffer(context, begin(h_a), end(h_a), true);
        d_b = cl::Buffer(context, begin(h_b), end(h_b), true);
        d_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * LENGTH);

        util::Timer timer;

        vadd(
             cl::EnqueueArgs(queue, cl::NDRange(count)),
             d_a,
             d_b,
             d_c,
             count);

        queue.finish();

        double rtime = static_cast<double>(timer.getTimeMicroseconds());
        printf("\nThe kernels ran in %lf milli seconds\n", rtime);

        cl::copy(queue, d_c, begin(h_c), end(h_c));

        // Test the results
        int correct = 0;
        float tmp;
        for(int i=0; i<count; i++){
            tmp = h_a[i] + h_b[i];
            tmp -= h_c[i];
            if(tmp*tmp < TOL*TOL){
                correct++;
            }
            else {
                printf(" tmp %f h_a %f h_b %f h_c %f \n",
                       tmp,
                       h_a[i],
                       h_b[i],
                       h_c[i]);
            }
        }

        // summarize results
        printf("vector add to find C = A+B: %d out of %d results were correct.\n",
               correct,
               count);
    } catch (cl::Error err) {
        std::cout << "Exception\n";
        std::cerr
            << "ERROR: "
            << err.what()
            << "("
            << err_code(err.err())
            << ")"
            << std::endl;
    }
}
*/

#include <sys/types.h>
#include <stdio.h>

/* Include the clBLAS header. It includes the appropriate OpenCL headers */
#include <clBLAS.h>
#include <Accelerate/Accelerate.h>
#include <OpenCL/OpenCL.h>
/* This example uses predefined matrices and their characteristics for
 * simplicity purpose.
 */

#define M  4
#define N  3
#define K  5

static const cl_float alpha = 10;

static const cl_float A[M*K] = {
    11, 12, 13, 14, 15,
    21, 22, 23, 24, 25,
    31, 32, 33, 34, 35,
    41, 42, 43, 44, 45,
};
static const size_t lda = K;        /* i.e. lda = K */

static const cl_float B[K*N] = {
    11, 12, 13,
    21, 22, 23,
    31, 32, 33,
    41, 42, 43,
    51, 52, 53,
};
static const size_t ldb = N;        /* i.e. ldb = N */

static const cl_float beta = 20;

static cl_float C[M*N] = {
    11, 12, 13,
    21, 22, 23,
    31, 32, 33,
    41, 42, 43,
};
static const size_t ldc = N;        /* i.e. ldc = N */

static cl_float result[M*N];

int main( void )
{
    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    int ret = 0;

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs( 1, &platform, NULL );
    err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
    queue = clCreateCommandQueue( ctx, device, 0, &err );

    /* Setup clBLAS */
    err = clblasSetup( );

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer( ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A),
                          NULL, &err );
    bufB = clCreateBuffer( ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B),
                          NULL, &err );
    bufC = clCreateBuffer( ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
                          NULL, &err );

    err = clEnqueueWriteBuffer( queue, bufA, CL_TRUE, 0,
                               M * K * sizeof( *A ), A, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queue, bufB, CL_TRUE, 0,
                               K * N * sizeof( *B ), B, 0, NULL, NULL );
    err = clEnqueueWriteBuffer( queue, bufC, CL_TRUE, 0,
                               M * N * sizeof( *C ), C, 0, NULL, NULL );

    /* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
    err = clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
                      M, N, K,
                      alpha, bufA, 0, lda,
                      bufB, 0, ldb, beta,
                      bufC, 0, ldc,
                      1, &queue, 0, NULL, &event );

    /* Wait for calculations to be finished. */
    err = clWaitForEvents( 1, &event );

    /* Fetch results of calculations from GPU memory. */
    err = clEnqueueReadBuffer( queue, bufC, CL_TRUE, 0,
                              M * N * sizeof(*result),
                              result, 0, NULL, NULL );

    /* Release OpenCL memory objects. */
    clReleaseMemObject( bufC );
    clReleaseMemObject( bufB );
    clReleaseMemObject( bufA );

    /* Finalize work with clBLAS */
    clblasTeardown( );

    /* Release OpenCL working objects. */
    clReleaseCommandQueue( queue );
    clReleaseContext( ctx );

    return ret;
}
