#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

//#include "/usr/local/cuda-8.0/targets/x86_64-linux/include/CL/cl.hpp"
#include "cl.hpp"
#include <iostream>
#include <fstream>
#include <string>

int main(){
    const int N_ELEMENTS = 1024;
    int *A = new int[N_ELEMENTS];
    int *B = new int[N_ELEMENTS];
    int *C = new int[N_ELEMENTS];

    for(int i=0; i<N_ELEMENTS; i++){
        A[i]=i;
        B[i]=i;
    }

    try{
        // Query for platforms
        cl::vector <cl::Platform>platforms;
        cl::Platform::get(&platforms);

        //Get a list of devices on this platform
        cl::vector<cl::Device>devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        //Create a context for the devices
        cl::Context context(devices);

        //Create a command queue for the first device
        cl::CommandQueue queue=
            cl::CommandQueue(context, devices[0]);

        //Create the memory buffers
        cl::Buffer bufferA = cl::Buffer(context, 
            CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));
        cl::Buffer bufferB = cl::Buffer(context, 
            CL_MEM_READ_ONLY, N_ELEMENTS * sizeof(int));
        cl::Buffer bufferC = cl::Buffer(context, 
            CL_MEM_WRITE_ONLY, N_ELEMENTS * sizeof(int));

        //Copy the input data to the input buffers using the
        //command queue for the first device
        queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0,
            N_ELEMENTS*sizeof(int), A);
        queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0,
            N_ELEMENTS*sizeof(int), B);

        //Read the program source
        std::ifstream sourceFile("vector_add_cernel.cl");
        std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1,
            std::make_pair(sourceCode.c_str(),
            sourceCode.length()+1));
        
        //Make program from the source code
        cl::Program program=cl::Program(context, source);

        //Build the program for the devices
        program.build(devices);

        //Make kernel
        cl::Kernel vecadd_kernel(program, "vecadd");

        //Set the kernel arguments
        vecadd_kernel.setArg(0, buferA);
        vecadd_kernel.setArg(1, buferB);
        vecadd_kernel.setArg(2, buferC);

        //Excute the kernel
        cl::NDRange global(N_ELEMENTS);
        cl::NDRange local(256);
        queue.enqueueNDRangeKernel(vecadd_kernel,
            cl::NullRange, global, local);

        //Copy the output data back to the host
        queue.enqueueReadBuffer(bufferC, CL_TRUE, 0
            N_ELEMENTS*sizeof(int), C);

        //Verify the result
        bool result=true;
        for (int i = 0; i<N_ELEMENTS;; i++){
            if(C[i] != A[i]+B[i]){
                result=false;
                break;
            }
        }

        if (result)
        std::cout < < “Success!” < < std::endl;
        else
        std::cout < < “Failed!” < < std::endl;
        }
        catch(cl::Error error)
        {
        std::cout < < error.what() < < “(” < <
        error.err() < < “)” < < std::endl;
        }
        return 0;
    }
}