#define CL_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <CL/cl.h>
#include <mpi.h>
#include <time.h>

#define ARRAY_SIZE 100

using namespace std;

int main()
{
    // Initialize MPI
    MPI_Init(NULL, NULL);

    clock_t start_time = clock();

    // Get the number of processes and the rank of this process
    int numProcesses, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Set up OpenCL context and command queue
    cl_device_id deviceId;
    cl_context context;
    cl_command_queue commandQueue;
    cl_int ret;

    cl_platform_id platformId;
    ret = clGetPlatformIDs(1, &platformId, NULL);

    ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 1, &deviceId, NULL);
    context = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &ret);
    commandQueue = clCreateCommandQueueWithProperties(context, deviceId, NULL, &ret);

    // Create OpenCL program from source code
    const char *source = "_kernel void sort(_global int *array, int start, int end) \
                            { \
                                int pivot = array[end]; \
                                int i = start - 1; \
                                for (int j = start; j <= end - 1; j++) \
                                { \
                                    if (array[j] < pivot) \
                                    { \
                                        i++; \
                                        int temp = array[i]; \
                                        array[i] = array[j]; \
                                        array[j] = temp; \
                                    } \
                                } \
                                int temp = array[i + 1]; \
                                array[i + 1] = array[end]; \
                                array[end] = temp; \
                            }";


    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &ret);
    ret = clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL);

    // Create OpenCL kernel from program
    cl_kernel kernel = clCreateKernel(program, "sort", &ret);

    // Create OpenCL buffer for the input array and write data to it
    int inputArray[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        inputArray[i] = rand();
    }

    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * ARRAY_SIZE, NULL, &ret);
    ret = clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, inputArray, 0, NULL, NULL);

    // Set the kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer);
    int start = rank * (ARRAY_SIZE / numProcesses);
    int end = start + (ARRAY_SIZE / numProcesses) - 1;
    ret = clSetKernelArg(kernel, 1, sizeof(int), (void *)0);
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        inputArray[i] = rand() % 100;
    }

    // Copy the input array to the input buffer
    ret = clEnqueueWriteBuffer(commandQueue, inputBuffer, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, inputArray, 0, NULL, NULL);

    // Create the kernel and set its arguments
    cl_kernel kernel1 = clCreateKernel(program, "sort", &ret);
    ret = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&inputBuffer);
    ret = clSetKernelArg(kernel1, 1, sizeof(int), (void *)&start);
    ret = clSetKernelArg(kernel1, 2, sizeof(int), (void *)(ARRAY_SIZE - 1));

    // Define the global work size and enqueue the kernel
    size_t globalWorkSize[1] = {ARRAY_SIZE};
    ret = clEnqueueNDRangeKernel(commandQueue, kernel1, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

    // Create OpenCL buffer for the output array
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * ARRAY_SIZE, NULL, &ret);

    // Copy the result from the input buffer to the output buffer
    ret = clEnqueueCopyBuffer(commandQueue, inputBuffer, outputBuffer, 0, 0, sizeof(int) * ARRAY_SIZE, 0, NULL, NULL);

    // Create a host-side output array and read the result from the output buffer
    int outputArray[ARRAY_SIZE];
    ret = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, sizeof(int) * ARRAY_SIZE, outputArray, 0, NULL, NULL);
    clock_t end_time = clock();

    double duration = (double) (end_time - start_time) / ((double)CLOCKS_PER_SEC/1000000);

    

    // Print the sorted output array
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        cout << outputArray[i] << " ";
    }
    cout << endl;

    printf("time taken: %f microseconds", duration);

    // Release OpenCL resources
    ret = clFlush(commandQueue);
    ret = clFinish(commandQueue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseKernel(kernel1);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(inputBuffer);
    ret = clReleaseMemObject(outputBuffer);
    ret = clReleaseCommandQueue(commandQueue);
    ret = clReleaseContext(context);

    return 0;
}
