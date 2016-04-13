#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;

#define ARRAY_LENGTH 100000000
//#define GPU
#define CPU

#include "my_cl.h"

int main() {
	cl_int err;
	cl_event event;

	// get first platform
	cl_platform_id platform;
	err = clGetPlatformIDs(1, &platform, NULL);
	cout << "1st Platfomr: " << GetPlatformName(platform) << endl;

	//get CPU device count
	cl_uint cpuDeviceCount;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &cpuDeviceCount);
	
	//get all CPU devices
	cl_device_id* cpuDevices;
	cpuDevices = new cl_device_id[cpuDeviceCount];
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, cpuDeviceCount, cpuDevices, NULL);

	for (cl_uint i = 0; i < cpuDeviceCount; ++i) {
		cout << "\t CPU (" << (i + 1) << ") : " << GetDeviceName (cpuDevices[i]) << endl;
	}

	// for each cpu device create a separate context AND queue
	cl_context* cpuContexts = new cl_context[cpuDeviceCount];
	cl_command_queue* cpuQueues = new cl_command_queue[cpuDeviceCount];
	for (int i = 0; i < cpuDeviceCount; i++) {
		cpuContexts[i] = clCreateContext(NULL, cpuDeviceCount, cpuDevices, NULL, NULL, &err);
		cpuQueues[i] = clCreateCommandQueue(cpuContexts[i], cpuDevices[i], CL_QUEUE_PROFILING_ENABLE, &err);
	}
	
	//get GPU device count
	cl_uint gpuDeviceCount;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &gpuDeviceCount);
	
	//get all GPU devices
	cl_device_id* gpuDevices;
	gpuDevices = new cl_device_id[gpuDeviceCount];
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, gpuDeviceCount, gpuDevices, NULL);

	for (cl_uint i = 0; i < gpuDeviceCount; ++i) {
		cout << "\t GPU (" << (i + 1) << ") : " << GetDeviceName (gpuDevices[i]) << endl;
	}

	// for each Gpu device create a separate context AND queue
	cl_context* gpuContexts = new cl_context[gpuDeviceCount];
	cl_command_queue* gpuQueues = new cl_command_queue[gpuDeviceCount];
	for (int i = 0; i < gpuDeviceCount; i++) {
		gpuContexts[i] = clCreateContext(NULL, gpuDeviceCount, gpuDevices, NULL, NULL, &err);
		gpuQueues[i] = clCreateCommandQueue(gpuContexts[i], gpuDevices[i], CL_QUEUE_PROFILING_ENABLE, &err);
	}

	/*
	 * Here you have one context and one command queue per device.
	 * You can choose to send your tasks to any of these queues.
	 */
	
	const float cons = 0.1;
#ifdef GPU	
	cout << "Use GPU ..." << endl;
	cl_program gpuProgram = CreateProgram(gpuContexts[0], gpuDevices[0], "kernel.cl");
	cl_kernel gpuKernel = clCreateKernel(gpuProgram, "compute" ,&err);
	err_check(err, "gpuKernel");	

	vector<float> h_A(ARRAY_LENGTH, cons);
	cout << "Buffer size: " << (float) ARRAY_LENGTH / (1024 * 1024) << " MB" << endl;
	float b = cons;
	float c = cons;
	float correctRes = cons * b + c;
	cl_mem d_A = clCreateBuffer(gpuContexts[0], CL_MEM_COPY_HOST_PTR,
			sizeof(float) * h_A.size(), &h_A[0], &err);
	err_check(err, "clCreateBuffer d_A");

	// set kernel arguments
	err = clSetKernelArg(gpuKernel, 0, sizeof(cl_mem), &d_A);
	err |= clSetKernelArg(gpuKernel, 1, sizeof(float), &b);
	err |= clSetKernelArg(gpuKernel, 2, sizeof(float), &c);
	err_check(err, "set Kernel Arg");

	clFinish(gpuQueues[0]);

	// Launch kernel
	size_t globalWorkSize[] = {h_A.size()};
	err = clEnqueueNDRangeKernel(gpuQueues[0], gpuKernel, 1, NULL, globalWorkSize, NULL,
			0, NULL, &event);
	err_check(err, "clEnqueueNDRangeKernel");

	clWaitForEvents(1, &event);
	
	// copy back data from GPU
	err = clEnqueueReadBuffer(gpuQueues[0], d_A, CL_TRUE, 0, sizeof(float) * h_A.size(), &h_A[0],
			0, NULL, NULL);
	err_check(err, "copy back ");
#endif

#ifdef CPU
	cout << "Use CPU ..." << endl;
	cl_program cpuProgram = CreateProgram(cpuContexts[0], cpuDevices[0], "kernel.cl");
	cl_kernel cpuKernel = clCreateKernel(cpuProgram, "compute" ,&err);
	err_check(err, "cpuKernel");	

	vector<float> h_A(ARRAY_LENGTH, cons);
	cout << "Buffer size: " << (float) ARRAY_LENGTH / (1024 * 1024) << " MB" << endl;
	float b = cons;
	float c = cons;
	float correctRes = cons * b + c;
	cl_mem d_A = clCreateBuffer(cpuContexts[0], CL_MEM_COPY_HOST_PTR,
			sizeof(float) * h_A.size(), &h_A[0], &err);
	err_check(err, "clCreateBuffer d_A");

	// set kernel arguments
	err = clSetKernelArg(cpuKernel, 0, sizeof(cl_mem), &d_A);
	err |= clSetKernelArg(cpuKernel, 1, sizeof(float), &b);
	err |= clSetKernelArg(cpuKernel, 2, sizeof(float), &c);
	err_check(err, "set Kernel Arg");

	clFinish(gpuQueues[0]);
	
	// Launch kernel
	size_t globalWorkSize[] = {h_A.size()};
	err = clEnqueueNDRangeKernel(cpuQueues[0], cpuKernel, 1, NULL, globalWorkSize, NULL,
			0, NULL, &event);
	err_check(err, "clEnqueueNDRangeKernel");

	clWaitForEvents(1, &event);
	
	// copy back data from device
	err = clEnqueueReadBuffer(cpuQueues[0], d_A, CL_TRUE, 0, sizeof(float) * h_A.size(), &h_A[0],
			0, NULL, NULL);
	err_check(err, "copy back ");
#endif

	// get the profiling data
	cl_ulong timeStart, timeEnd;
	double totalTime;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);
	totalTime = timeEnd - timeStart;
	cout << "\nKernel Execution time in milliseconds = " << totalTime/1000000.0 << " ms\n" << endl;

	for (int i = 0; i < (int )h_A.size(); ++i) {
		if (h_A[i] - correctRes > 0.1) {
			cout << "Result Not Correct: h_A[" << i << "] = " << h_A[i] << endl;
			return 0;
		}
	}
	cout << "Result Success" << endl;

	// cleanup CPU
	for(int i = 0; i < cpuDeviceCount; i++) {
		clReleaseDevice(cpuDevices[i]);
		clReleaseContext(cpuContexts[i]);
		clReleaseCommandQueue(cpuQueues[i]);
	}

	delete[] cpuDevices;
	delete[] cpuContexts;
	delete[] cpuQueues;

	// cleanup GPU
	for(int i = 0; i < gpuDeviceCount; i++) {
		clReleaseDevice(gpuDevices[i]);
		clReleaseContext(gpuContexts[i]);
		clReleaseCommandQueue(gpuQueues[i]);
	}

	delete[] gpuDevices;
	delete[] gpuContexts;
	delete[] gpuQueues;

	return 0;
}
