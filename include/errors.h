#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

/* taken from: https://leimao.github.io/blog/Proper-CUDA-Error-Checking/ */
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CU_ERROR(val) checkDrv((val), #val, __FILE__, __LINE__)
void checkDrv(CUresult err, const char* const func, const char* const file,
           const int line)
{
	const char *err_str;
    if (err != 0)
    {
		cuGetErrorString(err, &err_str);
        std::cerr << "CUDA Driver API Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << err_str << " " << func << std::endl;
    }
}
