#include <string>
#include <chrono>
#include <iomanip>
#include <getopt.h>
#include <thread>
#include <numeric>
#include <cassert>
#include <concepts>
#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <cuda.h>

#include "cli.h"
#include "errors.h"
#include "cuda_helpers.h"
#include "data.h"

using namespace std;

constexpr int threadsPerBlock = 512;
size_t warmup_size = 8 << 20;

template <typename T>
concept LatencyLoadType =
	std::same_as<T, uint> ||
	std::same_as<T, char> ||
	std::same_as<T, uint4> ||
	std::same_as<T, long>;

template <LatencyLoadType T> struct ptx_load_suffix; 

template <> struct ptx_load_suffix<char>   { static constexpr const char value[] = ".s8"; };
template <> struct ptx_load_suffix<uint>   { static constexpr const char value[] = ".u32"; };
template <> struct ptx_load_suffix<long>   { static constexpr const char value[] = ".s64"; };
template <> struct ptx_load_suffix<uint4>  { static constexpr const char value[] = ".v4.i32";  };


/*
	read from peer memory and check the latency

	dest: device accessible buffer
	results: where each thread will write read latency
	count: number of elements in the dest bufferfence
	stride: number of bytes between each read
 */
template <LatencyLoadType T>
__global__ void timed_reads(T *src, long *results, size_t size, size_t stride) {
	uint idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint thread_cap = size / stride;
	T *src_el = src + idx * (stride / sizeof(T));
	long *result_el = results + idx;
	long t0, t1;
	T local;
	if (idx < thread_cap) {
		asm volatile (
			"mov.u64  %0, %%clock64;\n\t"
			"ld.global%3 %1, [%2];\n\t"
			: "=l"(t0),
			  "=l"(local)
			: "l" (src_el),
			  "C" (ptx_load_suffix<T>::value)
		);
		*result_el = local - t0; // we need a RAW dependency
								 // in order to trigger the load to change the global state
		t1 = clock64();
		*result_el += t1;
	}
}

/*
 	kernel that calculates the latency between two subsequent reads from the same register
 */
__global__ void clock_readings(long *results, int tries) {
	for (int i = 0; i < tries; i ++) {
		long t0 = clock64();
		long t1 = clock64();
		results[i] = t1 - t0;
	}
}

[[maybe_unused]] void get_clock_latency(int tries) {
	std::vector<long> results(tries);
	long *d_results;

	CHECK_CUDA_ERROR(cudaMalloc(&d_results, tries * sizeof(long)));
	clock_readings<<<1, 1, 0>>>(d_results, tries);
	CHECK_CUDA_ERROR(cudaMemcpy(results.data(), d_results, tries * sizeof(long), cudaMemcpyDeviceToHost));

	print_stats(std::span(results), "cycles");

	CHECK_CUDA_ERROR(cudaFree(d_results));
}

template <LatencyLoadType T>
void run_latency(size_t size, int num_iters, size_t stride) {
	char *buffer;

	std::cout << "Running latency test with size " << size << " bytes" << std::endl;

	constexpr int reader_gpu = 1;
	constexpr int source_gpu = 0;

	std::vector<long> results(size / stride);
	long *d_results;

	CHECK_CUDA_ERROR(cudaSetDevice(source_gpu));
	CHECK_CUDA_ERROR(cudaMalloc(&buffer, size));
	CHECK_CUDA_ERROR(cudaMemset(buffer, 0, size));
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaSetDevice(reader_gpu));
	CHECK_CUDA_ERROR(cudaMalloc(&d_results, sizeof(long) * size / stride));
	CHECK_CUDA_ERROR(cudaMemset(d_results, 0, sizeof(long) * size / stride));

    cudaStream_t stream;
	CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	assert(stride % sizeof(T) == 0);
	size_t block_size = std::min((size_t)threadsPerBlock, size / stride);
	for (int i = 0; i < num_iters; i++) {
		dim3 block(block_size, 1, 1);
		dim3 grid(size / (block.x * stride), 1, 1);
		timed_reads<T><<<grid, block, 0, stream>>>
			((T *)buffer, d_results, size, stride);
		CHECK_CUDA_ERROR(cudaMemcpyAsync(results.data(), d_results, size * sizeof(long) / stride, cudaMemcpyDeviceToHost, stream));
	}
	CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
	print_stats(std::span(results), "cycles");
}

int main(int argc, char* argv[]) {
    size_t size = 16ULL << 10;
    int num_iters = 10;
	//std::string dtype = "uint";
	size_t stride = sizeof(long);

    static struct option long_options[] = {
        {"size", required_argument, 0, 's'},
        {"iters", required_argument, 0, 'i'},
		{"stride", required_argument, 0, 't'},
		//{"dtype", required_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's': size = parse_size(optarg); break;
            case 'i': num_iters = stoi(optarg); break;
			//case 'd': dtype = std::string(optarg); break;
			case 't': stride = parse_size(optarg); break;
        }
    }

    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
	assert(device_count >= 2 && "At least 2 GPUs are required for this benchmark.\n");

	print_gpu_info(device_count);
	enable_p2p(device_count);

	run_latency<long>(size, num_iters, stride);

    return 0;
}
