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
	std::same_as<T, short> ||
	std::same_as<T, uint4> ||
	std::same_as<T, long>;

template <LatencyLoadType T> struct ptx_load_suffix; 

template <> struct ptx_load_suffix<short>   { static constexpr const char value[] = ".u16"; };
template <> struct ptx_load_suffix<uint>   { static constexpr const char value[] = ".u32"; };
template <> struct ptx_load_suffix<long>   { static constexpr const char value[] = ".s64"; };
template <> struct ptx_load_suffix<uint4>  { static constexpr const char value[] = ".v4.i32";  };


/*
	Each thread in a grid will read one element

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

// A single thread will read the whole buffer
template <LatencyLoadType T>
__global__ void stride_timed_reads(T *src, long *results, long *tid, size_t size) {
	uint idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint threads = blockDim.x * gridDim.x;
	uint thread_cap = size / sizeof(T);
	long t0, t1;
	T local;
	for (int i = idx; i < thread_cap; i += threads) {
		T *src_el = src + i;
		long *result_el = results + i;
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

template <LatencyLoadType T>
__global__ void single_timed_reads(T *src, long *results, size_t size, size_t stride) {
	uint idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx != 0) return;
	uint thread_cap = size / stride;
	long t0, t1;
	T local;
	for (; idx < thread_cap; idx++) {
		T *src_el = src + idx * (stride / sizeof(T));
		long *result_el = results + idx;
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

enum Mode {
	SINGLE_READS, SERIAL_READS, INTER_READS
};

const char *mode_to_str(Mode mode) {
	if (mode == SINGLE_READS)
		return "timed_reads";
	else if (mode == SERIAL_READS)
		return "single_timed_reads";
	else if (mode == INTER_READS)
		return "interleaved_reads";
	return "";
}

Mode str_to_mode(std::string mode) {
	if (mode == "single") return SINGLE_READS;
	else if (mode == "serial") return SERIAL_READS;
	else if (mode == "inter") return INTER_READS;
}

template <LatencyLoadType T>
void run_latency(size_t size, int num_iters, size_t stride, Mode mode, size_t alignment, int start, int end, int src_gpu, int reader_gpu) {
	std::cout << "Running latency test " << " with size " << size << " bytes " 
			  << "Mode " << mode_to_str(mode) << std::endl;


	std::vector<long> results(size / stride);
	std::vector<long> tids(size / stride);
	long *d_results, *d_tid;

	CHECK_CUDA_ERROR(cudaSetDevice(src_gpu));
	AlignedBuffer buffer(size, alignment);
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	CHECK_CUDA_ERROR(cudaSetDevice(reader_gpu));
	CHECK_CUDA_ERROR(cudaMalloc(&d_results, sizeof(long) * size / stride));
	CHECK_CUDA_ERROR(cudaMalloc(&d_tid, sizeof(long) * size / stride));
	CHECK_CUDA_ERROR(cudaMemset(d_results, 0, sizeof(long) * size / stride));

    cudaStream_t stream;
	CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	assert(stride % sizeof(T) == 0);
	size_t block_size = std::min((size_t)threadsPerBlock, size / stride);
	for (int i = 0; i < num_iters; i++) {
		dim3 block(block_size, 1, 1);
		dim3 grid(size / (block.x * stride), 1, 1);
		if (mode == SINGLE_READS)
			timed_reads<T><<<grid, block, 0, stream>>>
			((T *)buffer.aligned_ptr, d_results, size, stride);
		else if (mode == SERIAL_READS)
			single_timed_reads<T><<<1, 1, 0, stream>>>
			((T *)buffer.aligned_ptr, d_results, size, stride);
		else if (mode == INTER_READS) {
			stride_timed_reads<T><<<1, block_size, 0, stream>>>
			((T *)buffer.aligned_ptr, d_results, d_tid, size);
		}
		CHECK_CUDA_ERROR(cudaMemcpyAsync(results.data(), d_results, (size / stride) * sizeof(long), cudaMemcpyDeviceToHost, stream));
		CHECK_CUDA_ERROR(cudaMemcpyAsync(tids.data(), d_tid, (size / stride) * sizeof(long), cudaMemcpyDeviceToHost, stream));
	}
	CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

	if (end != 0) {
		for (int i = start ; i < std::min(end, (int)results.size()); i++) 
			std::cout << i << ": " << tids[i] << ": " << results[i] << std::endl;
	} else {
		for (int i = 0 ; i < std::min((size_t)256, results.size()); i++) {
			std::cout << i << ": " << tids[i] << ": " << results[i] << std::endl;
		}
		if (mode == SINGLE_READS) {
			std::cout << "first 32" << std::endl;
			for (int i = 0 ; i < std::min((size_t)32, results.size()); i++) {
				std::cout << i << ": " << tids[i] << ": " << results[i] << std::endl;
			}
			std::cout << "next iter" << std::endl;
			for (int i = block_size * (size / stride) ; i < std::min(block_size * (size / stride) + 128, results.size()); i++) {
				std::cout << i << ": " << tids[i] << ": " << results[i] << std::endl;
			}
		}
		for (int i = results.size() / 2 ; i < std::min(results.size() / 2 + 1000, results.size()); i++) {
			std::cout << i << ": " << tids[i] << ": " << results[i] << std::endl;
		}
	}
	//print_stats(std::span(results), "cycles");
}

int main(int argc, char* argv[]) {
    size_t size = 1ULL << 30;
    int num_iters = 1;
	size_t stride = sizeof(long);
	size_t alignment = 1;
	Mode mode;
	int start = 0, end = 0;
	int reader_gpu = 0, src_gpu = 1;

    static struct option long_options[] = {
        {"size", required_argument, 0, 's'},
        {"iters", required_argument, 0, 'i'},
		{"stride", required_argument, 0, 't'},
		{"align", required_argument, 0, 'a'},
		{"mode", required_argument, 0, 'm'},
		{"left", required_argument, 0, 'l'},
		{"right", required_argument, 0, 'r'},
		{"from", required_argument, 0, 'f'},
		{"dest", required_argument, 0, 'd'},
		//{"dtype", required_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's': size = parse_size(optarg); break;
            case 'i': num_iters = stoi(optarg); break;
            case 'l': start = stoi(optarg); break;
            case 'r': end = stoi(optarg); break;
			//case 'd': dtype = std::string(optarg); break;
			case 't': stride = parse_size(optarg); break;
			case 'a': alignment = parse_size(optarg); break;
			case 'm': mode = str_to_mode(optarg); break;
			case 'f': src_gpu = stoi(optarg); break;
			case 'd': reader_gpu = stoi(optarg); break;
        }
    }

    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
	assert(device_count >= 2 && "At least 2 GPUs are required for this benchmark.\n");

	//print_gpu_info(device_count);
	enable_p2p(device_count);
	if (mode == INTER_READS) stride = sizeof(uint);


	run_latency<long>(size, num_iters, stride, mode, alignment, start, end, src_gpu, reader_gpu);

    return 0;
}
