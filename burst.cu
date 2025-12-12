#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <getopt.h>
#include <thread>
#include <numeric>
#include <cassert>
#include <concepts>
#include <memory>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda_helpers.h"
#include "errors.h"
#include "cli.h"
#include "data.h"

using namespace std;

__global__ void burst_write(void *buffer, size_t size, int stride) {
    size_t from = blockDim.x * blockIdx.x + threadIdx.x;
	long *buf = static_cast<long *>(buffer);
	int el_count = size / sizeof(long);
	int stride_skip = stride / sizeof(long);
    for (int i = 0; i < el_count; i += stride_skip) {
		buf[i] = static_cast<long>(from);
    }
}

void start_burst(int size, int num_iters, int warmup, int gpus, int target, int sm_count, int block_size, int stride) {
	std::vector<std::unique_ptr<l2flush>> l2_flushers;
	std::vector<cudaStream_t> streams(gpus, {});
	for (int i = 0; i < gpus; i++) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
		l2_flushers.push_back(std::make_unique<l2flush>());
		CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
	}

	CHECK_CUDA_ERROR(cudaSetDevice(target));
	AlignedBuffer buffer(size, 1 << 30);

    for (int iter = 0; iter < warmup; ++iter) {
		for (int i = 0; i < gpus; i++) {
			cudaSetDevice(i);
			l2_flushers[i]->flush_sync(0);
		}

		for (int i = 0; i < gpus; ++i) {
			CHECK_CUDA_ERROR(cudaSetDevice(i));
			burst_write<<<sm_count, block_size, 0, streams[i]>>>((void *)buffer.aligned_ptr, (size_t)size, stride);
		}

        for (int i = 0; i < gpus; ++i) {
            CHECK_CUDA_ERROR(cudaSetDevice(i));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    for (int iter = 0; iter < num_iters; ++iter) {
		for (int i = 0; i < gpus; i++) {
			cudaSetDevice(i);
			l2_flushers[i]->flush_sync(0);
		}

		for (int i = 0; i < gpus; ++i) {
			CHECK_CUDA_ERROR(cudaSetDevice(i));
			burst_write<<<sm_count, block_size, 0, streams[i]>>>((void *)buffer.aligned_ptr, (size_t)size, stride);
		}

        for (int i = 0; i < gpus; ++i) {
            CHECK_CUDA_ERROR(cudaSetDevice(i));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    CHECK_CUDA_ERROR(cudaSetDevice(target));
}

int main(int argc, char* argv[]) {
    size_t size = 8;
    int num_iters = 1;
    int warmup_iters = 0;
    int num_gpus = 3;
    int target = 1;
    int sm_count = get_sm_count(0);
	int block_size = 512;
	int stride = 8;
	int count = -1;

    static struct option long_options[] = {
        {"size", required_argument, 0, 's'},
        {"iters", required_argument, 0, 'i'},
        {"warmup", required_argument, 0, 'w'},
        {"gpus", required_argument, 0, 'g'},
        {"target", required_argument, 0, 't'},
		{"count-sm", required_argument, 0, 'c'},
		{"block-size", required_argument, 0, 'b'},
		{"stride", required_argument, 0, 'r'},
		{"count", required_argument, 0, 'n'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
        switch (opt) {
            case 's': size = parse_size(optarg); break;
            case 'i': num_iters = stoi(optarg); break;
            case 'w': warmup_iters = stoi(optarg); break;
            case 'g': num_gpus = stoi(optarg); break;
            case 't': target = stoi(optarg); break;
			case 'c': sm_count = stoi(optarg); break;
			case 'b': block_size = stoi(optarg); break;
			case 'r': stride = stoi(optarg); break;
			case 'n': count = stoi(optarg); break;
			default:
				cerr << "Usage: " << argv[0] << " [--size SIZE] [--iters ITERS] [--warmup WARMUP] [--gpus GPUS] [--target TARGET] [--count-sm COUNT_SM] [--block-size BLOCK_SIZE] [--stride STRIDE] [--count COUNT]" << endl;
				return 1;
        }
    }

	if (count != -1) {
		size = count * stride;
	}

    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (num_gpus > device_count) {
        cerr << "Requested " << num_gpus << " GPUs, but only " << device_count << " available." << endl;
        return 0;
    }

	print_gpu_info(num_gpus);
    enable_p2p(num_gpus);

	assert(size >= sizeof(long) && size % sizeof(long) == 0);
	start_burst(size, num_iters, warmup_iters, num_gpus, target, sm_count, block_size, stride);
    return 0;
}
