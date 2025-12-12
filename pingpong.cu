#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <getopt.h>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>

#include "cli.h"
#include "errors.h"
#include "cuda_helpers.h"
#include "data.h"

using namespace std;

constexpr int GPU_A = 0;
constexpr int GPU_B = 1;
constexpr int DEFAULT_ITERS = 10000;
constexpr int BARRIER_VAL = 0xFAFA;

__device__ __forceinline__ void ptx_membar_gl() {
    asm volatile ("membar.gl;" ::: "memory");
}

__global__ void ping_kernel(
    volatile uint32_t* remote_flag,
    volatile uint32_t* local_flag,
    uint64_t* results,
    int iters,
    uint32_t start_val,
	int delay
) {

    for (int i = 0; i < iters; ++i) {
        uint32_t ping = start_val + i;
        uint32_t pong = ping ^ 0xFFFFFFFF;

		long delay_start = clock64();
		while (clock64() - delay_start < delay)
			;

        // t0: send ping
        results[2*i] = clock64();
        *remote_flag = ping;
        ptx_membar_gl();

        uint32_t val;
        do {
            val = *local_flag;
        } while (val != pong);

        // t3: received pong
        results[2*i + 1] = clock64();
    }
}

__global__ void pong_kernel(
    volatile uint32_t* local_flag,
    volatile uint32_t* remote_flag,
    uint64_t* results,
    int iters,
    uint32_t start_val
) {

    for (int i = 0; i < iters; ++i) {
        uint32_t ping = start_val + i;
        uint32_t pong = ping ^ 0xFFFFFFFF;

        // t1: wait for ping
        uint32_t val;
        do {
            val = *local_flag;
        } while (val != ping);
        results[2*i] = clock64();  // t1

        *remote_flag = pong;
        ptx_membar_gl();
        // t2: send pong
        results[2*i + 1] = clock64(); // t4
    }
}

void run_nvlink_latency(int num_iters, int start, int end, int delay) {
    int total_iters = num_iters;
    uint32_t start_val = 0xC001C0DE;

    uint32_t *d_flag_a, *d_flag_b;
    uint64_t *d_results_a, *d_results_b;
    uint64_t *h_results_a, *h_results_b;
	uint32_t barrier;

	/*
    int clock_khz;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, GPU_A));
    double cycles_to_ns = 1000000.0 / clock_khz;
	*/

    CHECK_CUDA_ERROR(cudaSetDevice(GPU_A));
	l2flush flushl2_a{};
    CHECK_CUDA_ERROR(cudaMalloc(&d_flag_a, sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_flag_a, 0, sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaSetDevice(GPU_B));
	l2flush flushl2_b{};
    CHECK_CUDA_ERROR(cudaMalloc(&d_flag_b, sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_flag_b, 0, sizeof(uint32_t)));

    size_t result_bytes = total_iters * 2 * sizeof(uint64_t);
    h_results_a = new uint64_t[total_iters * 2];
    h_results_b = new uint64_t[total_iters * 2];

    CHECK_CUDA_ERROR(cudaSetDevice(GPU_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results_a, result_bytes));
    CHECK_CUDA_ERROR(cudaMemset(d_results_a, 0, result_bytes));

    CHECK_CUDA_ERROR(cudaSetDevice(GPU_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_results_b, result_bytes));
    CHECK_CUDA_ERROR(cudaMemset(d_results_b, 0, result_bytes));

    cudaStream_t stream_a, stream_b;
    CHECK_CUDA_ERROR(cudaSetDevice(GPU_A));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_a));
    CHECK_CUDA_ERROR(cudaSetDevice(GPU_B));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream_b));

    std::cout << "NVLink Ping pong latency: " << num_iters << " samples, "
              << std::endl; // clock=" << clock_khz << " kHz" << std::endl;

	printf("Ping address: 0x%p\n", d_results_a);
	printf("Pong address: 0x%p\n", d_results_b);

    CHECK_CUDA_ERROR(cudaSetDevice(GPU_B));
	flushl2_b.flush(stream_b);
    pong_kernel<<<1, 1, 0, stream_b>>>(d_flag_b, d_flag_a, d_results_b, total_iters, start_val);
    CHECK_CUDA_ERROR(cudaSetDevice(GPU_A));
	flushl2_a.flush(stream_a);
    ping_kernel<<<1, 1, 0, stream_a>>>(d_flag_b, d_flag_a, d_results_a, total_iters, start_val, delay);

	/*
	barrier = BARRIER_VAL;
    CHECK_CUDA_ERROR(cudaSetDevice(GPU_B));
	CHECK_CUDA_ERROR(cudaMemcpy(barrier_d2, &barrier, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaSetDevice(GPU_A));
	CHECK_CUDA_ERROR(cudaMemcpy(barrier_d1, &barrier, sizeof(uint32_t), cudaMemcpyHostToDevice));
	*/
    CHECK_CUDA_ERROR(cudaSetDevice(GPU_A)); CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_a));
    CHECK_CUDA_ERROR(cudaSetDevice(GPU_B)); CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_b));

    CHECK_CUDA_ERROR(cudaSetDevice(GPU_A));
    CHECK_CUDA_ERROR(cudaMemcpy(h_results_a, d_results_a, result_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaSetDevice(GPU_B));
    CHECK_CUDA_ERROR(cudaMemcpy(h_results_b, d_results_b, result_bytes, cudaMemcpyDeviceToHost));

    std::vector<long> rtts;
    for (int i = 0; i < total_iters; ++i) {
        uint64_t t0 = h_results_a[2*i];
        uint64_t t3 = h_results_a[2*i + 1];
        uint64_t t1 = h_results_b[2*i];
        uint64_t t2 = h_results_b[2*i + 1];

        if (t0 == 0 || t1 == 0 || t2 == 0 || t3 == 0) continue;

		//std::cout << t0 << " " << t1 << " " << t2 << " " << t3 << std::endl;

        uint64_t rtt_cycles = t3 - t0 - (t2 - t1);
        rtts.push_back(rtt_cycles);
    }

	start = std::min((int)rtts.size(), start);
	end = std::min((int)rtts.size(), end);

	for (int i = start; i < end; i++) {
		std::cout << i << ": " << rtts[i] << std::endl;
	}

	print_stats(std::span(rtts), "cycels");

    CHECK_CUDA_ERROR(cudaSetDevice(GPU_A));
    CHECK_CUDA_ERROR(cudaFree(d_flag_a));
    CHECK_CUDA_ERROR(cudaFree(d_results_a));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_a));

    CHECK_CUDA_ERROR(cudaSetDevice(GPU_B));
    CHECK_CUDA_ERROR(cudaFree(d_flag_b));
    CHECK_CUDA_ERROR(cudaFree(d_results_b));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream_b));
}

int main(int argc, char* argv[]) {
    int num_iters = DEFAULT_ITERS;
    int num_gpus = 2;
	int start = 0;
	int end = 128;
	int delay = 0;

    static struct option long_options[] = {
        {"iters", required_argument, 0, 'i'},
        {"gpus",  required_argument, 0, 'g'},
		{"left", required_argument, 0, 'l'},
		{"right", required_argument, 0, 'r'},
		{"delay", required_argument, 0, 'd'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'i': num_iters = stoi(optarg); break;
            case 'g': num_gpus = stoi(optarg); break;
            case 'l': start = stoi(optarg); break;
            case 'e': end = stoi(optarg); break;
            case 'd': delay = stoi(optarg); break;
        }
    }

    assert(num_gpus == 2 && "This benchmark requires exactly 2 GPUs.");

    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (num_gpus > device_count) {
        cerr << "Only " << device_count << " GPUs available." << endl;
        return 1;
    }

    print_gpu_info(num_gpus);
    enable_p2p(num_gpus);

    run_nvlink_latency(num_iters, start, end, delay);

    return 0;
}
