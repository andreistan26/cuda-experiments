#pragma once

#include <span>
#include <numeric>
#include <type_traits>
#include <algorithm>

struct AlignedBuffer {
    void* raw_ptr = nullptr;
    void* aligned_ptr = nullptr;
    size_t total_allocated = 0;
	int dev_idx;

    explicit AlignedBuffer(size_t size, size_t alignment) {
		CHECK_CUDA_ERROR(cudaGetDevice(&dev_idx));
        total_allocated = size + alignment;
        CHECK_CUDA_ERROR(cudaMalloc(&raw_ptr, total_allocated));
        CHECK_CUDA_ERROR(cudaMemset(raw_ptr, 0, total_allocated));

        uintptr_t raw_addr = (uintptr_t)raw_ptr;
        uintptr_t aligned_addr = (raw_addr + alignment - 1) & ~(alignment - 1);
        aligned_ptr = (void*)aligned_addr;
    }

    ~AlignedBuffer() {
        if (raw_ptr) {
			CHECK_CUDA_ERROR(cudaSetDevice(dev_idx));
            CHECK_CUDA_ERROR(cudaFree(raw_ptr));
            raw_ptr = nullptr;
        }
    }
};

template <typename T>
concept Arithmetic = requires {
	std::is_arithmetic_v<T>;
};

template <Arithmetic T>
void print_stats(std::span<T> spn, std::string unit) {
	if (spn.size() == 0) {
		std::cerr << "Could not print stats, no datapoints." << std::endl;
		return;
	}

    std::sort(spn.begin(), spn.end());
    T median = spn[spn.size() / 2];
    T p99 = spn[size_t(0.99 * spn.size())];
    T min_val = spn.front();
    T max_val = spn.back();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Samples : " << spn.size() << std::endl;
    std::cout << "Median  : " << median << " " << unit << std::endl;
	std::cout << "Mean    : " << std::reduce(spn.begin(), spn.end()) / spn.size() << " " << unit << std::endl;
    std::cout << "Min     : " << min_val << " " << unit << std::endl;
    std::cout << "Max     : " << max_val << " " << unit << std::endl;
    std::cout << "P99     : " << p99 << " " << unit << std::endl;
}
