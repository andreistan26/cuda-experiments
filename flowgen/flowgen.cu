#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <getopt.h>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <numeric>
#include <cassert>
#include <concepts>
#include <memory>
#include <algorithm>
#include <limits>
#include <deque>
#include <atomic>
#include <random>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define JSON_HAS_CPP_20 0
#include <nlohmann/json.hpp>

#include "cuda_helpers.h"
#include "errors.h"
#include "cli.h"

using namespace std;
using json = nlohmann::json;

using uint = unsigned int;

int sync_device = 0;
constexpr int threadsPerBlock = 512;

using benchclock = std::chrono::high_resolution_clock;
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

/* absolute host time in microseconds; workers record raw timestamps and the
 * report is normalized afterwards by subtracting the earliest launch time */
static double us_now() {
    return std::chrono::duration<double, std::micro>(benchclock::now().time_since_epoch()).count();
}

static double ns_now() {
    return std::chrono::duration<size_t, std::nano>(benchclock::now().time_since_epoch()).count();
}

bool debug = getenv("DEBUG") != nullptr;
bool nsys = getenv("NSYS") != nullptr;
bool leaky = getenv("LEAKY") != nullptr;
size_t var_target_lat;

/* barrier that runs a completion function in the last arriving thread
 * before releasing everyone (used to capture the common epoch)
 * */
struct SyncBarrier {
    SyncBarrier(int count, std::function<void()> on_complete = {})
        : count(count), on_complete(std::move(on_complete)) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lk(m);
        size_t gen = generation;
        if (++waiting == count) {
            waiting = 0;
            generation++;
            if (on_complete) on_complete();
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return gen != generation; });
        }
    }

private:
    std::mutex m;
    std::condition_variable cv;
    int count;
    int waiting = 0;
    size_t generation = 0;
    std::function<void()> on_complete;
};

struct BufferPair {
    void* src;
    void* dest;
    size_t size;
};

/* read the GPU-wide nanosecond timer; consistent across all SMs */
__device__ __forceinline__ unsigned long long d_globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

/* striding copy body (from nvbandwidth), factored out so both the one-shot
 * kernel and the persistent spaced kernel share the same memcpy loop */
template <typename T>
__device__ __forceinline__ void striding_copy_body(BufferPair *iov, size_t size,
                                                   size_t from, unsigned int totalThreadCount) {
    for (int i = 0; i < size; i++) {
        size_t buffer_size = iov[i].size;
        const size_t chunkSizeInElement = buffer_size / (sizeof(T) * totalThreadCount);
        const size_t bigChunkSizeInElement = chunkSizeInElement / 4;
        T *dstBigEnd = (T *)iov[i].dest + (bigChunkSizeInElement * 4) * totalThreadCount;
        T *dstEnd = (T *)((char *)iov[i].dest + buffer_size);
        T *cdst = (T *)iov[i].dest + from;
        T *csrc = (T *)iov[i].src + from;

        while (cdst < dstBigEnd) {
            T pipe_0 = *csrc; csrc += totalThreadCount;
            T pipe_1 = *csrc; csrc += totalThreadCount;
            T pipe_2 = *csrc; csrc += totalThreadCount;
            T pipe_3 = *csrc; csrc += totalThreadCount;

            *cdst = pipe_0; cdst += totalThreadCount;
            *cdst = pipe_1; cdst += totalThreadCount;
            *cdst = pipe_2; cdst += totalThreadCount;
            *cdst = pipe_3; cdst += totalThreadCount;
        }

        while (cdst < dstEnd) {
            *cdst = *csrc; cdst += totalThreadCount; csrc += totalThreadCount;
        }
    }
}

/* taken from nvbandwidth */
template <typename T>
__global__ void striding_memcpy_kernel(BufferPair *iov, size_t size) {
    size_t from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int totalThreadCount = blockDim.x * gridDim.x;
    striding_copy_body<T>(iov, size, from, totalThreadCount);
}

/* persistent spaced kernel: runs `iters` copies in a single launch, recording
 * per-iteration device timestamps and busy-waiting `gap_ns` between them.
 * iter_ts holds 2 entries per iteration: [start, stop]. Requires a cooperative
 * launch so grid.sync() brackets each iteration's copy across the whole grid. */
template <typename T>
__global__ void spaced_striding_kernel(BufferPair *iov, size_t size, int iters,
                                       unsigned long long gap_ns,
                                       unsigned long long *iter_ts,
                                       volatile int *arrival_flags,
                                       volatile int *start_flag) {
    cg::grid_group grid = cg::this_grid();
    size_t from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int totalThreadCount = blockDim.x * gridDim.x;
    bool leader = (from == 0);

    /* park the resident grid until the host gives the go signal */
    grid.sync();
    if (leader) {
        while (*start_flag == 0) { }
    }
    grid.sync();
    for (int it = 0; it < iters; ++it) {
        if (leader) iter_ts[2 * it] = d_globaltimer();
        grid.sync();

        striding_copy_body<T>(iov, size, from, totalThreadCount);

        grid.sync();
        __threadfence_system();
        grid.sync();
        if (leader && arrival_flags) {
            arrival_flags[it] = it + 1;
            __threadfence_system();
        }
        grid.sync();
        unsigned long long stop = d_globaltimer();
        if (leader) iter_ts[2 * it + 1] = stop;

        /* in-kernel spacing: hold the grid for gap_ns before the next copy */
        if (gap_ns > 0 && it + 1 < iters) {
            while (d_globaltimer() - stop < gap_ns) {
                //__nanosleep(100);
            }
        }
    }
}

template <typename T>
__global__ void copy_kernel(BufferPair *iov, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (int iov_i = 0; iov_i < size; iov_i++) {
        size_t el_count = iov[iov_i].size / sizeof(T);
        T *src = (T *)iov[iov_i].src;
        T *dst = (T *)iov[iov_i].dest;

        for (size_t i = idx; i < el_count; i += stride) {
            dst[i] = src[i];
        }
    }
}

template <typename T>
__global__ void serial_copy_kernel(BufferPair *iov, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t thread_count = gridDim.x * blockDim.x;

    for (int iov_i = 0; iov_i < size; iov_i++) {
        size_t el_count = (iov[iov_i].size + sizeof(T) - 1) / sizeof(T);
		size_t chunk_count = el_count / thread_count;
		size_t el_start = chunk_count * idx;
		size_t el_end = min((chunk_count + 1) * idx, el_count);
        T *src = (T *)iov[iov_i].src;
        T *dst = (T *)iov[iov_i].dest;

        for (size_t i = el_start; i < el_end; i += 1) {
            dst[i] = src[i];
        }
    }
}

template <typename T>
__global__ void tma_memcpy(BufferPair *iov, size_t size) {

}

struct FlowDescriptor {
    int src_gpu;
    int dst_gpu;
};

enum FlowAlgo {
    SM_SIMPLE,
    SM_UNROLL,
    SM_SERIAL,
    SM_SPACED,
    TMA,
    CE,
    CC_CE,
	RATE_CE
};

std::string algo_to_string(FlowAlgo algo) {
    switch(algo) {
        case SM_SIMPLE: return "SM_SIMPLE";
        case SM_UNROLL: return "SM_UNROLL";
        case SM_SERIAL: return "SM_SERIAL";
        case SM_SPACED: return "SM_SPACED";
        case TMA: return "TMA";
        case CE: return "CE";
		case CC_CE: return "CC_CE";
        default: return "UNKNOWN";
    }
}

FlowAlgo parse_algo_string(const std::string& str) {
    if (str == "SM_SIMPLE" || str == "simple") return SM_SIMPLE;
    if (str == "SM_UNROLL" || str == "sm") return SM_UNROLL;
    if (str == "SM_SERIAL" || str == "serial") return SM_SERIAL;
    if (str == "SM_SPACED" || str == "spaced") return SM_SPACED;
    if (str == "TMA") return TMA;
	if (str == "CE" || str == "ce") return CE;
    if (str == "CC_CE" || str == "cc_ce") return CC_CE;
    std::cerr << "Unknown Algo String: " << str << ", defaulting to CE" << std::endl;
    return CE;
}

struct FlowConfig {
    FlowDescriptor desc;
    size_t buffer_size;
    size_t iov_size;
    int sm_count;
    FlowAlgo type;
    /* scheduling */
    double start_us;   /* launch offset from the common epoch */
    int iters;         /* timed iterations for this flow */
    double gap_us;     /* wait between iteration completion and next launch */
};

struct IterSample {
    double launch_us;   /* host launch time relative to the common epoch */
    double latency_us;  /* device time between start/stop events */
};

struct Flow {
    FlowDescriptor desc;
    cudaStream_t stream;
    cudaEvent_t start_event, stop_event;
    BufferPair *iov;
    BufferPair *d_iov;
    size_t iov_size;
    int sm_count;
    FlowAlgo type;
	size_t buffer_size;

    double start_us;
    int iters;
    double gap_us;
    std::vector<IterSample> samples;

    Flow(FlowConfig config)
        : desc(config.desc), iov_size(config.iov_size), sm_count(config.sm_count), type(config.type),
          start_us(config.start_us), iters(config.iters), gap_us(config.gap_us), buffer_size(config.buffer_size) {
        iov = new BufferPair[iov_size];
        for (int i = 0; i < iov_size; i++) {
            CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
            CHECK_CUDA_ERROR(cudaMalloc(&iov[i].src, buffer_size));
            CHECK_CUDA_ERROR(cudaMemset(iov[i].src, 0, buffer_size));
            CHECK_CUDA_ERROR(cudaSetDevice(desc.dst_gpu));
            CHECK_CUDA_ERROR(cudaMalloc(&iov[i].dest, buffer_size));
            CHECK_CUDA_ERROR(cudaMemset(iov[i].dest, 0, buffer_size));
            iov[i].size = buffer_size;
        }
        CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
		CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUDA_ERROR(cudaEventCreate(&start_event));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_event));
        CHECK_CUDA_ERROR(cudaMalloc(&d_iov, sizeof(BufferPair) * iov_size));
        CHECK_CUDA_ERROR(cudaMemcpy(d_iov, iov, sizeof(BufferPair) * iov_size, cudaMemcpyHostToDevice));
    }

    virtual void __launch_copy_kernel() = 0;

    virtual void launch_copy_kernel() {
        CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        CHECK_CUDA_ERROR(cudaEventRecord(start_event, stream));
        __launch_copy_kernel();
        CHECK_CUDA_ERROR(cudaEventRecord(stop_event, stream));
    }

    /* persistent flows run all iters (and their in-kernel gaps) in a single
     * launch; the host launches once and reads device-side timestamps back via
     * collect_samples instead of timing each iteration with CUDA events. */
    virtual bool persistent() const { return false; }
    virtual void collect_samples(double base_launch_us) {}
    virtual cudaError_t query_complete() { return cudaEventQuery(stop_event); }
    virtual void wait_idle() {}

    /* persistent kernels are launched up front and park in-kernel waiting for
     * signal_start(); the host signals at the scheduled time, keeping the
     * cooperative-launch cost (>1ms between enqueue and kernel start) out of
     * the timeline */
    virtual void signal_start() {}

    virtual ~Flow() {
        CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
        CHECK_CUDA_ERROR(cudaEventDestroy(start_event));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop_event));
        for (int i = 0; i < iov_size; i++) {
            CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
            CHECK_CUDA_ERROR(cudaFree(iov[i].src));
            CHECK_CUDA_ERROR(cudaSetDevice(desc.dst_gpu));
            CHECK_CUDA_ERROR(cudaFree(iov[i].dest));
        }
        CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        CHECK_CUDA_ERROR(cudaFree(d_iov));

        delete[] iov;
    }
};
struct CC_CEFlow : public Flow {
	size_t cwnd;
    CC_CEFlow(FlowConfig config) : Flow(config) {
		if (iov_size != 1) {
			std::cerr << "CC_CE does not support --kiter/iov_size != 1 yet" << std::endl;
			exit(1);
		}

    	cwnd = kStartCwndSz;
		CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
		for (int i = 0; i < kEventPoolSz; i++) {
			cudaEvent_t ev;
			CHECK_CUDA_ERROR(cudaEventCreate(&ev));
			free_events.push_back(ev);
		}
		CHECK_CUDA_ERROR(cudaEventCreate(&iter_start_event));
		CHECK_CUDA_ERROR(cudaEventCreate(&iter_end_event));
	}

	~CC_CEFlow() override {
		if (worker.joinable())
			worker.join();
		CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
		for (auto ev : free_events)
			CHECK_CUDA_ERROR(cudaEventDestroy(ev));
		CHECK_CUDA_ERROR(cudaEventDestroy(iter_start_event));
		CHECK_CUDA_ERROR(cudaEventDestroy(iter_end_event));
	}

	bool persistent() const override { return true; }

	void launch_copy_kernel() override {
		wait_idle();

		start.store(false, std::memory_order_release);
		worker_done.store(false, std::memory_order_release);
		pending_samples.clear();
		latencies_ns.clear();
		inflight_chunks.clear();
		inflight_size = 0;
    	cwnd = kStartCwndSz;
		worker = std::thread(&CC_CEFlow::run, this);
	}

	void signal_start() override {
		start.store(true, std::memory_order_release);
	}

	cudaError_t query_complete() override {
		return worker_done.load(std::memory_order_acquire) ? cudaSuccess : cudaErrorNotReady;
	}

	void wait_idle() override {
		if (worker.joinable())
			worker.join();
	}

	void collect_samples(double base_launch_us) override {
		samples.insert(samples.end(), pending_samples.begin(), pending_samples.end());
	}

    void __launch_copy_kernel() override {}

private:
	double serializationDelayNs(size_t size) {
		return size / kLinkSpeed_Bpns;
	}

	double targetDelayNs(size_t size) {
		size_t additional_oh_ns = nsys ? 3000 : 0;
		double base_target_delay = kLaunchOverhead_ns + serializationDelayNs(size) + additional_oh_ns;
		static constexpr size_t fs_min_cwnd = 1 * 1024 * 1024;
		static constexpr size_t fs_max_cwnd = 8 * 1024 * 1024;
		static constexpr double fs_range = 7000;
		static constexpr double isqrt_fs_min = 1 / 1024;
		static constexpr double isqrt_fs_max = 2896.30937574;
		static constexpr double alpha = fs_range / (isqrt_fs_min - isqrt_fs_max);
		static constexpr double beta = -alpha * isqrt_fs_max;
		return base_target_delay + std::max<double>(0, std::min(alpha / std::sqrt(cwnd) + beta, fs_range));
	}

	size_t available_window() const {
		if (cwnd <= inflight_size)
			return 0;
		return cwnd - inflight_size;
	}

	void send(size_t size, size_t offset) {
		auto start_event = free_events.front();
		free_events.pop_front();
		auto end_event = free_events.front();
		free_events.pop_front();


		CHECK_CUDA_ERROR(cudaEventRecord(start_event, stream));
		CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(
			(char *)iov[0].dest + offset,
			desc.dst_gpu,
			(char *)iov[0].src + offset,
			desc.src_gpu,
			size,
			stream
		));
		CHECK_CUDA_ERROR(cudaEventRecord(end_event, stream));
		inflight_chunks.push_back(InflightChunk{
			.start = start_event,
			.end = end_event,
			.query_time = ns_now() + targetDelayNs(size) - 2000, // start polling 1us before expected completion time
			.size = size,
		});
		//inflight_size += size;
	}

	bool first_call = true;
	void update_on_ack(size_t size, double latency_ns) {
		//latencies_ns.push_back(latency_ns);
		double target_delay = targetDelayNs(size);
		double queueing_delay = latency_ns - target_delay;
		double actual_rtt = latency_ns - serializationDelayNs(size) - 3200;
		constexpr static double target_lat = 7000;
		if (unlikely(debug))
			std::cout
				<< "On ACK :\n" 
				<< "\ttarget latency = " << target_delay << "\n"
				<< "\tactual latency = " << latency_ns << "\n"
				<< "\tactual rtt     = " << actual_rtt << "\n" 
				<< "\tqueueing delay = " << queueing_delay << "\n"
				<< "\tsize           = " << size << "\n"
				<< "\tcwnd           = " << cwnd << "\n"
				<< "\tinflight       = " << inflight_size << "\n";

		if (actual_rtt > target_delay && !first_call) {
			cwnd = std::max<size_t>(
					cwnd * std::max<double>(1 - (actual_rtt - target_lat) / actual_rtt, 0.5),
					kMinCwndSz);
		} else {
			cwnd = std::min(cwnd + size * 0.1, (double)kMaxCwndSz);
		}
		first_call = false;

		//inflight_size -= size;
	}

	// Has to be done with queueing delay
	void check_inflight(size_t now, bool force = false) {
		float ms;
		while (!inflight_chunks.empty()) {
			auto front = inflight_chunks.front();
			if (force) now = ns_now();
			if (!force && front.query_time > now)
				break;

			cudaError_t err = cudaEventQuery(front.end);
			if (err == cudaErrorNotReady) {
				if (force) continue;
				else break;
			}
			CHECK_CUDA_ERROR(err);

			inflight_chunks.pop_front();
			CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, front.start, front.end));
			double latency_ns = ms * 1e6;
			update_on_ack(front.size, latency_ns);
			free_events.push_back(front.start);
			free_events.push_back(front.end);
		}
	}

	void run() {
		CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
		while (!start.load(std::memory_order_acquire))
			;//std::this_thread::yield();

		double next_iter_ns = ns_now();
		for (int iter = 0; iter < iters; ++iter) {
			while (true) {
				double now = ns_now();
				check_inflight(now);
				if (now >= next_iter_ns)
					break;
			}

			//CHECK_CUDA_ERROR(cudaEventRecord(iter_start_event, stream));
			double launch_ns = ns_now();
			size_t offset = 0;
			while (offset < buffer_size) {
				size_t now = ns_now();
				check_inflight(now);
				size_t window = available_window();
				size_t remaining = buffer_size - offset;
				if (!can_send() || window == 0 || (remaining >= kMinCwndSz && window < kMinCwndSz)) {
					//std::this_thread::yield();
					continue;
				}

				size_t current_chunk = std::min(kMaxCwndSz, std::min(window, remaining));
				send(current_chunk, offset);
				offset += current_chunk;
			}

			//CHECK_CUDA_ERROR(cudaEventRecord(iter_end_event, stream));
			check_inflight(ns_now(), true);
			//CHECK_CUDA_ERROR(cudaEventSynchronize(iter_end_event));
			double end_ns = ns_now();

			//float ms = 0;
			//CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, iter_start_event, iter_end_event));
			pending_samples.push_back({(double)launch_ns / 1000, (double)(end_ns - launch_ns) / 1000});
			next_iter_ns = ns_now() + gap_us * 1000.0;
		}

		worker_done.store(true, std::memory_order_release);
	}

	bool can_send() {
		return inflight_chunks.size() < maxQueuedChunks;
	}

	static constexpr long kLaunchOverhead_ns = 4500;
	static constexpr long kLinkSpeed = 400 * 1e9; // H100/H200 400 GB/s
	static constexpr long kLinkSpeed_Bpns = 400; // H100/H200 400 GB/s
	static constexpr long kEventPoolSz = 16 * 2; // we assume 16 queued requests
	static constexpr size_t kStartCwndSz = 16 * 1024 * 1024; // 8 MB (with CE ~340B/s) which is the fair bw share on h100/h200
	static constexpr size_t kMinCwndSz = 512 * 1024; // 512 KB (with CE ~50GB/s) which is the fair bw share on h100/h200
	static constexpr size_t kMaxCwndSz = 32 * 1024 * 1024; // 128MB (with CE ~390GB/s) which is the fair bw share on h100/h200
	static constexpr long maxQueuedChunks = 4;

	struct InflightChunk {
		cudaEvent_t start, end;
		double query_time; // To lower overhead of cudaEventQuery()
		size_t size; // size of the chunk
	};

	std::thread worker;
	cudaEvent_t iter_start_event, iter_end_event;
	std::deque<cudaEvent_t> free_events;
	std::deque<InflightChunk> inflight_chunks;
	std::vector<double> latencies_ns;
	std::vector<IterSample> pending_samples;
	size_t inflight_size{0};

	std::atomic<bool> start{false};
	std::atomic<bool> worker_done{false};
};

// Copy of CC_CEFlow should be abstracted but not now :(
// Implementation of a leaky bucket algorithm for CE congestion control
struct CC_LEAKY_CEFlow : public Flow {
	double admission_rate_GBps;
	size_t bucket_size;
	std::mt19937_64 rng;
    CC_LEAKY_CEFlow(FlowConfig config) : Flow(config), rng(std::random_device{}()) {
		if (iov_size != 1) {
			std::cerr << "CC_CE does not support --kiter/iov_size != 1 yet" << std::endl;
			exit(1);
		}

		admission_rate_GBps = kMaxAdmissionRateGBps;
		CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
		for (int i = 0; i < kEventPoolSz; i++) {
			cudaEvent_t ev;
			CHECK_CUDA_ERROR(cudaEventCreate(&ev));
			free_events.push_back(ev);
		}
		CHECK_CUDA_ERROR(cudaEventCreate(&iter_start_event));
		CHECK_CUDA_ERROR(cudaEventCreate(&iter_end_event));
	}

	~CC_LEAKY_CEFlow() override {
		if (worker.joinable())
			worker.join();
		CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
		for (auto ev : free_events)
			CHECK_CUDA_ERROR(cudaEventDestroy(ev));
		CHECK_CUDA_ERROR(cudaEventDestroy(iter_start_event));
		CHECK_CUDA_ERROR(cudaEventDestroy(iter_end_event));
	}

	bool persistent() const override { return true; }

	void launch_copy_kernel() override {
		wait_idle();

		start.store(false, std::memory_order_release);
		worker_done.store(false, std::memory_order_release);
		pending_samples.clear();
		latencies_ns.clear();
		inflight_chunks.clear();
		inflight_size = 0;
		admission_rate_GBps = kMaxAdmissionRateGBps;
		bucket_size = 0;
		worker = std::thread(&CC_LEAKY_CEFlow::run, this);
	}

	void signal_start() override {
		start.store(true, std::memory_order_release);
	}

	cudaError_t query_complete() override {
		return worker_done.load(std::memory_order_acquire) ? cudaSuccess : cudaErrorNotReady;
	}

	void wait_idle() override {
		if (worker.joinable())
			worker.join();
	}

	void collect_samples(double base_launch_us) override {
		samples.insert(samples.end(), pending_samples.begin(), pending_samples.end());
	}

    void __launch_copy_kernel() override {}

private:
	double serializationDelayNs(size_t size) {
		return size / kLinkSpeed_Bpns;
	}

	double targetDelayNs(size_t size) {
		size_t additional_oh_ns = nsys ? 3000 : 0;
		double base_target_delay = kLaunchOverhead_ns + serializationDelayNs(size) + additional_oh_ns;
		static constexpr size_t fs_min_cwnd = 80;
		static constexpr size_t fs_max_cwnd = 350;
		static constexpr double fs_range = 7000;
		static constexpr double isqrt_fs_min = 1 / 8.9;
		static constexpr double isqrt_fs_max = 18.708;
		static constexpr double alpha = fs_range / (isqrt_fs_min - isqrt_fs_max);
		static constexpr double beta = -alpha * isqrt_fs_max;
		return base_target_delay + std::max<double>(0, std::min(alpha / std::sqrt(admission_rate_GBps) + beta, fs_range));
	}


	void send(size_t size, size_t offset) {
		auto start_event = free_events.front();
		free_events.pop_front();
		auto end_event = free_events.front();
		free_events.pop_front();


		CHECK_CUDA_ERROR(cudaEventRecord(start_event, stream));
		CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(
			(char *)iov[0].dest + offset,
			desc.dst_gpu,
			(char *)iov[0].src + offset,
			desc.src_gpu,
			size,
			stream
		));
		CHECK_CUDA_ERROR(cudaEventRecord(end_event, stream));
		inflight_chunks.push_back(InflightChunk{
			.start = start_event,
			.end = end_event,
			.query_time = ns_now() + targetDelayNs(size) - 2000, // start polling 1us before expected completion time
			.size = size,
		});
		//inflight_size += size;
	}

	bool first_call = true;
	void update_on_ack(size_t size, double latency_ns) {

		double target_delay = targetDelayNs(size);
		double queueing_delay = latency_ns - target_delay;
		double actual_rtt = latency_ns - serializationDelayNs(size) - 3200;
		constexpr static double target_lat = 7000;
		if (unlikely(debug))
			std::cout
				<< "On ACK :\n" 
				<< "\ttarget latency = " << target_delay << "\n"
				<< "\tactual latency = " << latency_ns << "\n"
				<< "\tactual rtt     = " << actual_rtt << "\n" 
				<< "\tqueueing delay = " << queueing_delay << "\n"
				<< "\tsize           = " << size << "\n"
				<< "\tadmission rate = " << admission_rate_GBps << "\n"
				<< "\tinflight chunks= " << inflight_chunks.size() << "\n";

		if (actual_rtt > target_delay && !first_call) {
			admission_rate_GBps = std::max<double>(
					admission_rate_GBps * std::max<double>(1 - (actual_rtt - target_delay) / actual_rtt, 0.5),
					kMinAdmissionRateGBps);
		} else {
			admission_rate_GBps = std::min(admission_rate_GBps + admission_rate_GBps * 0.1, kMaxAdmissionRateGBps);
		}
		first_call = false;

		//inflight_size -= size;
	}

	// Has to be done with queueing delay
	void check_inflight(size_t now, bool force = false) {
		float ms;
		while (!inflight_chunks.empty()) {
			auto front = inflight_chunks.front();
			if (force) now = ns_now();
			if (!force && front.query_time > now)
				break;

			cudaError_t err = cudaEventQuery(front.end);
			if (err == cudaErrorNotReady) {
				if (force) continue;
				else break;
			}
			CHECK_CUDA_ERROR(err);

			inflight_chunks.pop_front();
			CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, front.start, front.end));
			double latency_ns = ms * 1e6;
			update_on_ack(front.size, latency_ns);
			free_events.push_back(front.start);
			free_events.push_back(front.end);
		}
	}
	void update_bucket(size_t last_ns, size_t current_ns) {
		size_t delta_ns = current_ns - last_ns;
		bucket_size += admission_rate_GBps * delta_ns;
	}

	void run() {
		CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
		while (!start.load(std::memory_order_acquire))
			;//std::this_thread::yield();

		double next_iter_ns = ns_now();
		for (int iter = 0; iter < iters; ++iter) {
			while (true) {
				double now = ns_now();
				check_inflight(now);
				if (now >= next_iter_ns)
					break;
			}

			//CHECK_CUDA_ERROR(cudaEventRecord(iter_start_event, stream));
			double launch_ns = ns_now();
			size_t offset = 0;
			while (offset < buffer_size) {
				size_t remaining = buffer_size - offset;
				size_t current_chunk = std::min(kChunkSize, remaining);
				size_t send_threshold =
					std::uniform_int_distribution<size_t>(0, current_chunk)(rng);
				size_t last_ns = ns_now();
				bool sent = false;

				while (bucket_size < current_chunk || !sent) {
					size_t now = ns_now();
					update_bucket(last_ns, now);
					last_ns = now;
					bucket_size = std::min(bucket_size, current_chunk);
					check_inflight(now);

					if (!sent &&
						bucket_size >= send_threshold &&
						inflight_chunks.size() < maxQueuedChunks) {
						send(current_chunk, offset);
						sent = true;
					}
				}

				bucket_size = 0;
				offset += current_chunk;
			}

			//CHECK_CUDA_ERROR(cudaEventRecord(iter_end_event, stream));
			check_inflight(ns_now(), true);
			//CHECK_CUDA_ERROR(cudaEventSynchronize(iter_end_event));
			double end_ns = ns_now();

			//float ms = 0;
			//CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, iter_start_event, iter_end_event));
			pending_samples.push_back({(double)launch_ns / 1000, (double)(end_ns - launch_ns) / 1000});
			next_iter_ns = ns_now() + gap_us * 1000.0;
		}

		worker_done.store(true, std::memory_order_release);
	}

	static constexpr long kLaunchOverhead_ns = 4500;
	static constexpr long kLinkSpeed = 400 * 1e9; // H100/H200 400 GB/s
	static constexpr long kLinkSpeed_Bpns = 400; // H100/H200 400 GB/s
	static constexpr long kEventPoolSz = 16 * 2; // we assume 16 queued requests
	static constexpr size_t kStartCwndSz = 16 * 1024 * 1024; // 8 MB (with CE ~340B/s) which is the fair bw share on h100/h200
	static constexpr size_t kMinCwndSz = 512 * 1024; // 512 KB (with CE ~50GB/s) which is the fair bw share on h100/h200
	static constexpr size_t kMaxCwndSz = 32 * 1024 * 1024; // 128MB (with CE ~390GB/s) which is the fair bw share on h100/h200
	static constexpr long maxQueuedChunks = 4;
	static constexpr double kMaxAdmissionRateGBps = 400; // Assumint H100
	static constexpr double kMinAdmissionRateGBps = 50; // Assuming 8GPUs in a scale-up domain
	static constexpr size_t kChunkSize = 32 * 1024 * 1024; // 16MB should be enough to go line rate

	struct InflightChunk {
		cudaEvent_t start, end;
		double query_time; // To lower overhead of cudaEventQuery()
		size_t size; // size of the chunk
	};

	std::thread worker;
	cudaEvent_t iter_start_event, iter_end_event;
	std::deque<cudaEvent_t> free_events;
	std::deque<InflightChunk> inflight_chunks;
	std::vector<double> latencies_ns;
	std::vector<IterSample> pending_samples;
	size_t inflight_size{0};

	std::atomic<bool> start{false};
	std::atomic<bool> worker_done{false};
};

struct CEFlow : public Flow {
    CEFlow(FlowConfig config) : Flow(config) {}

    void __launch_copy_kernel() {
        for (int i = 0; i < iov_size; i++) {
            CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(
                iov[i].dest, desc.dst_gpu,
                iov[i].src, desc.src_gpu,
                iov[i].size, stream
            ));
        }
    }
};

struct SMSimpleFlow : public Flow {
    SMSimpleFlow(FlowConfig config) : Flow(config) {}

    void __launch_copy_kernel() {
        auto buf_size = iov[0].size;
        int grid_size = std::min((int)sm_count, (int)((buf_size + threadsPerBlock - 1) / threadsPerBlock));

        dim3 gridDim(grid_size, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);
        copy_kernel<int><<<gridDim, blockDim, 0, stream>>>(d_iov, iov_size);
    }
};

struct SMSerialFlow : public Flow {
    SMSerialFlow(FlowConfig config) : Flow(config) {}

    void __launch_copy_kernel() {
        auto buf_size = iov[0].size;
        int grid_size = std::min((int)sm_count, (int)((buf_size + threadsPerBlock - 1) / threadsPerBlock));

        dim3 gridDim(grid_size, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);
        copy_kernel<char><<<gridDim, blockDim, 0, stream>>>(d_iov, iov_size);
    }
};


struct SMUnrolledFlow : public Flow {
    SMUnrolledFlow(FlowConfig config) : Flow(config) {}

    void __launch_copy_kernel() {
        using T = uint4;
        int grid_size = std::min((int)((iov[0].size + (sizeof(T) * 4 * threadsPerBlock) - 1)
                / (sizeof(T) * 4 * threadsPerBlock)), sm_count);
        const unsigned int totalThreadCount = grid_size * threadsPerBlock;
        for (int i = 0; i < iov_size; i++) {
            auto &buf = iov[i];
            assert((buf.size % sizeof(T)) == 0);
            /* 12 is the unroll factor, stride is totalThreadCount with pipeline length of totalThreadCount */
            //assert(buf.size >= (12 * totalThreadCount * sizeof(T)));
            /* check that buffers dont overlap */
            assert((buf.src > buf.dest && (char*)buf.src >= (char *)buf.dest + buf.size)
                    || (buf.src < buf.dest && (char *)buf.src + buf.size <= (char*)buf.dest));
        }

        dim3 gridDim(grid_size, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);
        striding_memcpy_kernel<uint4><<<gridDim, blockDim, 0, stream>>>(d_iov, iov_size);
    }
};

struct SMSpacedFlow : public Flow {
    using T = uint4;
    unsigned long long *d_iter_ts = nullptr;   /* 2 entries per iter: [start, stop] */
    int *d_arrival_flags = nullptr; /* written from src GPU into dst GPU memory */
    std::vector<unsigned long long> h_iter_ts;
    std::vector<int> h_arrival_flags;
    int *h_start_flag = nullptr;  /* mapped, CPU->GPU: go signal */
    int *d_start_flag = nullptr;

    SMSpacedFlow(FlowConfig config) : Flow(config) {
        h_iter_ts.resize(std::max(0, iters) * 2);
        h_arrival_flags.resize(std::max(0, iters));
        CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        CHECK_CUDA_ERROR(cudaMalloc(&d_iter_ts, sizeof(unsigned long long) * h_iter_ts.size()));
        CHECK_CUDA_ERROR(cudaHostAlloc((void **)&h_start_flag, sizeof(int), cudaHostAllocMapped));
        CHECK_CUDA_ERROR(cudaHostGetDevicePointer((void **)&d_start_flag, h_start_flag, 0));
        *h_start_flag = 0;
        if (!h_arrival_flags.empty()) {
            CHECK_CUDA_ERROR(cudaSetDevice(desc.dst_gpu));
            CHECK_CUDA_ERROR(cudaMalloc(&d_arrival_flags, sizeof(int) * h_arrival_flags.size()));
            CHECK_CUDA_ERROR(cudaMemset(d_arrival_flags, 0, sizeof(int) * h_arrival_flags.size()));
            CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        }
    }

    bool persistent() const override { return true; }

    void signal_start() override { *(volatile int *)h_start_flag = 1; }

    void __launch_copy_kernel() override {
        int data_grid = std::min((int)((iov[0].size + (sizeof(T) * 4 * threadsPerBlock) - 1)
                / (sizeof(T) * 4 * threadsPerBlock)), sm_count);

        //CHECK_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        //    &blocks_per_sm, (void *)spaced_striding_kernel<T>, threadsPerBlock, 0));
        int grid_size = std::max(1, data_grid);

        unsigned long long gap_ns = (unsigned long long)(gap_us * 1000.0);

        *(volatile int *)h_start_flag = 0;
        if (d_arrival_flags) {
            CHECK_CUDA_ERROR(cudaSetDevice(desc.dst_gpu));
            CHECK_CUDA_ERROR(cudaMemset(d_arrival_flags, 0, sizeof(int) * h_arrival_flags.size()));
            CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        }

        dim3 gridDim(grid_size, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);
        void *args[] = {&d_iov, &iov_size, &iters, &gap_ns, &d_iter_ts, &d_arrival_flags, &d_start_flag};
        CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel(
            (void *)spaced_striding_kernel<T>, gridDim, blockDim, args, 0, stream));
    }

    void collect_samples(double base_launch_us) override {
        CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        CHECK_CUDA_ERROR(cudaMemcpy(h_iter_ts.data(), d_iter_ts,
            sizeof(unsigned long long) * h_iter_ts.size(), cudaMemcpyDeviceToHost));
        if (d_arrival_flags) {
            CHECK_CUDA_ERROR(cudaSetDevice(desc.dst_gpu));
            CHECK_CUDA_ERROR(cudaMemcpy(h_arrival_flags.data(), d_arrival_flags,
                sizeof(int) * h_arrival_flags.size(), cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        }

        unsigned long long ref = h_iter_ts.empty() ? 0 : h_iter_ts[0];
        for (int it = 0; it < iters; ++it) {
            unsigned long long start = h_iter_ts[2 * it];
            unsigned long long stop = h_iter_ts[2 * it + 1];
            if (!h_arrival_flags.empty() && h_arrival_flags[it] != it + 1) {
                std::cerr << "SM_SPACED arrival ack missing for iter " << it
                          << ": expected " << (it + 1)
                          << ", got " << h_arrival_flags[it] << std::endl;
                exit(1);
            }
            IterSample s;
            s.launch_us = base_launch_us + (double)(start - ref) / 1000.0;
            s.latency_us = (double)(stop - start) / 1000.0;
            samples.push_back(s);
        }
    }

    ~SMSpacedFlow() override {
        if (d_arrival_flags) {
            CHECK_CUDA_ERROR(cudaSetDevice(desc.dst_gpu));
            CHECK_CUDA_ERROR(cudaFree(d_arrival_flags));
        }
        CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        CHECK_CUDA_ERROR(cudaFree(d_iter_ts));
        CHECK_CUDA_ERROR(cudaFreeHost(h_start_flag));
    }
};

std::unique_ptr<Flow> to_flow(FlowConfig cfg) {
    switch (cfg.type) {
    case CC_CE:
		if (cfg.iov_size != 1) {
			std::cerr << "CC_CE does not support --kiter/iov_size != 1 yet" << std::endl;
			exit(1);
		}
		if (leaky)
			return std::make_unique<CC_LEAKY_CEFlow>(cfg);
        return std::make_unique<CC_CEFlow>(cfg);
    case CE:
        return std::make_unique<CEFlow>(cfg);
    case SM_SIMPLE:
        return std::make_unique<SMSimpleFlow>(cfg);
    case SM_SERIAL:
        return std::make_unique<SMSerialFlow>(cfg);
    case SM_UNROLL:
        return std::make_unique<SMUnrolledFlow>(cfg);
    case SM_SPACED:
        return std::make_unique<SMSpacedFlow>(cfg);
    default:
        std::cerr << "Unimplemented flow alg " << cfg.type << " " << std::endl;
        exit(1);
    }
}

enum Pattern {
    SINGLE,
    INCAST,
    OUTCAST,
    ALLTOALL,
    RING,
    INCAST_WITH_SEND,
    INCAST_WITH_SEND_REV
};

const char *pattern_to_str(Pattern &pattern) {
    switch (pattern) {
        case SINGLE: return "single";
        case INCAST: return "incast";
        case OUTCAST: return "outcast";
        case ALLTOALL: return "alltoall";
        case RING: return "ring";
        case INCAST_WITH_SEND: return "incast with send";
        case INCAST_WITH_SEND_REV: return "incast with additional flow";
        default: return "unknown";
    }
}

vector<FlowConfig> generate_flows(Pattern pattern, int num_gpus, FlowAlgo flow_type,
                                  size_t size, size_t kernel_iter, int sm_limit,
                                  double start_us, int iters, double gap_us,
                                  int target_gpu = 0, int send_target = 1) {
    vector<FlowDescriptor> descs;
    vector<FlowConfig> configs;

    switch (pattern) {
        case SINGLE:
            descs.push_back({send_target, target_gpu});
            break;
        case INCAST:
            for (int i = 0; i < num_gpus; ++i) {
                if (i != target_gpu) {
                    descs.push_back({i, target_gpu});
                }
            }
            break;
        case OUTCAST:
            for (int i = 0; i < num_gpus; ++i) {
                if (i != target_gpu) {
                    descs.push_back({target_gpu, i});
                }
            }
            break;
        case ALLTOALL:
            for (int i = 0; i < num_gpus; ++i) {
                for (int j = 0; j < num_gpus; ++j) {
                    if (i != j) {
                        descs.push_back({i, j});
                    }
                }
            }
            break;
        case INCAST_WITH_SEND_REV: {
            int pair[2] = {-1, -1};
            int idx = 0;
            for (int i = 0; i < num_gpus; ++i) {
                if (i != target_gpu) {
                    if (idx <= 1) pair[idx++] = i;
                    descs.push_back({i, target_gpu});
                }
            }
            assert(idx == 2);
            descs.push_back({pair[0], pair[1]});
            break;
        }
        case RING:
            for (int i = 0; i < num_gpus; ++i) {
                int next = (i + 1) % num_gpus;
                descs.push_back({i, next});
            }
            break;
        case INCAST_WITH_SEND:
            for (int i = 0; i < num_gpus; ++i) {
                if (i != target_gpu) {
                    descs.push_back({i, target_gpu});
                }
            }
            if (target_gpu != send_target) {
                descs.push_back({target_gpu, send_target});
            }
            break;
    }

    for (auto& d : descs) {
        FlowConfig c;
        c.desc = d;
        c.buffer_size = size;
        c.iov_size = kernel_iter;
        c.type = flow_type;
        c.sm_count = sm_limit;
        c.start_us = start_us;
        c.iters = iters;
        c.gap_us = gap_us;
        configs.push_back(c);
    }
    return configs;
}

std::vector<FlowConfig> from_json(const string &filename, double def_start_us, int def_iters, double def_gap_us) {
	vector<FlowConfig> configs;
	std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Failed to open JSON file: " << filename << std::endl;
        exit(1);
    }
    json j;
    f >> j;

    for (auto& el : j) {
        FlowConfig c;
        c.desc.src_gpu = el.at("src_gpu").get<int>();
        c.desc.dst_gpu = el.at("dst_gpu").get<int>();
        c.buffer_size = parse_size(el.value("buffer_size", "1m").c_str());

        c.iov_size = el.value("iov_size", 1ULL);
        std::string algo_str = el.value("type", "CE");
        c.type = parse_algo_string(algo_str);
        c.sm_count = el.value("sm_count", 0);
        c.start_us = el.value("start_us", def_start_us);
        c.iters = el.value("iters", def_iters);
        c.gap_us = el.value("gap_us", def_gap_us);

        configs.push_back(c);
    }
    return configs;
}

struct FlowResult {
    int src, dst;
    FlowAlgo type;
    size_t buffer_size;
    size_t iov_size;
	int sm_count;
    double start_us;
    double gap_us;
    std::vector<IterSample> samples;

    size_t bytes_per_iter() const { return buffer_size * iov_size; }

    double bandwidth_GBs(double latency_us) const {
        if (latency_us <= 0.01) return 0.0;
        return (static_cast<double>(bytes_per_iter()) / 1.0e9) / (latency_us / 1.0e6);
    }

    double avg_latency_us() const {
        if (samples.empty()) return 0.0;
        double sum = 0;
        for (const auto& s : samples) sum += s.latency_us;
        return sum / samples.size();
    }

    double min_latency_us() const {
        double m = std::numeric_limits<double>::max();
        for (const auto& s : samples) m = std::min(m, s.latency_us);
        return samples.empty() ? 0.0 : m;
    }

    double max_latency_us() const {
        double m = 0;
        for (const auto& s : samples) m = std::max(m, s.latency_us);
        return m;
    }

    json to_json() const {
        std::vector<json> iters_json;
        for (const auto& s : samples) {
            iters_json.push_back(json{
                {"launch_us", s.launch_us},
                {"latency_us", s.latency_us},
                {"bandwidth_GBs", bandwidth_GBs(s.latency_us)}
            });
        }
        return json{
            {"src", src},
            {"dst", dst},
            {"type", algo_to_string(type)},
            {"buffer_size", buffer_size},
            {"iov_size", iov_size},
            {"bytes_per_iter", bytes_per_iter()},
            {"sm_count", sm_count},
            {"start_us", start_us},
            {"gap_us", gap_us},
            {"iters_completed", samples.size()},
            {"avg_latency_us", avg_latency_us()},
            {"min_latency_us", min_latency_us()},
            {"max_latency_us", max_latency_us()},
            {"avg_bandwidth_GBs", bandwidth_GBs(avg_latency_us())},
            {"iterations", iters_json}
        };
    }
};

struct BenchmarkReport {
    std::vector<FlowResult> flows;

    json as_perfetto_json() const {
        std::vector<json> events;

        for (const auto& f : flows) {
            events.push_back(json{
                {"name", "process_name"}, {"ph", "M"},
                {"pid", f.src},
                {"args", {{"name", "GPU " + std::to_string(f.src) + " (sender)"}}}
            });
        }

        for (size_t i = 0; i < flows.size(); ++i) {
            const auto& f = flows[i];
            int tid = (int)i;
            std::string flow_name = std::to_string(f.src) + "->" + std::to_string(f.dst)
                + " " + algo_to_string(f.type)
                + " " + std::to_string(f.bytes_per_iter()) + "B";

            events.push_back(json{
                {"name", "thread_name"}, {"ph", "M"},
                {"pid", f.src}, {"tid", tid},
                {"args", {{"name", flow_name}}}
            });

            for (const auto& s : f.samples) {
                events.push_back(json{
                    {"name", flow_name},
                    {"ph", "X"},
                    {"ts", s.launch_us},
                    {"dur", s.latency_us},
                    {"pid", f.src},
                    {"tid", tid},
                    {"args", {
                        {"latency_us", s.latency_us},
                        {"bandwidth_GBs", f.bandwidth_GBs(s.latency_us)},
                        {"bytes", f.bytes_per_iter()}
                    }}
                });
            }
        }

        return json{{"traceEvents", events}};
    }

    json as_json() const {
        double avg_lat = 0;
        if(!flows.empty()) {
            for(const auto& f : flows) avg_lat += f.avg_latency_us();
            avg_lat /= flows.size();
        }

        json j;
        j["average_latency_us"] = avg_lat;

        std::vector<json> flows_json;
        for(const auto& f : flows) flows_json.push_back(f.to_json());
        j["flows"] = flows_json;
        return j;
    }

    void pretty_print() const {
        if (flows.empty()) {
            cout << "No flows measured." << endl;
            return;
        }

        cout << fixed << setprecision(2);
        cout << "Per-flow details:" << endl;

        for (const auto& f : flows) {
            cout << "Flow " << f.src << " -> " << f.dst
                 << " (" << algo_to_string(f.type) << ", "
                 << f.bytes_per_iter() << " B/iter, start " << f.start_us << " us, gap " << f.gap_us << " us)"
                 << ": " << f.samples.size() << " iters, "
                 << "latency avg " << f.avg_latency_us()
                 << " / min " << f.min_latency_us()
                 << " / max " << f.max_latency_us() << " us, "
                 << "avg bandwidth " << f.bandwidth_GBs(f.avg_latency_us()) << " GB/s" << endl;

            for (const auto& s : f.samples) {
                cout << "    t=" << setw(12) << s.launch_us << " us"
                     << "  latency " << setw(10) << s.latency_us << " us"
                     << "  " << setw(8) << f.bandwidth_GBs(s.latency_us) << " GB/s"
                     << endl;
            }
        }
    }
};

struct ScheduledFlow {
    Flow *flow;
    enum State { WAITING, INFLIGHT, DONE } state = WAITING;
    double next_launch_us = 0;
    double launch_us = 0;
    int completed = 0;
};

void gpu_worker(int gpu, vector<Flow*> flows, int warmup_iters,
                SyncBarrier &start_barrier, const double &epoch_us) {
    CHECK_CUDA_ERROR(cudaSetDevice(gpu));

    for (int w = 0; w < warmup_iters; ++w) {
        for (auto *f : flows) {
            f->launch_copy_kernel();
            f->signal_start();   /* persistent kernels park until signaled */
        }
        CHECK_CUDA_ERROR(cudaSetDevice(gpu));
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        for (auto *f : flows)
            f->wait_idle();
    }

    {
        CHECK_CUDA_ERROR(cudaSetDevice(gpu));
        l2flush flusher;
        flusher.flush_sync(0);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    /* launch persistent kernels up front; they park in-kernel waiting for the
     * go signal, so the blocking cooperative launch stays before the epoch */
    for (auto *f : flows) {
        if (f->persistent() && f->iters > 0) {
            f->launch_copy_kernel();
        }
    }

    start_barrier.arrive_and_wait();

	std::deque<ScheduledFlow> sched;
    size_t done = 0;
    for (auto *f : flows) {
        ScheduledFlow s;
        s.flow = f;
        s.next_launch_us = epoch_us + f->start_us;
        if (f->iters <= 0) {
            s.state = ScheduledFlow::DONE;
            done++;
        }
        sched.push_back(s);
    }

    /* WAITING flows kept sorted by next_launch_us so the launch pass can stop
     * at the first flow that is not due yet */
    auto by_deadline = [](const ScheduledFlow *a, const ScheduledFlow *b) {
        return a->next_launch_us < b->next_launch_us;
    };
    std::deque<ScheduledFlow *> waiting;
    auto enqueue_waiting = [&](ScheduledFlow *s) {
        waiting.insert(std::lower_bound(waiting.begin(), waiting.end(), s, by_deadline), s);
    };
    std::vector<ScheduledFlow *> inflight;

    for (auto &s : sched)
        if (s.state == ScheduledFlow::WAITING)
            enqueue_waiting(&s);

    while (done < sched.size()) {
        double now = us_now();
        while (!waiting.empty() && waiting.front()->next_launch_us <= now) {
            ScheduledFlow *s = waiting.front();
            waiting.pop_front();
            s->launch_us = us_now();
            if (s->flow->persistent())
                s->flow->signal_start();   /* kernel already resident, parked */
            else
                s->flow->launch_copy_kernel();
            s->state = ScheduledFlow::INFLIGHT;
            inflight.push_back(s);
        }

        for (size_t i = 0; i < inflight.size();) {
            ScheduledFlow *s = inflight[i];
            cudaError_t q = s->flow->query_complete();
            if (q == cudaErrorNotReady) {
                ++i;
                continue;
            }
            CHECK_CUDA_ERROR(q);

            if (s->flow->persistent()) {
                s->flow->collect_samples(s->launch_us);
                s->state = ScheduledFlow::DONE;
                done++;
            } else {
                float ms = 0;
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, s->flow->start_event, s->flow->stop_event));
                s->flow->samples.push_back({s->launch_us, (double)ms * 1000.0});
                s->completed++;
                if (s->completed >= s->flow->iters) {
                    s->state = ScheduledFlow::DONE;
                    done++;
                } else {
                    s->state = ScheduledFlow::WAITING;
                    s->next_launch_us = us_now() + s->flow->gap_us;
                    enqueue_waiting(s);
                }
            }
            inflight[i] = inflight.back();
            inflight.pop_back();
        }
    }
}

BenchmarkReport run_benchmark(vector<FlowConfig>& configs, int warmup_iters, int num_gpus) {
    BenchmarkReport report;

    std::vector<std::unique_ptr<Flow>> flows;
    for(auto& c : configs) {
        flows.push_back(to_flow(c));
    }

    vector<vector<Flow*>> per_gpu(num_gpus);
    for (auto& f : flows) {
        per_gpu[f->desc.src_gpu].push_back(f.get());
    }

    double epoch_us = 0;
    SyncBarrier start_barrier(num_gpus, [&epoch_us] { epoch_us = us_now(); });

    vector<std::thread> workers;
    for (int g = 0; g < num_gpus; ++g) {
        workers.emplace_back(gpu_worker, g, per_gpu[g], warmup_iters,
                             std::ref(start_barrier), std::cref(epoch_us));
    }
    for (auto& t : workers) t.join();

    /* samples carry absolute host timestamps; shift the whole run so the
     * earliest launch across all flows is t=0 */
    double min_launch_us = std::numeric_limits<double>::max();
    for (auto& f : flows)
        for (auto& s : f->samples)
            min_launch_us = std::min(min_launch_us, s.launch_us);
    if (min_launch_us != std::numeric_limits<double>::max())
        for (auto& f : flows)
            for (auto& s : f->samples)
                s.launch_us -= min_launch_us;

    for (size_t i = 0; i < flows.size(); ++i) {
        FlowResult res = {};
        res.src = flows[i]->desc.src_gpu;
        res.dst = flows[i]->desc.dst_gpu;
        res.type = flows[i]->type;
        res.buffer_size = configs[i].buffer_size;
        res.iov_size = configs[i].iov_size;
		res.sm_count = configs[i].sm_count;
        res.start_us = configs[i].start_us;
        res.gap_us = configs[i].gap_us;
        res.samples = flows[i]->samples;
        report.flows.push_back(res);
    }

    return report;
}

//__device__ __forceinline__ void calibrate(BufferPair *iov, size_t size,

void calibrate_gpu_globaltimer_drift(int64_t offset) {

}

int main(int argc, char* argv[]) {
    Pattern pattern = ALLTOALL;
    size_t size = 1ULL << 30;
    int num_iters = 1;
    int warmup_iters = 0;
    int num_gpus = 3;
	FlowAlgo flow_type = SM_UNROLL;
    int target_gpu = 0;
    int send_target = 1;
    string pattern_str = "alltoall";
    size_t kernel_iter = 1;
    int sm_count = get_sm_count(0);
    double start_us = 0.0;
    double gap_us = 0.0;

    string input_json = "";
    bool output_json = false;
    string perfetto_out = "";

    static struct option long_options[] = {
        {"pattern", required_argument, 0, 'p'},
        {"size", required_argument, 0, 's'},
        {"iters", required_argument, 0, 'i'},
        {"warmup", required_argument, 0, 'w'},
        {"gpus", required_argument, 0, 'g'},
        {"algo", required_argument, 0, 'a'},
        {"target", required_argument, 0, 't'},
        {"send_target", required_argument, 0, 'e'},
        {"kiter", required_argument, 0, 'k'},
        {"config-json", required_argument, 0, 'j'},
        {"output-json", no_argument, 0, 'o'},
        {"output-perfetto", required_argument, 0, 'P'},
		{"count-sm", required_argument, 0, 'c'},
        {"start-us", required_argument, 0, 'S'},
        {"gap-us", required_argument, 0, 'G'},
        {0, 0, 0, 0}
    };

	char *target_lat_str = getenv("LAT");
	if (target_lat_str) {
		var_target_lat = std::atoi(target_lat_str);
	}

    int opt;
    while ((opt = getopt_long(argc, argv, "", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'p':
                pattern_str = optarg;
                if (pattern_str == "incast") pattern = INCAST;
                else if (pattern_str == "outcast") pattern = OUTCAST;
                else if (pattern_str == "alltoall") pattern = ALLTOALL;
                else if (pattern_str == "ring") pattern = RING;
                else if (pattern_str == "incast_with_send") pattern = INCAST_WITH_SEND;
                else if (pattern_str == "incast_with_send_rev") pattern = INCAST_WITH_SEND_REV;
                else if (pattern_str == "single") pattern = SINGLE;
                else cerr << "Invalid pattern" << endl;
                break;
            case 's': size = parse_size(optarg); break;
            case 'i': num_iters = stoi(optarg); break;
            case 'w': warmup_iters = stoi(optarg); break;
            case 'g': num_gpus = stoi(optarg); break;
            case 'a': flow_type = parse_algo_string(optarg); break;
            case 't': target_gpu = stoi(optarg); break;
            case 'e': send_target = stoi(optarg); break;
            case 'k': kernel_iter = stoi(optarg); break;
            case 'j': input_json = optarg; break;
            case 'o': output_json = true; break;
            case 'P': perfetto_out = optarg; break;
			case 'c': sm_count = stoi(optarg); break;
            case 'S': start_us = stod(optarg); break;
            case 'G': gap_us = stod(optarg); break;
        }
    }

    vector<FlowConfig> configs;
    if (input_json.empty()) {
        configs = generate_flows(pattern, num_gpus, flow_type, size, kernel_iter, sm_count,
                                 start_us, num_iters, gap_us, target_gpu, send_target);
    } else {
        configs = from_json(input_json, start_us, num_iters, gap_us);
        /* size the worker pool to whatever the config references */
        int max_gpu = 0;
        for (auto& c : configs) {
            max_gpu = std::max({max_gpu, c.desc.src_gpu, c.desc.dst_gpu});
        }
        num_gpus = std::max(num_gpus, max_gpu + 1);
    }

    int device_count;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
    if (num_gpus > device_count) {
        cerr << "Requested " << num_gpus << " GPUs, but only " << device_count << " available." << endl;
        return 0;
    }

	if (!output_json)
		print_gpu_info(num_gpus);

    enable_p2p(num_gpus);

    BenchmarkReport report = run_benchmark(configs, warmup_iters, num_gpus);
    if (output_json) {
        std::cout << report.as_json().dump(4) << std::endl;
    } else {
        report.pretty_print();
    }

    if (!perfetto_out.empty()) {
        std::ofstream f(perfetto_out);
        if (!f.is_open()) {
            cerr << "Failed to open perfetto output file: " << perfetto_out << endl;
            return 1;
        }
        f << report.as_perfetto_json().dump(4) << std::endl;
        if (!output_json)
            cout << "Perfetto trace written to " << perfetto_out << endl;
    }

    return 0;
}
