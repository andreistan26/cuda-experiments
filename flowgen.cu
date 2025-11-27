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

struct BufferPair {
    void* src;
    void* dest;
    size_t size;
};

/* taken from nvbandwidth */ 
template <typename T>
__global__ void striding_memcpy_kernel(BufferPair *iov, size_t size) {
    size_t from = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int totalThreadCount = blockDim.x * gridDim.x;

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
__global__ void tma_memcpy(BufferPair *iov, size_t size) {

}

struct FlowDescriptor {
    int src_gpu;
    int dst_gpu;
};

enum FlowAlgo {
    SM_SIMPLE,
    SM_UNROLL,
    TMA,
    CE
};

std::string algo_to_string(FlowAlgo algo) {
    switch(algo) {
        case SM_SIMPLE: return "SM_SIMPLE";
        case SM_UNROLL: return "SM_UNROLL";
        case TMA: return "TMA";
        case CE: return "CE";
        default: return "UNKNOWN";
    }
}

FlowAlgo parse_algo_string(const std::string& str) {
    if (str == "SM_SIMPLE" || str == "simple") return SM_SIMPLE;
    if (str == "SM_UNROLL" || str == "sm") return SM_UNROLL;
    if (str == "TMA") return TMA;
    if (str == "CE") return CE;
    std::cerr << "Unknown Algo String: " << str << ", defaulting to CE" << std::endl;
    return CE;
}

struct FlowConfig {
    FlowDescriptor desc;
    size_t buffer_size;
    size_t iov_size;
    int sm_count;
    FlowAlgo type;
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

    Flow(FlowConfig config)
        : desc(config.desc), iov_size(config.iov_size), sm_count(config.sm_count), type(config.type) {
        auto buffer_size = config.buffer_size;
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

    void launch_copy_kernel() {
        CHECK_CUDA_ERROR(cudaSetDevice(desc.src_gpu));
        CHECK_CUDA_ERROR(cudaEventRecord(start_event, stream));
        __launch_copy_kernel();
        CHECK_CUDA_ERROR(cudaEventRecord(stop_event, stream));
    }

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

std::unique_ptr<Flow> to_flow(FlowConfig cfg) {
    switch (cfg.type) {
    case CE:
        return std::make_unique<CEFlow>(cfg);
    case SM_SIMPLE:
        return std::make_unique<SMSimpleFlow>(cfg);
    case SM_UNROLL:
        return std::make_unique<SMUnrolledFlow>(cfg);
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
        configs.push_back(c);
    }
    return configs;
}

std::vector<FlowConfig> from_json(const string &filename) {
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
        
        configs.push_back(c);
    }
    return configs;
}
struct FlowResult {
    int src, dst;
    FlowAlgo type;
    size_t bytes;
    size_t iov_size;
    double latency_ms;
	int sm_count;

    double get_latency_us() const { return latency_ms * 1000.0; }
        
    double get_bandwidth_GBs() const {
        if (latency_ms <= 0.0001) return 0.0;
        return (static_cast<double>(bytes) / 1.0e9) / (latency_ms / 1000.0);
    }

    json to_json() {
        return json{
            {"src", src},
            {"dst", dst},
            {"type", algo_to_string(type)},
            {"total_size", bytes},
            {"iov_size", iov_size},
            {"sm_count", sm_count},
            {"total_latency_us", get_latency_us()},
            {"bandwidth_GBs", get_bandwidth_GBs()}
        };
    }
};

struct BenchmarkReport {
    std::vector<FlowResult> flows;
    int num_iters;

    json as_json() const {
        double avg_lat = 0;
        if(!flows.empty()) {
            for(const auto& f : flows) avg_lat += f.get_latency_us();
            avg_lat /= flows.size();
        }

        json j;
        j["average_latency_us"] = avg_lat;
        j["num_iters"] = num_iters;

        std::vector<json> flows_json;
        for(auto f : flows) flows_json.push_back(f.to_json());
        j["flows"] = flows_json;
        return j;
    }

    void pretty_print() const {
        if (flows.empty()) {
            cout << "No flows measured." << endl;
            return;
        }

        double avg_latency_us = 0;
        for(const auto& f : flows) avg_latency_us += f.get_latency_us();
        avg_latency_us /= flows.size();

        cout << fixed << setprecision(2);
        cout << "Average Latency per Flow: " << avg_latency_us << " us" << endl;
        cout << "Per-flow details:" << endl;

        for (const auto& f : flows) {
            string bw_str;
            if (f.latency_ms <= 0.0001) {
                bw_str = "N/A";
            } else {
                bw_str = to_string(f.get_bandwidth_GBs()) + " GB/s";
            }

            cout << "Flow " << f.src << " -> " << f.dst 
                 << " (" << algo_to_string(f.type) << ")"
                 << ": Latency " << f.get_latency_us() << " us, "
                 << "Bandwidth " << bw_str << endl;
        }
    }
};

BenchmarkReport run_benchmark(
    vector<FlowConfig>& configs, 
    int num_iters, 
    int warmup_iters, 
    int num_gpus
) {
    BenchmarkReport report;
    report.num_iters = num_iters;

	// This can be inside Flow but that would make one memset for each flow
	// is there are multiple flows for each GPU that are overlapped that would
	// run for both of them
	std::vector<std::unique_ptr<l2flush>> l2_flushers;

    std::vector<std::unique_ptr<Flow>> flows;
    for(auto& c : configs) {
        flows.push_back(to_flow(c));
    }

	for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
		l2_flushers.push_back(std::make_unique<l2flush>());
	}

    //vector<cudaEvent_t> iter_sync_events(std::max(num_iters, warmup_iters));
    //for (int i = 0; i < std::max(num_iters, warmup_iters); i++) {
    //    CHECK_CUDA_ERROR(cudaSetDevice(sync_device));
    //    CHECK_CUDA_ERROR(cudaEventCreate(&iter_sync_events[i], cudaEventDisableTiming));
    //}

    //CHECK_CUDA_ERROR(cudaSetDevice(sync_device));
    //cudaStream_t control_stream;
    //CHECK_CUDA_ERROR(cudaStreamCreate(&control_stream));

    /* warmup */
    for (int iter = 0; iter < warmup_iters; ++iter) {
		// flush L2 cache before each iter	
		for (int i = 0; i < num_gpus; i++) {
			cudaSetDevice(i);
			l2_flushers[i]->flush_sync(0);
		}

        for (auto &flow : flows) {
            //CHECK_CUDA_ERROR(cudaStreamWaitEvent(flow->stream, iter_sync_events[iter]));
            flow->__launch_copy_kernel();
        }
        CHECK_CUDA_ERROR(cudaSetDevice(sync_device));
        //CHECK_CUDA_ERROR(cudaEventRecord(iter_sync_events[iter], control_stream));

        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA_ERROR(cudaSetDevice(i));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    /* cleanup syncs */
    //CHECK_CUDA_ERROR(cudaSetDevice(sync_device));
    //for (int i = 0; i < warmup_iters; i++) {
    //    CHECK_CUDA_ERROR(cudaEventDestroy(iter_sync_events[i]));
    //    CHECK_CUDA_ERROR(cudaEventCreate(&iter_sync_events[i], cudaEventDisableTiming));
    //}

    vector<float> total_ms(flows.size(), 0.0f);

    for (int iter = 0; iter < num_iters; ++iter) {
		// flush L2 cache before each iter	
		for (int i = 0; i < num_gpus; i++) {
			cudaSetDevice(i);
			l2_flushers[i]->flush_sync(0);
		}

        for (auto &flow : flows) {
            //CHECK_CUDA_ERROR(cudaStreamWaitEvent(flow->stream, iter_sync_events[iter]));
            flow->launch_copy_kernel();
        }
        CHECK_CUDA_ERROR(cudaSetDevice(sync_device));
        //CHECK_CUDA_ERROR(cudaEventRecord(iter_sync_events[iter], control_stream));

        for (int i = 0; i < num_gpus; ++i) {
            CHECK_CUDA_ERROR(cudaSetDevice(i));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        for (size_t i = 0; i < flows.size(); i++) {
            cudaSetDevice(flows[i]->desc.src_gpu);
            float ms;
            cudaEventElapsedTime(&ms, flows[i]->start_event, flows[i]->stop_event);
            total_ms[i] += ms / (float)flows[i]->iov_size;
        }
    }

    for (size_t i = 0; i < flows.size(); ++i) {
        FlowResult res = {};
        res.src = flows[i]->desc.src_gpu;
        res.dst = flows[i]->desc.dst_gpu;
        res.type = flows[i]->type;
        res.bytes = configs[i].buffer_size;
        res.iov_size = configs[i].iov_size;
		res.sm_count = configs[i].sm_count;
        res.latency_ms = total_ms[i] / num_iters;
        report.flows.push_back(res);
    }

    CHECK_CUDA_ERROR(cudaSetDevice(sync_device));
    //CHECK_CUDA_ERROR(cudaStreamDestroy(control_stream));

    return report;
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

    string input_json = "";
    bool output_json = false;

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
		{"count-sm", required_argument, 0, 'c'},
        {0, 0, 0, 0}
    };

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
			case 'c': sm_count = stoi(optarg); break;
        }
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

    vector<FlowConfig> configs;
    if (input_json.empty()) {
        configs = generate_flows(pattern, num_gpus, flow_type, size, kernel_iter, sm_count, target_gpu, send_target);
    } else {
        configs = from_json(input_json);
    }

    BenchmarkReport report = run_benchmark(configs, num_iters, warmup_iters, num_gpus);
    if (output_json) {
        std::cout << report.as_json().dump(4) << std::endl;
    } else {
        report.pretty_print();
    }

    return 0;
}
