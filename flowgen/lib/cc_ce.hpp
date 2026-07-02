#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <deque>
#include <dlfcn.h>
#include <errors.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using benchclock = std::chrono::high_resolution_clock;
using timestamp_ns = size_t;

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

static bool nsys = getenv("NSYS") != nullptr;
static bool debug = getenv("DEBUG") != nullptr;

static timestamp_ns ns_now() {
  return std::chrono::duration<size_t, std::nano>(
             benchclock::now().time_since_epoch())
      .count();
}

// Driver layer calls cu* should be preferred but this is a good start
using RealMemcpyAsync = cudaError_t(CUDARTAPI *)(void *, const void *, size_t,
                                                 cudaMemcpyKind, cudaStream_t);
using RealMemcpyPeerAsync = cudaError_t(CUDARTAPI *)(void *, int, const void *,
                                                     int, size_t, cudaStream_t);
using RealMemcpyBatchAsync = cudaError_t(CUDARTAPI *)(
    void **, void **, size_t *, size_t, cudaMemcpyAttributes *, size_t *,
    size_t, size_t *, cudaStream_t);

static RealMemcpyAsync real_cudaMemcpyAsync = nullptr;
static RealMemcpyPeerAsync real_cudaMemcpyPeerAsync = nullptr;
static RealMemcpyBatchAsync real_cudaMemcpyBatchAsync = nullptr;

constexpr int kWorkerCount = 8;
constexpr int kRequestPoolSz = 128 * 2;
constexpr size_t kSizeBypassCC = 512 * 1024;

// Keeps the same stream sematics for the caller as if the
// the memcpy runs as a single stream event
struct RequestGate {
  cudaEvent_t ready, reclaim;
  cudaStream_t worker_stream;
  void *done_ptr;
  CUdeviceptr done;
};

struct CopyDescriptor {
  int src_gpu, dst_gpu;
  void *src_data, *dst_data;
  size_t size, offset;
  cudaMemcpyKind kind;
  bool peer;
  cudaStream_t stream;
  RequestGate *gate;
};

class ControllerState {
public:
  ControllerState(int src_gpu, int) : cwnd_(kStartCwndSz) {
    CHECK_CUDA_ERROR(cudaSetDevice(src_gpu));
    for (int i = 0; i < kEventPoolSz; i++) {
      cudaEvent_t ev;
      CHECK_CUDA_ERROR(cudaEventCreate(&ev));
      free_events_.push_back(ev);
    }
  }

  struct InflightChunk {
    cudaEvent_t start, end;
    timestamp_ns query_time;
    size_t size;
  };

  bool can_send() const {
    return inflight_chunks_.size() < maxQueuedChunks &&
           free_events_.size() >= 2;
  }

  double serializationDelayNs(size_t size) const {
    return size / kLinkSpeed_Bpns;
  }

  double targetDelayNs(size_t size) const {
    double base =
        kLaunchOverhead_ns + serializationDelayNs(size) + (nsys ? 3000 : 0);
    static constexpr size_t fs_min_cwnd = 1 * 1024 * 1024;
    static constexpr size_t fs_max_cwnd = 8 * 1024 * 1024;
    static constexpr double fs_range = 10000;
    static constexpr double isqrt_fs_min = 1 / 1024;
    static constexpr double isqrt_fs_max = 2896.30937574;
    static constexpr double alpha = fs_range / (isqrt_fs_min - isqrt_fs_max);
    static constexpr double beta = -alpha * isqrt_fs_max;
    return base + std::max<double>(
                      0, std::min(alpha / std::sqrt(cwnd_) + beta, fs_range));
  }

  void send_chunk(CopyDescriptor &desc) {
    cudaEvent_t start_event = free_events_.front();
    free_events_.pop_front();
    cudaEvent_t end_event = free_events_.front();
    free_events_.pop_front();
    size_t chunk_size = std::min(desc.size - desc.offset, cwnd_);
    CHECK_CUDA_ERROR(cudaEventRecord(start_event, desc.stream));

    if (desc.peer) {
      CHECK_CUDA_ERROR(real_cudaMemcpyPeerAsync(
          (char *)desc.dst_data + desc.offset, desc.dst_gpu,
          (char *)desc.src_data + desc.offset, desc.src_gpu, chunk_size,
          desc.stream));
    } else {
      CHECK_CUDA_ERROR(real_cudaMemcpyAsync((char *)desc.dst_data + desc.offset,
                                            (char *)desc.src_data + desc.offset,
                                            chunk_size, desc.kind,
                                            desc.stream));
    }

    CHECK_CUDA_ERROR(cudaEventRecord(end_event, desc.stream));
    inflight_chunks_.push_back(
        {start_event, end_event,
         ns_now() + (timestamp_ns)targetDelayNs(chunk_size) - 2000, // start querying the event before expected end time
         chunk_size});
    desc.offset += chunk_size;
  }

  void update_on_ack(size_t size, double latency_ns) {
    double target_delay = targetDelayNs(size);
    double actual_delay = latency_ns - 3200;//- serializationDelayNs(size) - 3200;

    if (unlikely(debug))
      std::cout << "ACK latency=" << latency_ns << " delay=" << actual_delay
                << " target=" << target_delay << " size=" << size
                << " cwnd=" << cwnd_ << "\n";

    if (actual_delay > target_delay && !first_call_)
      cwnd_ = std::max<size_t>(
          cwnd_ * std::max<double>(
                      1 - (actual_delay - target_delay) / actual_delay, 0.7),
          kMinCwndSz);
    else
      cwnd_ = std::min(cwnd_ + size * 0.1, (double)kMaxCwndSz);
    first_call_ = false;
  }

  void check_inflight(timestamp_ns now, bool force = false) {
    float ms;

    while (!inflight_chunks_.empty()) {
      auto front = inflight_chunks_.front();
      if (!force && front.query_time > now)
        break;

      cudaError_t err = cudaEventQuery(front.end);
      if (err == cudaErrorNotReady)
        break;

      CHECK_CUDA_ERROR(err);
      inflight_chunks_.pop_front();
      CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, front.start, front.end));
      update_on_ack(front.size, ms * 1e6);
      free_events_.push_back(front.start);
      free_events_.push_back(front.end);
    }
  }

  void drain_inflight() {
    while (!inflight_chunks_.empty()) {
      check_inflight(ns_now(), true);
      if (!inflight_chunks_.empty())
        std::this_thread::yield();
    }
  }

private:
  static constexpr long kLaunchOverhead_ns = 4500;
  static constexpr long kLinkSpeed_Bpns = 400;
  static constexpr long kEventPoolSz = 64 * 2;
  static constexpr size_t kStartCwndSz = 32 * 1024 * 1024;
  static constexpr size_t kMinCwndSz = 512 * 1024;
  static constexpr size_t kMaxCwndSz = 32 * 1024 * 1024;
  static constexpr long maxQueuedChunks = 8;

  std::deque<cudaEvent_t> free_events_;
  std::deque<InflightChunk> inflight_chunks_;
  size_t cwnd_;
  bool first_call_ = true;
};

// Thread running on each GPU and maintins congestion state
// for all destinations
class Worker {
public:
  Worker(int src_gpu, int device_count)
      : src_gpu_(src_gpu), device_count_(device_count) {
    CHECK_CUDA_ERROR(cudaSetDevice(src_gpu_));

    for (int i = 0; i < device_count_; i++)
      controllers_[i] = std::make_unique<ControllerState>(src_gpu_, i);

    pending_.resize(kRequestPoolSz);
    pending_.clear();
    reclaim_gates_.resize(kRequestPoolSz);
    reclaim_gates_.clear();

    for (auto &gate : gates_) {
      CHECK_CUDA_ERROR(
          cudaEventCreateWithFlags(&gate.ready, cudaEventDisableTiming));
      CHECK_CUDA_ERROR(
          cudaEventCreateWithFlags(&gate.reclaim, cudaEventDisableTiming));
      CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&gate.worker_stream,
                                                 cudaStreamNonBlocking));
      CHECK_CUDA_ERROR(cudaMalloc(&gate.done_ptr, sizeof(uint32_t)));
      CHECK_CUDA_ERROR(cudaMemset(gate.done_ptr, 0, sizeof(uint32_t)));
      gate.done = (CUdeviceptr)gate.done_ptr;
      free_gates_.push_back(&gate);
    }
  }

  void start() {
    thread_ = std::thread(&Worker::run, this);
    thread_.detach();
  }

  RequestGate *acquire_gate() {
    while (true) {
      {
        std::lock_guard<std::mutex> lg(gate_mu_);
        reclaim_locked();
        if (!free_gates_.empty()) {
          RequestGate *gate = free_gates_.back();
          free_gates_.pop_back();
          return gate;
        }
      }
      std::this_thread::yield();
    }
  }

  void reclaim_after(RequestGate *gate) {
    std::lock_guard<std::mutex> lg(gate_mu_);
    reclaim_gates_.push_back(gate);
  }

  void release_now(RequestGate *gate) {
    std::lock_guard<std::mutex> lg(gate_mu_);
    free_gates_.push_back(gate);
  }

  void enqueue(CopyDescriptor desc) {
    std::unique_lock<std::mutex> lk(mu_);

    while (pending_.size() == kRequestPoolSz) {
      lk.unlock();
      std::cerr << "Hit max work count" << std::endl;
      std::this_thread::yield();
      lk.lock();
    }

    pending_.push_back(desc);
    lk.unlock();
    cv_.notify_one();
  }

  bool supports_dst(int dst_gpu) const {
    return dst_gpu >= 0 && dst_gpu < device_count_ && controllers_[dst_gpu];
  }

private:
  void reclaim_locked() {
    size_t count = reclaim_gates_.size();

    for (size_t i = 0; i < count; i++) {
      RequestGate *gate = reclaim_gates_.front();
      reclaim_gates_.pop_front();
      cudaError_t err = cudaEventQuery(gate->reclaim);

      if (err == cudaSuccess)
        free_gates_.push_back(gate);
      else if (err == cudaErrorNotReady)
        reclaim_gates_.push_back(gate);
      else
        CHECK_CUDA_ERROR(err);
    }
  }

  CopyDescriptor next_request() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return !pending_.empty(); });
    CopyDescriptor desc = pending_.front();
    pending_.pop_front();
    return desc;
  }

  void run() {
    CHECK_CUDA_ERROR(cudaSetDevice(src_gpu_));

    while (true) {
      CopyDescriptor current_transfer = next_request();
      auto *controller = controllers_[current_transfer.dst_gpu].get();
      CHECK_CUDA_ERROR(cudaStreamWaitEvent(current_transfer.stream,
                                           current_transfer.gate->ready, 0));

      while (current_transfer.offset < current_transfer.size) {
        controller->check_inflight(ns_now());
        if (!controller->can_send()) {
          std::this_thread::yield();
          continue;
        }
        controller->send_chunk(current_transfer);
      }

      CHECK_CU_ERROR(cuStreamWriteValue32((CUstream)current_transfer.stream,
                                          current_transfer.gate->done, 1,
                                          CU_STREAM_WRITE_VALUE_DEFAULT));
      controller->drain_inflight();
    }
  }

  int src_gpu_, device_count_;
  std::array<std::unique_ptr<ControllerState>, kWorkerCount> controllers_;
  std::array<RequestGate, kRequestPoolSz> gates_;
  std::deque<RequestGate *> free_gates_, reclaim_gates_;
  std::mutex gate_mu_;

  std::deque<CopyDescriptor> pending_;
  std::thread thread_;
  std::mutex mu_;
  std::condition_variable cv_;
};

static std::array<Worker *, kWorkerCount> workers;
static int worker_count;

static cudaError_t forward_cudaMemcpyAsync(void *dst, const void *src,
                                           size_t count, cudaMemcpyKind kind,
                                           cudaStream_t stream) {
  if (unlikely(!real_cudaMemcpyAsync))
    return cudaErrorUnknown;
  return real_cudaMemcpyAsync(dst, src, count, kind, stream);
}

static cudaError_t forward_cudaMemcpyPeerAsync(void *dst, int dst_gpu,
                                               const void *src, int src_gpu,
                                               size_t count,
                                               cudaStream_t stream) {
  if (unlikely(!real_cudaMemcpyPeerAsync))
    return cudaErrorUnknown;
  return real_cudaMemcpyPeerAsync(dst, dst_gpu, src, src_gpu, count, stream);
}

static cudaError_t forward_cudaMemcpyBatchAsync(
    void **dsts, void **srcs, size_t *sizes, size_t count,
    cudaMemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs,
    size_t *failIdx, cudaStream_t stream) {
  if (unlikely(!real_cudaMemcpyBatchAsync))
    return cudaErrorUnknown;
  return real_cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attrsIdxs,
                                   numAttrs, failIdx, stream);
}

static bool device_pointer(const void *ptr, cudaPointerAttributes *attr) {
  cudaError_t err = cudaPointerGetAttributes(attr, ptr);

  if (unlikely(err != cudaSuccess)) {
    cudaGetLastError();
    return false;
  }

  return attr->type == cudaMemoryTypeDevice;
}

static bool cc_route_supported(int src_gpu, int dst_gpu) {
  return src_gpu >= 0 && src_gpu < worker_count && workers[src_gpu] &&
         workers[src_gpu]->supports_dst(dst_gpu);
}

static bool cc_compatible_batch_attrs(cudaMemcpyAttributes *attrs,
                                      size_t *attrsIdxs, size_t numAttrs,
                                      size_t count) {
  constexpr unsigned int kSupportedFlags = cudaMemcpyFlagPreferOverlapWithCompute;

  if (unlikely(!attrs || !attrsIdxs || !numAttrs || numAttrs > count ||
               attrsIdxs[0] != 0))
    return false;

  for (size_t i = 0; i < numAttrs; i++) {
    if (unlikely(attrsIdxs[i] >= count ||
                 (i && attrsIdxs[i] <= attrsIdxs[i - 1]) ||
                 attrs[i].srcAccessOrder != cudaMemcpySrcAccessOrderStream ||
                 (attrs[i].flags & ~kSupportedFlags)))
      return false;
  }

  return true;
}

static cudaError_t enqueue_cc_copy(int src_gpu, int dst_gpu, void *dst,
                                   const void *src, size_t count,
                                   cudaMemcpyKind kind, bool peer,
                                   cudaStream_t stream) {
  if (unlikely((peer && !real_cudaMemcpyPeerAsync) ||
               (!peer && !real_cudaMemcpyAsync)))
    return cudaErrorNotSupported;

  if (unlikely(!cc_route_supported(src_gpu, dst_gpu)))
    return cudaErrorNotSupported;

  RequestGate *gate = workers[src_gpu]->acquire_gate();
  CUstream caller_stream = (CUstream)stream;
  CUresult cu_err = cuStreamWriteValue32(caller_stream, gate->done, 0,
                                         CU_STREAM_WRITE_VALUE_DEFAULT);

  if (unlikely(cu_err != CUDA_SUCCESS)) {
    workers[src_gpu]->release_now(gate);
    return cudaErrorUnknown;
  }

  cudaError_t err = cudaEventRecord(gate->ready, stream);
  if (unlikely(err != cudaSuccess)) {
    workers[src_gpu]->release_now(gate);
    return err;
  }

  cu_err = cuStreamWaitValue32(caller_stream, gate->done, 1,
                               CU_STREAM_WAIT_VALUE_EQ);
  if (unlikely(cu_err != CUDA_SUCCESS)) {
    cudaEventRecord(gate->reclaim, stream);
    workers[src_gpu]->reclaim_after(gate);
    return cudaErrorUnknown;
  }

  workers[src_gpu]->enqueue({src_gpu, dst_gpu, (void *)src, dst, count, 0, kind,
                             peer, gate->worker_stream, gate});
  err = cudaEventRecord(gate->reclaim, stream);
  if (unlikely(err != cudaSuccess))
    return err;

  workers[src_gpu]->reclaim_after(gate);
  return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src,
                                                 size_t count,
                                                 cudaMemcpyKind kind,
                                                 cudaStream_t stream) {
  if (unlikely(count < kSizeBypassCC ||
               (kind != cudaMemcpyDeviceToDevice && kind != cudaMemcpyDefault)))
    return forward_cudaMemcpyAsync(dst, src, count, kind, stream);

  cudaPointerAttributes src_attr, dst_attr;
  if (unlikely(!device_pointer(src, &src_attr) ||
               !device_pointer(dst, &dst_attr)))
    return forward_cudaMemcpyAsync(dst, src, count, kind, stream);

  cudaError_t err = enqueue_cc_copy(src_attr.device, dst_attr.device, dst, src,
                                    count, kind, false, stream);
  if (unlikely(err == cudaErrorNotSupported))
    return forward_cudaMemcpyAsync(dst, src, count, kind, stream);
  return err;
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyPeerAsync(void *dst, int dst_gpu,
                                                     const void *src,
                                                     int src_gpu, size_t count,
                                                     cudaStream_t stream) {
  if (unlikely(count < kSizeBypassCC))
    return forward_cudaMemcpyPeerAsync(dst, dst_gpu, src, src_gpu, count,
                                       stream);

  cudaError_t err = enqueue_cc_copy(src_gpu, dst_gpu, dst, src, count,
                                    cudaMemcpyDefault, true, stream);
  if (unlikely(err == cudaErrorNotSupported))
    return forward_cudaMemcpyPeerAsync(dst, dst_gpu, src, src_gpu, count,
                                       stream);
  return err;
}

extern "C" cudaError_t CUDARTAPI cudaMemcpyBatchAsync(
    void **dsts, void **srcs, size_t *sizes, size_t count,
    cudaMemcpyAttributes *attrs, size_t *attrsIdxs, size_t numAttrs,
    size_t *failIdx, cudaStream_t stream) {
  if (unlikely(!count || !dsts || !srcs || !sizes || !stream ||
               !cc_compatible_batch_attrs(attrs, attrsIdxs, numAttrs, count)))
    return forward_cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs,
                                        attrsIdxs, numAttrs, failIdx, stream);

  std::vector<int> src_gpus(count), dst_gpus(count);
  for (size_t i = 0; i < count; i++) {
    cudaPointerAttributes src_attr, dst_attr;
    if (unlikely(!device_pointer(srcs[i], &src_attr) ||
                 !device_pointer(dsts[i], &dst_attr)))
      return forward_cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs,
                                          attrsIdxs, numAttrs, failIdx, stream);

    src_gpus[i] = src_attr.device;
    dst_gpus[i] = dst_attr.device;
    if (sizes[i] >= kSizeBypassCC &&
        unlikely(!cc_route_supported(src_gpus[i], dst_gpus[i])))
      return forward_cudaMemcpyBatchAsync(dsts, srcs, sizes, count, attrs,
                                          attrsIdxs, numAttrs, failIdx, stream);
  }

  for (size_t i = 0; i < count; i++) {
    cudaError_t err;

    if (sizes[i] < kSizeBypassCC) {
      err = forward_cudaMemcpyAsync(dsts[i], srcs[i], sizes[i],
                                    cudaMemcpyDefault, stream);
    } else {
      err = enqueue_cc_copy(src_gpus[i], dst_gpus[i], dsts[i], srcs[i],
                            sizes[i], cudaMemcpyDefault, false, stream);
    }

    if (unlikely(err != cudaSuccess)) {
      if (failIdx)
        *failIdx = i;
      return err;
    }
  }

  return cudaSuccess;
}

__attribute__((constructor)) static void setup() {
  real_cudaMemcpyAsync = (RealMemcpyAsync)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
  real_cudaMemcpyPeerAsync =
      (RealMemcpyPeerAsync)dlsym(RTLD_NEXT, "cudaMemcpyPeerAsync");
  real_cudaMemcpyBatchAsync =
      (RealMemcpyBatchAsync)dlsym(RTLD_NEXT, "cudaMemcpyBatchAsync");
  int max_gpus = 0;

  if (cudaGetDeviceCount(&max_gpus) != cudaSuccess)
    return;

  worker_count = std::min(kWorkerCount, max_gpus);
  for (int i = 0; i < worker_count; i++) {
    workers[i] = new Worker(i, worker_count);
    workers[i]->start();
  }
}
