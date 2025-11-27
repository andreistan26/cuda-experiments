#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

#include "errors.h"

void enable_p2p(int num_gpus) {
    for (int i = 0; i < num_gpus; ++i) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        for (int j = 0; j < num_gpus; ++j) {
            if (i != j) {
                int can_access;
                CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(j, 0));
                } else {
					fprintf(stderr, "Could not enable P2P on device %d -> %d\n", i, j);
                }
            }
        }
    }
}

int get_sm_count(int dev_idx) {
	cudaDeviceProp prop;
	CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, dev_idx));

	return prop.multiProcessorCount;
}

void print_gpu_info(int num_gpus) {
	for (int id = 0; id < num_gpus; id++) {
		cudaDeviceProp prop;
		CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, id));
		printf("Device %s %d: %04x:%02x:%02x\n", prop.name, id, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
	}
}

/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

struct l2flush
{
  __forceinline__ l2flush()
  {
    int dev_id{};
    CHECK_CUDA_ERROR(cudaGetDevice(&dev_id));
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id));
    if (m_l2_size > 0)
    {
      void *buffer = m_l2_buffer;
      CHECK_CUDA_ERROR(cudaMalloc(&buffer, static_cast<std::size_t>(m_l2_size)));
      m_l2_buffer = reinterpret_cast<int *>(buffer);
    }
  }

  __forceinline__ ~l2flush()
  {
    if (m_l2_buffer)
    {
      CHECK_CUDA_ERROR(cudaFree(m_l2_buffer));
    }
  }

  __forceinline__ void flush(cudaStream_t stream)
  {
    if (m_l2_size > 0)
    {
      CHECK_CUDA_ERROR(
        cudaMemsetAsync(m_l2_buffer, 0, static_cast<std::size_t>(m_l2_size), stream));
    }
  }

  __forceinline__ void flush_sync(cudaStream_t stream)
  {
    if (m_l2_size > 0)
    {
      CHECK_CUDA_ERROR(
        cudaMemset(m_l2_buffer, 0, static_cast<std::size_t>(m_l2_size)));
    }
  }

private:
  int m_l2_size{};
  int *m_l2_buffer{};
};
