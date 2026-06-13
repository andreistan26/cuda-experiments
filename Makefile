NVFLAGS = -g -O2 -lineinfo -lcuda -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 \

CFLAGS = -Iinclude 

all: flowgen latency pingpong burst

burst: burst.cu
	nvcc $(CFLAGS) -std=c++20 $(NVFLAGS) $^ -o $@

latency: latency.cu
	nvcc $(CFLAGS) -std=c++20 $(NVFLAGS) $^ -o $@

latency.o: latency.cu
	nvcc $(CFLAGS) $(NVFLAGS) $^ -o $@ -cubin

pingpong: pingpong.cu
	nvcc $(CFLAGS) -std=c++20 $(NVFLAGS) $^ -o $@ 

clean:
	rm -f flowgen latency
