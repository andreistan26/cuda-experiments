NVFLAGS = -arch=compute_90 --gpu-code=sm_90  -g -O2 -lineinfo -lcuda -code=compute_90
CFLAGS = -Iinclude 

all: flowgen latency pingpong

flowgen: flowgen.cu
	nvcc $(CFLAGS) -std=c++17 $(NVFLAGS) $^ -o $@

latency: latency.cu
	nvcc $(CFLAGS) -std=c++20 $(NVFLAGS) $^ -o $@

latency.o: latency.cu
	nvcc $(CFLAGS) $(NVFLAGS) $^ -o $@ -cubin

pingpong: pingpong.cu
	nvcc $(CFLAGS) -std=c++20 $(NVFLAGS) $^ -o $@ 

tma: tma.cu
	nvcc $(CFLAGS) -std=c++20 $(NVFLAGS) $^ -o $@ 

clean:
	rm -f flowgen latency
