NVFLAGS = -arch=compute_90,compute_80 --gpu-code=sm_90,sm_80  -g -O2 -lineinfo -lcuda -code=compute_90,compute_80
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
