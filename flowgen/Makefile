NVCC      ?= nvcc
NVCCFLAGS ?= -std=c++17 -O2 -Iinclude
LDFLAGS   ?= -lcuda

GENCODE   ?= -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_90,code=sm_90

TARGET := flowgen
SRC    := flowgen.cu
HDRS   := $(wildcard include/*.h)

all: $(TARGET)

$(TARGET): $(SRC) $(HDRS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
