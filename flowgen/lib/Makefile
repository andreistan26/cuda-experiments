CXX ?= g++
CUDA_HOME ?= /usr/local/cuda
CXXFLAGS ?= -std=c++17 -O2 -fPIC -I../../include -I$(CUDA_HOME)/include
LDFLAGS ?= -shared -L$(CUDA_HOME)/lib64 -lcudart -lcuda -ldl

TARGET := libcc_ce.so
SRC := cc_ce.hpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) -x c++ $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
