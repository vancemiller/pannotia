NVCC=nvcc
NVCCFLAGS=-std=c++11 -Wno-deprecated-gpu-targets
INC=./inc

SRCS = $(wildcard *.cu)
TARGETS = $(patsubst %.cu,%,$(SRCS))

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< -I$(INC)

clean:
	rm -f $(TARGETS)

