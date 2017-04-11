include ../../common/make.config

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include -g

backprop_cuda: backprop_cuda.cu backprop.h
	$(NVCC) $(NVCC_FLAGS) backprop_cuda.cu -o backprop_cuda

clean:
	rm -f *.o *~ backprop_cuda
