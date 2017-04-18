include ../../common/make.config

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

C_C = gcc
OMP_LIB = -lgomp
OMP_FLAG = -fopenmp

CUD_C = $(CUDA_DIR)/bin/nvcc
# OMP_FLAG = 	-Xcompiler paste_one_here
CUDA_FLAG = -G -g

# link objects (binaries) together
a.out:		main.o \
			./util/num/num.o \
			./util/device/device.o
	$(CUD_C)	$(KERNEL_DIM) main.o \
			./util/num/num.o \
			./util/device/device.o \
			-lm \
			-L/usr/local/cuda/lib64 \
			-lcuda -lcudart \
			$(OMP_LIB) \
			-o lavaMD

# compile function files into objects (binaries)
main.o:		main.h \
			main.cu \
			./util/num/num.h \
			./util/num/num.c \
			./util/device/device.h \
			./util/device/device.cu
	$(CUD_C)	$(KERNEL_DIM) $(OUTPUT) main.cu \
			-c \
			-o main.o \
			-O3

./util/num/num.o:	./util/num/num.h \
					./util/num/num.c
	$(C_C)			./util/num/num.c \
					-c \
					-o ./util/num/num.o \
					-O3

./util/device/device.o:	./util/device/device.h \
						./util/device/device.cu
	$(CUD_C)			./util/device/device.cu \
						-c \
						-o ./util/device/device.o \
						-O3

# delete all object and executable files
clean:
	rm -f	*.o \
		./kernel/*.o \
		./util/num/*.o \
		./util/device/*.o \
		lavaMD
