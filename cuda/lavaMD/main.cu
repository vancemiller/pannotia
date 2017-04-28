#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "./util/num/num.h"
#include "./main.h"
#include "./kernel/kernel_gpu_cuda.cu"
#include "helper_cuda.h"

#include "../timing.h"

int main(int argc, char *argv[]) {
  float time_pre = 0;
  float time_post = 0;
  float time_serial = 0;
  float time_copy_in = 0;
  float time_copy_out = 0;
  float time_kernel = 0;
  float time_malloc = 0;
  float time_free = 0;

  printf("thread block size of kernel = %d \n", NUMBER_THREADS);

  // counters
  int i, j, k, l, m, n;

  // system memory
  par_str par_cpu;
  dim_str dim_cpu;
  box_str* box_cpu;
  FOUR_VECTOR* rv_cpu;
  double* qv_cpu;
  FOUR_VECTOR* fv_cpu;
  int nh;
  bool unified = false;


  //	CHECK INPUT ARGUMENTS

  // assing default values
  dim_cpu.boxes1d_arg = 1;

  // go through arguments
  for (dim_cpu.cur_arg = 1; dim_cpu.cur_arg < argc; dim_cpu.cur_arg++) {
    // check if -boxes1d
    if (strcmp(argv[dim_cpu.cur_arg], "-u") == 0) {
      unified = true;
    } else if (strcmp(argv[dim_cpu.cur_arg], "-boxes1d") == 0) {
      // check if value provided
      if (argc >= dim_cpu.cur_arg + 1) {
        // check if value is a number
        if (isInteger(argv[dim_cpu.cur_arg + 1]) == 1) {
          dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg + 1]);
          if (dim_cpu.boxes1d_arg < 0) {
            printf("ERROR: Wrong value to -boxes1d parameter, cannot be <=0\n");
            return 0;
          }
          dim_cpu.cur_arg = dim_cpu.cur_arg + 1;
        } else {
          // value is not a number
          printf("ERROR: Value to -boxes1d parameter in not a number\n");
          return 0;
        }
      } else {
        // value not provided
        printf("ERROR: Missing value to -boxes1d parameter\n");
        return 0;
      }
    } else {
      // unknown
      printf("ERROR: Unknown parameter\n");
      return 0;
    }
  }

  // Print configuration
  printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);


  //	INPUTS

  par_cpu.alpha = 0.5;

  //	DIMENSIONS

  // total number of boxes
  dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

  // how many particles space has in each direction
  dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
  dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
  dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(double);

  // box array
  dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

  //	SYSTEM MEMORY

  //	BOX

  TIMESTAMP(t0);
  // allocate boxes
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&box_cpu, dim_cpu.box_mem));
    checkCudaErrors(cudaMallocManaged(&rv_cpu, dim_cpu.space_mem));
    checkCudaErrors(cudaMallocManaged(&qv_cpu, dim_cpu.space_mem2));
    checkCudaErrors(cudaMallocManaged(&fv_cpu, dim_cpu.space_mem));
  } else {
    box_cpu = (box_str*) malloc(dim_cpu.box_mem);
    rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
    qv_cpu = (double*) malloc(dim_cpu.space_mem2);
    fv_cpu = (FOUR_VECTOR*) malloc(dim_cpu.space_mem);
  }
  TIMESTAMP(t1);
  time_malloc += ELAPSED(t0, t1);

  // initialize number of home boxes
  nh = 0;

  // home boxes in z direction
  for (i = 0; i < dim_cpu.boxes1d_arg; i++) {
    // home boxes in y direction
    for (j = 0; j < dim_cpu.boxes1d_arg; j++) {
      // home boxes in x direction
      for (k = 0; k < dim_cpu.boxes1d_arg; k++) {

        // current home box
        box_cpu[nh].x = k;
        box_cpu[nh].y = j;
        box_cpu[nh].z = i;
        box_cpu[nh].number = nh;
        box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

        // initialize number of neighbor boxes
        box_cpu[nh].nn = 0;

        // neighbor boxes in z direction
        for (l = -1; l < 2; l++) {
          // neighbor boxes in y direction
          for (m = -1; m < 2; m++) {
            // neighbor boxes in x direction
            for (n = -1; n < 2; n++) {

              // check if (this neighbor exists) and (it is not the same as home box)
              if ((((i + l) >= 0 && (j + m) >= 0 && (k + n) >= 0) == true
                    && ((i + l) < dim_cpu.boxes1d_arg && (j + m) < dim_cpu.boxes1d_arg
                      && (k + n) < dim_cpu.boxes1d_arg) == true)
                  && (l == 0 && m == 0 && n == 0) == false) {

                // current neighbor box
                box_cpu[nh].nei[box_cpu[nh].nn].x = (k + n);
                box_cpu[nh].nei[box_cpu[nh].nn].y = (j + m);
                box_cpu[nh].nei[box_cpu[nh].nn].z = (i + l);
                box_cpu[nh].nei[box_cpu[nh].nn].number = (box_cpu[nh].nei[box_cpu[nh].nn].z
                    * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg)
                  + (box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg)
                  + box_cpu[nh].nei[box_cpu[nh].nn].x;
                box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number
                  * NUMBER_PAR_PER_BOX;

                // increment neighbor box
                box_cpu[nh].nn = box_cpu[nh].nn + 1;

              }

            } // neighbor boxes in x direction
          } // neighbor boxes in y direction
        } // neighbor boxes in z direction

        // increment home box
        nh = nh + 1;

      } // home boxes in x direction
    } // home boxes in y direction
  } // home boxes in z direction

  //	PARAMETERS, DISTANCE, CHARGE AND FORCE

  // random generator seed set to random value - time in this case
  srand (time(NULL));

  // input (distances)
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    rv_cpu[i].v = (rand() % 10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
    rv_cpu[i].x = (rand() % 10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
    rv_cpu[i].y = (rand() % 10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
    rv_cpu[i].z = (rand() % 10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
  }

  // input (charge)
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    qv_cpu[i] = (rand() % 10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
  }

  // output (forces)
  for (i = 0; i < dim_cpu.space_elem; i = i + 1) {
    fv_cpu[i].v = 0;								// set to 0, because kernels keeps adding to initial value
    fv_cpu[i].x = 0;								// set to 0, because kernels keeps adding to initial value
    fv_cpu[i].y = 0;								// set to 0, because kernels keeps adding to initial value
    fv_cpu[i].z = 0;								// set to 0, because kernels keeps adding to initial value
  }

  //	KERNEL

  box_str* d_box_gpu;
	FOUR_VECTOR* d_rv_gpu;
	double* d_qv_gpu;
	FOUR_VECTOR* d_fv_gpu;

	dim3 threads;
	dim3 blocks;

	blocks.x = dim_cpu.number_boxes;
	blocks.y = 1;
	threads.x = NUMBER_THREADS;											// define the number of threads in the block
	threads.y = 1;

  TIMESTAMP(t2);
  time_pre += ELAPSED(t1, t2);

  if (!unified) {
    cudaMalloc(	(void **)&d_box_gpu,
          dim_cpu.box_mem);

    cudaMalloc(	(void **)&d_rv_gpu,
          dim_cpu.space_mem);

    cudaMalloc(	(void **)&d_qv_gpu,
          dim_cpu.space_mem2);

    cudaMalloc(	(void **)&d_fv_gpu,
          dim_cpu.space_mem);
  }
  TIMESTAMP(t3);
  time_malloc += ELAPSED(t2, t3);

  if (unified) {
    d_box_gpu = box_cpu;
    d_rv_gpu = rv_cpu;
    d_qv_gpu = qv_cpu;
    d_fv_gpu = fv_cpu;
  } else {
    checkCudaErrors(cudaMemcpy(	d_box_gpu,
				box_cpu,
				dim_cpu.box_mem,
				cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(	d_rv_gpu,
				rv_cpu,
				dim_cpu.space_mem,
				cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(	d_qv_gpu,
				qv_cpu,
				dim_cpu.space_mem2,
				cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(	d_fv_gpu,
				fv_cpu,
				dim_cpu.space_mem,
				cudaMemcpyHostToDevice));
  }

  TIMESTAMP(t4);
  time_copy_in += ELAPSED(t3, t4);

	// launch kernel - all boxes
	kernel_gpu_cuda<<<blocks, threads>>>(	par_cpu,
											dim_cpu,
											d_box_gpu,
											d_rv_gpu,
											d_qv_gpu,
											d_fv_gpu);

	checkCudaErrors(cudaThreadSynchronize());
  TIMESTAMP(t5);
  time_kernel += ELAPSED(t4, t5);

  if (!unified) {
    checkCudaErrors(cudaMemcpy(	fv_cpu,
				d_fv_gpu,
				dim_cpu.space_mem,
				cudaMemcpyDeviceToHost));
  }
  TIMESTAMP(t6);
  time_copy_out += ELAPSED(t5, t6);

  if (!unified) {
    checkCudaErrors(cudaFree(d_rv_gpu));
    checkCudaErrors(cudaFree(d_qv_gpu));
    checkCudaErrors(cudaFree(d_fv_gpu));
    checkCudaErrors(cudaFree(d_box_gpu));
  }
  TIMESTAMP(t7);
  time_free += ELAPSED(t6, t7);

  // dump results
#ifdef OUTPUT
  FILE *fptr;
  fptr = fopen("result.txt", "w");
  for(i=0; i<dim_cpu.space_elem; i=i+1) {
    fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
  }
  fclose(fptr);
#endif

  TIMESTAMP(t8);
  time_post += ELAPSED(t7, t8);

  if (unified) {
    checkCudaErrors(cudaFree(rv_cpu));
    checkCudaErrors(cudaFree(qv_cpu));
    checkCudaErrors(cudaFree(fv_cpu));
    checkCudaErrors(cudaFree(box_cpu));
  } else {
    free(rv_cpu);
    free(qv_cpu);
    free(fv_cpu);
    free(box_cpu);
  }

  TIMESTAMP(t9);
  time_free += ELAPSED(t8, t9);

  printf("====Timing info====\n");
  printf("time malloc = %f ms\n", time_malloc);
  printf("time pre = %f ms\n", time_pre);
  printf("time copyIn = %f ms\n", time_copy_in);
  printf("time kernel = %f ms\n", time_kernel);
  printf("time serial = %f ms\n", time_serial);
  printf("time copyOut = %f ms\n", time_copy_out);
  printf("time post = %f ms\n", time_post);
  printf("time free = %f ms\n", time_free);
  printf("time end-to-end = %f ms\n", ELAPSED(t0, t9));
  exit(EXIT_SUCCESS);
}

