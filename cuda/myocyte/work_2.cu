//====================================================================================================100
//		DEFINE / INCLUDE
//====================================================================================================100
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include "helper_cuda.h"
#include "kernel_fin_2.cu"
#include "kernel_ecc_2.cu"
#include "kernel_cam_2.cu"
#include "kernel_2.cu"
#include "embedded_fehlberg_7_8_2.cu"
#include "solver_2.cu"

#include "../timing.h"

int work_2(int xmax, int workload, bool unified) {

  float time_pre = 0;
  float time_post = 0;
  float time_serial = 0;
  float time_copy_in = 0;
  float time_copy_out = 0;
  float time_kernel = 0;
  float time_malloc = 0;
  float time_free = 0;

  //============================================================60
  //		COUNTERS, POINTERS
  //============================================================60

  long memory;
  int i;
  int pointer;

  //============================================================60
  //		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
  //============================================================60

  float* y;
  float* d_y;
  long y_mem;

  float* x;
  float* d_x;
  long x_mem;

  float* params;
  float* d_params;
  int params_mem;

  //============================================================60
  //		TEMPORARY SOLVER VARIABLES
  //============================================================60

  float* d_com;
  int com_mem;

  float* d_err;
  int err_mem;

  float* d_scale;
  int scale_mem;

  float* d_yy;
  int yy_mem;

  float* d_initvalu_temp;
  int initvalu_temp_mem;

  float* d_finavalu_temp;
  int finavalu_temp_mem;

  //============================================================60
  //		CUDA KERNELS EXECUTION PARAMETERS
  //============================================================60

  dim3 threads;
  dim3 blocks;
  int blocks_x;

  //================================================================================80
  // 	ALLOCATE MEMORY
  //================================================================================80

  //============================================================60
  //		MEMORY CHECK
  //============================================================60

  TIMESTAMP(t0);

  memory = workload * (xmax + 1) * EQUATIONS * 4;
  if (memory > 1000000000) {
    printf(
        "ERROR: trying to allocate more than 1.0GB of memory, decrease workload and span parameters or change memory parameter\n");
    return 0;
  }

  //============================================================60
  // 	ALLOCATE ARRAYS
  //============================================================60

  //========================================40
  //		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
  //========================================40

  y_mem = workload * (xmax + 1) * EQUATIONS * sizeof(float);
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&y, y_mem));
  } else {
    y = (float *) malloc(y_mem);
    checkCudaErrors(cudaMalloc((void **) &d_y, y_mem));
  }

  x_mem = workload * (xmax + 1) * sizeof(float);
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&x, x_mem));
  } else {
    x = (float *) malloc(x_mem);
    checkCudaErrors(cudaMalloc((void **) &d_x, x_mem));
  }

  params_mem = workload * PARAMETERS * sizeof(float);
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&params, params_mem));
  } else {
    params = (float *) malloc(params_mem);
    checkCudaErrors(cudaMalloc((void **) &d_params, params_mem));
  }

  //========================================40
  //		TEMPORARY SOLVER VARIABLES
  //========================================40

  com_mem = workload * 3 * sizeof(float);
  checkCudaErrors(cudaMalloc((void **) &d_com, com_mem));

  err_mem = workload * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **) &d_err, err_mem));

  scale_mem = workload * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **) &d_scale, scale_mem));

  yy_mem = workload * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **) &d_yy, yy_mem));

  initvalu_temp_mem = workload * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **) &d_initvalu_temp, initvalu_temp_mem));

  finavalu_temp_mem = workload * 13 * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **) &d_finavalu_temp, finavalu_temp_mem));

  TIMESTAMP(t1);
  time_malloc += ELAPSED(t0, t1);

  //================================================================================80
  // 	READ FROM FILES OR SET INITIAL VALUES
  //================================================================================80

  for (i = 0; i < workload; i++) {
    pointer = i * (xmax + 1) + 0;
    x[pointer] = 0;
  }
  for (i = 0; i < workload; i++) {
    pointer = i * ((xmax + 1) * EQUATIONS) + 0 * (EQUATIONS);
    read("../../data/myocyte/y.txt", &y[pointer], 91, 1, 0);
  }
  for (i = 0; i < workload; i++) {
    pointer = i * PARAMETERS;
    read("../../data/myocyte/params.txt", &params[pointer], 18, 1, 0);
  }
  TIMESTAMP(t2);
  time_pre += ELAPSED(t1, t2);

  if (unified) {
    d_x = x;
    d_y = y;
    d_params = params;
  } else {
    checkCudaErrors(cudaMemcpy(d_x, x, x_mem, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y, y_mem, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_params, params, params_mem, cudaMemcpyHostToDevice));
  }

  TIMESTAMP(t3);
  time_copy_in += ELAPSED(t2, t3);

  //================================================================================80
  //		EXECUTION IF THERE ARE MANY WORKLOADS
  //================================================================================80

  if (workload == 1) {
    threads.x = 32;																			// define the number of threads in the block
    threads.y = 1;
    blocks.x = 4;																				// define the number of blocks in the grid
    blocks.y = 1;
  } else {
    threads.x = NUMBER_THREADS;												// define the number of threads in the block
    threads.y = 1;
    blocks_x = workload / threads.x;
    if (workload % threads.x != 0) {	// compensate for division remainder above by adding one grid
      blocks_x = blocks_x + 1;
    }
    blocks.x = blocks_x;																	// define the number of blocks in the grid
    blocks.y = 1;
  }

  solver_2<<<blocks, threads>>>( workload,
      xmax,
      d_x,
      d_y,
      d_params,
      d_com,
      d_err,
      d_scale,
      d_yy,
      d_initvalu_temp,
      d_finavalu_temp);

  cudaThreadSynchronize();
  TIMESTAMP(t4);
  time_kernel += ELAPSED(t3, t4);

  //================================================================================80
  //		COPY DATA BACK TO CPU
  //================================================================================80

  if (!unified) {
    checkCudaErrors(cudaMemcpy(x, d_x, x_mem, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(y, d_y, y_mem, cudaMemcpyDeviceToHost));
  }
  TIMESTAMP(t5);
  time_copy_out += ELAPSED(t4, t5);

  //================================================================================80
  //		PRINT RESULTS (ENABLE SELECTIVELY FOR TESTING ONLY)
  //================================================================================80

  for (int i = 0; i < workload; i++){
    for (int j = 0; j < (xmax + 1); j++){
      for (int k = 0; k < EQUATIONS; k++){
        // touch data to bring it back to cpu
        // this is not computationally meaningful
        y[i * ((xmax + 1) * EQUATIONS) + j * (EQUATIONS) + k] += 1;
      }
    }
  }

  for (int i = 0; i < workload; i++){
    for (int j = 0; j < (xmax + 1); j++){
      // touch data to bring it back to cpu
      // this is not computationally meaningful
      x[i * (xmax+1) + j] += 1;
    }
  }

  //================================================================================80
  //		DEALLOCATION
  //================================================================================80

  //============================================================60
  //		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
  //============================================================60

  TIMESTAMP(t6);
  time_post += ELAPSED(t5, t6);

  if (unified) {
    checkCudaErrors(cudaFree(y));
    checkCudaErrors(cudaFree(x));
    checkCudaErrors(cudaFree(params));
  } else {
    free(y);
    checkCudaErrors(cudaFree(d_y));

    free(x);
    checkCudaErrors(cudaFree(d_x));

    free(params);
    checkCudaErrors(cudaFree(d_params));
  }

  //============================================================60
  //		TEMPORARY SOLVER VARIABLES
  //============================================================60

  checkCudaErrors(cudaFree(d_com));

  checkCudaErrors(cudaFree(d_err));
  checkCudaErrors(cudaFree(d_scale));
  checkCudaErrors(cudaFree(d_yy));

  checkCudaErrors(cudaFree(d_initvalu_temp));
  checkCudaErrors(cudaFree(d_finavalu_temp));

  TIMESTAMP(t7);
  time_free += ELAPSED(t6, t7);

  //================================================================================80
  //		DISPLAY TIMING
  //================================================================================80

  printf("====Timing info====\n");
  printf("time malloc = %f ms\n", time_malloc);
  printf("time pre = %f ms\n", time_pre);
  printf("time copyIn = %f ms\n", time_copy_in);
  printf("time kernel = %f ms\n", time_kernel);
  printf("time serial = %f ms\n", time_serial);
  printf("time copyOut = %f ms\n", time_copy_out);
  printf("time post = %f ms\n", time_post);
  printf("time free = %f ms\n", time_free);
  printf("time end-to-end = %f ms\n", ELAPSED(t0, t7));
  exit(EXIT_SUCCESS);

  //====================================================================================================100
  //		END OF FILE
  //====================================================================================================100

  return 0;

}

