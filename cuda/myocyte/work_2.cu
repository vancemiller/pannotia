//====================================================================================================100
//		DEFINE / INCLUDE
//====================================================================================================100

#include "helper_cuda.h"
#include "kernel_fin_2.cu"
#include "kernel_ecc_2.cu"
#include "kernel_cam_2.cu"
#include "kernel_2.cu"
#include "embedded_fehlberg_7_8_2.cu"
#include "solver_2.cu"

//====================================================================================================100
//		MAIN FUNCTION
//====================================================================================================100

int work_2(int xmax, int workload, bool unified) {

  //================================================================================80
  //		VARIABLES
  //================================================================================80

  //============================================================60
  //		TIME
  //============================================================60

  long long time0;
  long long time1;
  long long time2;
  long long time3;
  long long time4;
  long long time5;
  long long time6;

  time0 = get_time();

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

  time1 = get_time();

  //================================================================================80
  // 	ALLOCATE MEMORY
  //================================================================================80

  //============================================================60
  //		MEMORY CHECK
  //============================================================60

  memory = workload*(xmax+1)*EQUATIONS*4;
  if(memory>1000000000){
    printf("ERROR: trying to allocate more than 1.0GB of memory, decrease workload and span parameters or change memory parameter\n");
    return 0;
  }

  //============================================================60
  // 	ALLOCATE ARRAYS
  //============================================================60

  //========================================40
  //		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
  //========================================40

  y_mem = workload * (xmax+1) * EQUATIONS * sizeof(float);
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&y, y_mem));
  } else {
    y = (float *) malloc(y_mem);
    checkCudaErrors(cudaMalloc((void **)&d_y, y_mem));
  }

  x_mem = workload * (xmax+1) * sizeof(float);
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&x, x_mem));
  } else {
    x = (float *) malloc(x_mem);
    checkCudaErrors(cudaMalloc((void **)&d_x, x_mem));
  }

  params_mem = workload * PARAMETERS * sizeof(float);
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&params, params_mem));
  } else {
    params= (float *) malloc(params_mem);
    checkCudaErrors(cudaMalloc((void **)&d_params, params_mem));
  }

  //========================================40
  //		TEMPORARY SOLVER VARIABLES
  //========================================40

  com_mem = workload * 3 * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_com, com_mem));

  err_mem = workload * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_err, err_mem));

  scale_mem = workload * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_scale, scale_mem));

  yy_mem = workload * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_yy, yy_mem));

  initvalu_temp_mem = workload * EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_initvalu_temp, initvalu_temp_mem));

  finavalu_temp_mem = workload * 13* EQUATIONS * sizeof(float);
  checkCudaErrors(cudaMalloc((void **)&d_finavalu_temp, finavalu_temp_mem));

  time2 = get_time();

  //================================================================================80
  // 	READ FROM FILES OR SET INITIAL VALUES
  //================================================================================80

  //========================================40
  //		X
  //========================================40

  for(i=0; i<workload; i++){
    pointer = i * (xmax+1) + 0;
    x[pointer] = 0;
  }
  if (unified) {
    d_x = x;
  } else {
    checkCudaErrors(cudaMemcpy(d_x, x, x_mem, cudaMemcpyHostToDevice));
  }

  //========================================40
  //		Y
  //========================================40

  for(i=0; i<workload; i++){
    pointer = i*((xmax+1)*EQUATIONS) + 0*(EQUATIONS);
    read("../../data/myocyte/y.txt",
        &y[pointer],
        91,
        1,
        0);
  }
  if (unified) {
    d_y = y;
  } else {
    checkCudaErrors(cudaMemcpy(d_y, y, y_mem, cudaMemcpyHostToDevice));
  }

  //========================================40
  //		PARAMS
  //========================================40

  for(i=0; i<workload; i++){
    pointer = i*PARAMETERS;
    read("../../data/myocyte/params.txt",
        &params[pointer],
        18,
        1,
        0);
  }
  if (unified) {
    d_params = params;
  } else {
    checkCudaErrors(cudaMemcpy(d_params, params, params_mem, cudaMemcpyHostToDevice));
  }

  time3 = get_time();

  //================================================================================80
  //		EXECUTION IF THERE ARE MANY WORKLOADS
  //================================================================================80

  if(workload == 1){
    threads.x = 32;																			// define the number of threads in the block
    threads.y = 1;
    blocks.x = 4;																				// define the number of blocks in the grid
    blocks.y = 1;
  }
  else{
    threads.x = NUMBER_THREADS;												// define the number of threads in the block
    threads.y = 1;
    blocks_x = workload/threads.x;
    if (workload % threads.x != 0){												// compensate for division remainder above by adding one grid
      blocks_x = blocks_x + 1;
    }
    blocks.x = blocks_x;																	// define the number of blocks in the grid
    blocks.y = 1;
  }

  solver_2<<<blocks, threads>>>(	workload,
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

  // cudaThreadSynchronize();
  // printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));

  time4 = get_time();

  //================================================================================80
  //		COPY DATA BACK TO CPU
  //================================================================================80

  if (!unified) {
    checkCudaErrors(cudaMemcpy(x, d_x, x_mem, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(y, d_y, y_mem, cudaMemcpyDeviceToHost));
  }
  time5 = get_time();

  //================================================================================80
  //		PRINT RESULTS (ENABLE SELECTIVELY FOR TESTING ONLY)
  //================================================================================80

  // int j, k;

  // for(i=0; i<workload; i++){
  // printf("WORKLOAD %d:\n", i);
  // for(j=0; j<(xmax+1); j++){
  // printf("\tTIME %d:\n", j);
  // for(k=0; k<EQUATIONS; k++){
  // printf("\t\ty[%d][%d][%d]=%13.10f\n", i, j, k, y[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k]);
  // }
  // }
  // }

  // for(i=0; i<workload; i++){
  // printf("WORKLOAD %d:\n", i);
  // for(j=0; j<(xmax+1); j++){
  // printf("\tTIME %d:\n", j);
  // printf("\t\tx[%d][%d]=%13.10f\n", i, j, x[i * (xmax+1) + j]);
  // }
  // }

  //================================================================================80
  //		DEALLOCATION
  //================================================================================80

  //============================================================60
  //		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
  //============================================================60

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

  time6= get_time();

  //================================================================================80
  //		DISPLAY TIMING
  //================================================================================80

  printf("Time spent in different stages of the application:\n");
  printf("%.12f s, %.12f %% : SETUP VARIABLES\n", 															(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time6-time0) * 100);
  printf("%.12f s, %.12f %% : ALLOCATE CPU MEMORY AND GPU MEMORY\n", 				(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time6-time0) * 100);
  printf("%.12f s, %.12f %% : READ DATA FROM FILES, COPY TO GPU MEMORY\n", 		(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time6-time0) * 100);
  printf("%.12f s, %.12f %% : RUN GPU KERNEL\n", 															(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time6-time0) * 100);
  printf("%.12f s, %.12f %% : COPY GPU DATA TO CPU MEMORY\n", 								(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time6-time0) * 100);
  printf("%.12f s, %.12f %% : FREE MEMORY\n", 																(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time6-time0) * 100);
  printf("Total time:\n");
  printf("%.12f s\n", 																											(float) (time6-time0) / 1000000);

  //====================================================================================================100
  //		END OF FILE
  //====================================================================================================100

  return 0;

}

