// includes, system
#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "srad.h"

// includes, project
#include <cuda.h>
#include "helper_cuda.h"

// includes, kernels
#include "srad_kernel.cu"

#include "../timing.h"

void random_matrix(float *I, int rows, int cols);
void runTest(int argc, char** argv);
void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <rows> <cols> <y1> <y2> <x1> <x2> <lamda> <no. of iter>\n", argv[0]);
  fprintf(stderr, "\t<rows>   - number of rows\n");
  fprintf(stderr, "\t<cols>    - number of cols\n");
  fprintf(stderr, "\t<y1> 	 - y1 value of the speckle\n");
  fprintf(stderr, "\t<y2>      - y2 value of the speckle\n");
  fprintf(stderr, "\t<x1>       - x1 value of the speckle\n");
  fprintf(stderr, "\t<x2>       - x2 value of the speckle\n");
  fprintf(stderr, "\t<lamda>   - lambda (0,1)\n");
  fprintf(stderr, "\t<no. of iter>   - number of iterations\n");
  fprintf(stderr, "\t<unified flag>   - unified or default memory\n");

  exit(1);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
  runTest(argc, argv);

  return EXIT_SUCCESS;
}

void runTest(int argc, char** argv) {
  float time_pre = 0;
  float time_post = 0;
  float time_serial = 0;
  float time_copy_in = 0;
  float time_copy_out = 0;
  float time_kernel = 0;
  float time_malloc = 0;
  float time_free = 0;
  int rows, cols, size_I, size_R, niter = 10, iter;
  float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI, varROI;

  float *J_cuda;
  float *C_cuda;
  float *E_C, *W_C, *N_C, *S_C;

  unsigned int r1, r2, c1, c2;
  float *c;
  bool unified;


  if (argc >= 9) {
    rows = atoi(argv[1]);  //number of rows in the domain
    cols = atoi(argv[2]);  //number of cols in the domain
    if ((rows % 16 != 0) || (cols % 16 != 0)) {
      fprintf(stderr, "rows and cols must be multiples of 16\n");
      exit(1);
    }
    r1 = atoi(argv[3]);  //y1 position of the speckle
    r2 = atoi(argv[4]);  //y2 position of the speckle
    c1 = atoi(argv[5]);  //x1 position of the speckle
    c2 = atoi(argv[6]);  //x2 position of the speckle
    lambda = atof(argv[7]); //Lambda value
    niter = atoi(argv[8]); //number of iterations
    unified = argc == 10; // extra arg == unified

  } else {
    usage(argc, argv);
  }

  size_I = cols * rows;
  size_R = (r2 - r1 + 1) * (c2 - c1 + 1);

  TIMESTAMP(t0);

  I = (float *) malloc(size_I * sizeof(float));
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&J, size_I * sizeof(float)));
  } else {
    J = (float*) malloc(size_I * sizeof(float));
  }
  assert(J);
  c = (float*) malloc(sizeof(float) * size_I);

  //Allocate device memory
  if (unified) {
    J_cuda = J;
  } else {
    checkCudaErrors(cudaMalloc(&J_cuda, sizeof(float) * size_I));
  }
  checkCudaErrors(cudaMalloc(&C_cuda, sizeof(float) * size_I));
  checkCudaErrors(cudaMalloc(&E_C, sizeof(float) * size_I));
  checkCudaErrors(cudaMalloc(&W_C, sizeof(float) * size_I));
  checkCudaErrors(cudaMalloc(&S_C, sizeof(float) * size_I));
  checkCudaErrors(cudaMalloc(&N_C, sizeof(float) * size_I));

  TIMESTAMP(t1);
  time_malloc += ELAPSED(t0, t1);

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  printf("Randomizing the input matrix\n");
  //Generate a random matrix
  random_matrix(I, rows, cols);

  for (int k = 0; k < size_I; k++) {
    J[k] = (float) exp(I[k]);
  }

  TIMESTAMP(t2);
  time_pre += ELAPSED(t1, t2);

  printf("Start the SRAD main loop\n");
  for (iter = 0; iter < niter; iter++) {
    sum = 0;
    sum2 = 0;
    for (int i = r1; i <= r2; i++) {
      for (int j = c1; j <= c2; j++) {
        tmp = J[i * cols + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    }
    meanROI = sum / size_R;
    varROI = (sum2 / size_R) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

    //Currently the input size must be divided by 16 - the block size
    int block_x = cols / BLOCK_SIZE;
    int block_y = rows / BLOCK_SIZE;

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(block_x, block_y);

    //Copy data from main memory to device memory
    if (!unified) {
      TIMESTAMP(copy0);
      checkCudaErrors(cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice));
      TIMESTAMP(copy1);
      time_copy_in += ELAPSED(copy0, copy1);
    }

    //Run kernels
    TIMESTAMP(kernel0);
    srad_cuda_1<<<dimGrid, dimBlock, 0, stream>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr);
    srad_cuda_2<<<dimGrid, dimBlock, 0, stream>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr);
    checkCudaErrors(cudaStreamSynchronize(stream));
    TIMESTAMP(kernel1);
    time_kernel += ELAPSED(kernel0, kernel1);

    //Copy data from device memory to main memory
    if (!unified) {
      TIMESTAMP(copy0);
      checkCudaErrors(cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost));
      TIMESTAMP(copy1);
      time_copy_out += ELAPSED(copy0, copy1);
    }
  }

  TIMESTAMP(t3);
  // bring data back to cpu
  // not computationally correct
  for( int i = 0; i < rows; i++) {
    for ( int j = 0; j < cols; j++) {
      J[i * cols + j] += 1;
    }
  }
  TIMESTAMP(t4);
  time_post += ELAPSED(t3, t4);

  printf("Computation Done\n");

  free(I);
  if (unified)
    checkCudaErrors(cudaFree(J));
  else
    free(J);
  checkCudaErrors(cudaFree(C_cuda));
  if (!unified)
    checkCudaErrors(cudaFree(J_cuda));
  checkCudaErrors(cudaFree(E_C));
  checkCudaErrors(cudaFree(W_C));
  checkCudaErrors(cudaFree(N_C));
  checkCudaErrors(cudaFree(S_C));
  free(c);
  TIMESTAMP(t5);
  time_free += ELAPSED(t4, t5);

  printf("====Timing info====\n");
  printf("time malloc = %f ms\n", time_malloc);
  printf("time pre = %f ms\n", time_pre);
  printf("time copyIn = %f ms\n", time_copy_in);
  printf("time kernel = %f ms\n", time_kernel);
  printf("time serial = %f ms\n", time_serial);
  printf("time copyOut = %f ms\n", time_copy_out);
  printf("time post = %f ms\n", time_post);
  printf("time free = %f ms\n", time_free);
  printf("time end-to-end = %f ms\n", ELAPSED(t0, t5));
  exit(EXIT_SUCCESS);
}

void random_matrix(float *I, int rows, int cols) {
  srand(666);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      I[i * cols + j] = rand() / (float) RAND_MAX;
    }
  }
}

