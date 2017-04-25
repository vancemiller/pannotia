#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "helper_cuda.h"

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define TIMESTAMP(NAME) \
  struct timespec NAME; \
  if (clock_gettime(CLOCK_MONOTONIC, &NAME)) { \
    fprintf(stderr, "Failed to get time: %s\n", strerror(errno)); \
  }

#define ELAPSED(start, end) \
  ((uint64_t) 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
bool unified;
#define M_SEED 9
int pyramid_height;

long long time_pre = 0;
long long time_post = 0;
long long time_serial = 0;
long long time_copy_in = 0;
long long time_copy_out = 0;
long long time_kernel = 0;
long long time_malloc = 0;
long long time_free = 0;

void init(int argc, char** argv) {
  if (argc >= 4) {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
    unified = argc == 5;
  } else {
    printf("Usage: dynproc row_len col_len pyramid_height unified_flag\n");
    exit(EXIT_FAILURE);
  }
}

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc, int *gpuResults, int cols,
    int rows, int startStep, int border) {
  __shared__ int prev[BLOCK_SIZE];
  __shared__ int result[BLOCK_SIZE];

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size
  int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;

  // calculate the boundary for the block according to
  // the boundary of its small block
  int blkX = small_block_cols * bx - border;
  int blkXmax = blkX + BLOCK_SIZE - 1;

  // calculate the global thread coordination
  int xidx = blkX + tx;

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;

  int W = tx - 1;
  int E = tx + 1;

  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;

  bool isValid = IN_RANGE(tx, validXmin, validXmax);

  if (IN_RANGE(xidx, 0, cols - 1)) {
    prev[tx] = gpuSrc[xidx];
  }
  __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if ( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) && isValid) {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = cols * (startStep + i) + xidx;
      result[tx] = shortest + gpuWall[index];
    }
    __syncthreads();
    if (i == iteration - 1)
      break;
    if (computed)	 //Assign the computation range
      prev[tx] = result[tx];
    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
  }

  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on ``computed''
  if (computed) {
    gpuResults[xidx] = result[tx];
  }
}

/*
   compute N time steps
 */
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, int pyramid_height,
    int blockCols, int borderCols) {
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(blockCols);

  int src = 1, dst = 0;
  for (int t = 0; t < rows - 1; t += pyramid_height) {
    int temp = src;
    src = dst;
    dst = temp;
    dynproc_kernel<<<dimGrid, dimBlock>>>(
        MIN(pyramid_height, rows-t-1),
        gpuWall, gpuResult[src], gpuResult[dst],
        cols,rows, t, borderCols);
  }
  return dst;
}

int main(int argc, char** argv) {
  int num_devices;
  checkCudaErrors(cudaGetDeviceCount(&num_devices));
  if (num_devices > 1)
    checkCudaErrors(cudaSetDevice(DEVICE));

  run(argc, argv);

  return EXIT_SUCCESS;
}

void run(int argc, char** argv) {
  init(argc, argv);
  TIMESTAMP(t0);
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&data, sizeof(int) * rows * cols));
  } else {
    data = (int*) malloc(sizeof(int) * rows * cols);
    result = (int*) malloc(sizeof(int) * cols);
  }
  wall = new int*[rows];
  TIMESTAMP(t1);
  time_malloc += ELAPSED(t0, t1);

  for (int n = 0; n < rows; n++)
    wall[n] = data + cols * n;

  int seed = M_SEED;
  srand(seed);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      wall[i][j] = rand() % 10;
    }
  }
  TIMESTAMP(t2);
  time_pre += ELAPSED(t1, t2);

#ifdef BENCH_PRINT
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%d ", wall[i][j]);
    }
    printf("\n");
  }
#endif

  /* --------------- pyramid parameters --------------- */
  int borderCols = (pyramid_height) * HALO;
  int smallBlockCol = BLOCK_SIZE - (pyramid_height) * HALO * 2;
  int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

  printf(
      "pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
      pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

  int *gpuWall, *gpuResult[2];
  int size = rows * cols;

  TIMESTAMP(t3);
  if (unified) {
    gpuResult[0] = data;
    checkCudaErrors(cudaMallocManaged(&gpuResult[1], sizeof(int) * cols));
    gpuWall = data + cols;
  } else {
    checkCudaErrors(cudaMalloc((void**) &gpuResult[0], sizeof(int) * cols));
    checkCudaErrors(cudaMalloc((void**) &gpuResult[1], sizeof(int) * cols));
    checkCudaErrors(cudaMalloc((void**) &gpuWall, sizeof(int) * (size - cols)));
  }
  TIMESTAMP(t4);
  time_malloc += ELAPSED(t3, t4);
  if (!unified) {
    checkCudaErrors(cudaMemcpy(gpuResult[0], data, sizeof(int) * cols, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpuWall, data + cols, sizeof(int) * (size - cols), cudaMemcpyHostToDevice));
  }
  TIMESTAMP(t5);
  time_copy_in = ELAPSED(t4, t5);

  int final_ret = calc_path(gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);

  TIMESTAMP(t6);
  time_kernel += ELAPSED(t5, t6);

  if (unified) {
    result = gpuResult[final_ret];
  } else {
    checkCudaErrors(cudaMemcpy(result, gpuResult[final_ret], sizeof(int) * cols, cudaMemcpyDeviceToHost));
  }
  TIMESTAMP(t7);
  time_copy_out += ELAPSED(t6, t7);

  // Touch the data to bring it back to cpu
  // not computationally correct
  for (int i = 0; i < cols; i++)
    data[i] += 1;
  for (int i = 0; i < cols; i++)
    result[i] += 1;
  TIMESTAMP(t8);
  time_post += ELAPSED(t7, t8);

  if (unified) {
    checkCudaErrors(cudaFree(data));
    checkCudaErrors(cudaFree(gpuResult[1]));
  } else {
    checkCudaErrors(cudaFree(gpuWall));
    checkCudaErrors(cudaFree(gpuResult[0]));
    checkCudaErrors(cudaFree(gpuResult[1]));
    free(data);
    free(result);
  }
  delete[] wall;
  TIMESTAMP(t9);
  time_free += ELAPSED(t8, t9);

  printf("====Timing info====\n");
  printf("time malloc = %f ms\n", time_malloc * 1e-6);
  printf("time pre = %f ms\n", time_pre * 1e-6);
  printf("time CPU to GPU memory copy = %f ms\n", time_copy_in * 1e-6);
  printf("time kernel = %f ms\n", time_kernel * 1e-6);
  printf("time serial = %f ms\n", time_serial * 1e-6);
  printf("time GPU to CPU memory copy back = %f ms\n", time_copy_out * 1e-6);
  printf("time post = %f ms\n", time_post * 1e-6);
  printf("time free = %f ms\n", time_free * 1e-6);
  printf("End-to-end = %f ms\n", ELAPSED(t0, t9) * 1e-6);
  exit(EXIT_SUCCESS);
}

