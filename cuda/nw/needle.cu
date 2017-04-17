#define LIMIT -999
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "needle_kernel.cu"

#define TIMESTAMP(NAME) \
  struct timespec NAME; \
  if (clock_gettime(CLOCK_MONOTONIC, &NAME)) { \
    fprintf(stderr, "Failed to get time: %s\n", strerror(errno)); \
  }

#define ELAPSED(start, end) \
  ((uint64_t) 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);

int blosum62[24][24] = {
  { 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
  {-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
  {-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
  {-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
  { 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
  {-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
  {-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
  { 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
  {-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
  {-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
  {-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
  {-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
  {-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
  {-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
  {-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
  { 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
  { 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
  {-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
  {-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
  { 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
  {-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
  {-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
  { 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
  {-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
  printf("WG size of kernel = %d \n", BLOCK_SIZE);
  runTest(argc, argv);
  return EXIT_SUCCESS;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
  fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
  fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
  exit(1);
}

void runTest(int argc, char** argv) {
  long long time_serial = 0;
  long long time_copy_in = 0;
  long long time_copy_out = 0;
  long long time_kernel = 0;
  long long time_malloc = 0;
  long long time_free = 0;
  int max_rows, max_cols, penalty;
  int *input_itemsets, *output_itemsets, *reference;
  int *matrix_cuda, *reference_cuda;
  int size;
  bool unified;

  // the lengths of the two sequences should be able to divided by 16.
  // And at current stage  max_rows needs to equal max_cols
  if (argc >= 3) {
    max_rows = atoi(argv[1]);
    max_cols = atoi(argv[1]);
    penalty = atoi(argv[2]);
    unified = argc == 4;
  } else {
    usage(argc, argv);
  }

  if (atoi(argv[1]) % 16 != 0) {
    fprintf(stderr, "The dimension values must be a multiple of 16\n");
    exit(1);
  }

  max_rows = max_rows + 1;
  max_cols = max_cols + 1;
  size = max_cols * max_rows;
  TIMESTAMP(t0);
  if (unified) {
    cudaMallocManaged(&reference, sizeof(int) * max_rows * max_cols);
    cudaMallocManaged(&input_itemsets, sizeof(int) * max_rows * max_cols);
  } else {
    reference = (int *) malloc(max_rows * max_cols * sizeof(int));
    input_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
    output_itemsets = (int *) malloc(max_rows * max_cols * sizeof(int));
    cudaMalloc((void**) &reference_cuda, sizeof(int) * size);
    cudaMalloc((void**) &matrix_cuda, sizeof(int) * size);
  }
  TIMESTAMP(t1);
  time_malloc += ELAPSED(t0, t1);

  if (!input_itemsets)
    fprintf(stderr, "error: can not allocate memory");

  srand(7);

  for (int i = 0; i < max_cols; i++) {
    for (int j = 0; j < max_rows; j++) {
      input_itemsets[i * max_cols + j] = 0;
    }
  }

  printf("Start Needleman-Wunsch\n");

  for (int i = 1; i < max_rows; i++) {    //please define your own sequence.
    input_itemsets[i * max_cols] = rand() % 10 + 1;
  }
  for (int j = 1; j < max_cols; j++) {    //please define your own sequence.
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (int i = 1; i < max_cols; i++) {
    for (int j = 1; j < max_rows; j++) {
      reference[i * max_cols + j] = blosum62[input_itemsets[i * max_cols]][input_itemsets[j]];
    }
  }

  for (int i = 1; i < max_rows; i++)
    input_itemsets[i * max_cols] = -i * penalty;
  for (int j = 1; j < max_cols; j++)
    input_itemsets[j] = -j * penalty;


  TIMESTAMP(t2);
  time_serial += ELAPSED(t1, t2);

  if (unified) {
    reference_cuda = reference;
    matrix_cuda = input_itemsets;
  } else {
    cudaMemcpy(reference_cuda, reference, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size, cudaMemcpyHostToDevice);
  }
  TIMESTAMP(t3);
  time_copy_in += ELAPSED(t2, t3);

  dim3 dimGrid;
  dim3 dimBlock(BLOCK_SIZE, 1);
  int block_width = (max_cols - 1) / BLOCK_SIZE;

  printf("Processing top-left matrix\n");
  //process top-left matrix
  for (int i = 1; i <= block_width; i++) {
    dimGrid.x = i;
    dimGrid.y = 1;
    needle_cuda_shared_1<<<dimGrid, dimBlock>>>(reference_cuda, matrix_cuda ,max_cols, penalty,
        i, block_width);
  }
  printf("Processing bottom-right matrix\n");
  //process bottom-right matrix
  for (int i = block_width - 1; i >= 1; i--) {
    dimGrid.x = i;
    dimGrid.y = 1;
    needle_cuda_shared_2<<<dimGrid, dimBlock>>>(reference_cuda, matrix_cuda, max_cols, penalty,
        i, block_width);
  }

  TIMESTAMP(t4);
  time_kernel += ELAPSED(t3, t4);

  if (unified) {
    output_itemsets = matrix_cuda;
  } else {
    cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost);
  }
  TIMESTAMP(t5);
  time_copy_out += ELAPSED(t4, t5);

  // post processing
  for (int i = max_rows - 2, j = max_rows - 2; i >= 0 && j >= 0;) {
    int nw, n, w, traceback;
    if ( i == 0 && j == 0 )
      break;
    if ( i > 0 && j > 0 ) {
      nw = output_itemsets[(i - 1) * max_cols + j - 1];
      w = output_itemsets[ i * max_cols + j - 1 ];
      n = output_itemsets[(i - 1) * max_cols + j];
    } else if ( i == 0 ) {
      nw = n = LIMIT;
      w = output_itemsets[ i * max_cols + j - 1 ];
    } else if ( j == 0 ) {
      nw = w = LIMIT;
      n = output_itemsets[(i - 1) * max_cols + j];
    } else {
    }

    //traceback = maximum(nw, w, n);
    int new_nw, new_w, new_n;
    new_nw = nw + reference[i * max_cols + j];
    new_w = w - penalty;
    new_n = n - penalty;

    traceback = maximum(new_nw, new_w, new_n);
    if(traceback == new_nw)
      traceback = nw;
    if(traceback == new_w)
      traceback = w;
    if(traceback == new_n)
      traceback = n;

    if(traceback == nw )
    { i--; j--; continue;}

    else if(traceback == w )
    { j--; continue;}

    else if(traceback == n )
    { i--; continue;}
  }

  TIMESTAMP(t6);
  time_serial += ELAPSED(t5, t6);
  if (unified) {
    cudaFree(reference);
    cudaFree(input_itemsets);
  } else {
    cudaFree(reference_cuda);
    cudaFree(matrix_cuda);
    free(reference);
    free(input_itemsets);
    free(output_itemsets);
  }
  TIMESTAMP(t7);
  time_free += ELAPSED(t6, t7);
  printf("====Timing info====\n");
  printf("time serial = %f ms\n", time_serial * 1e-6);
  printf("time GPU malloc = %f ms\n", time_malloc * 1e-6);
  printf("time CPU to GPU memory copy = %f ms\n", time_copy_in * 1e-6);
  printf("time kernel = %f ms\n", time_kernel * 1e-6);
  printf("time GPU to CPU memory copy back = %f ms\n", time_copy_out * 1e-6);
  printf("time GPU free = %f ms\n", time_free * 1e-6);
  printf("End-to-end = %f ms\n", ELAPSED(t0, t7) * 1e-6);
}

