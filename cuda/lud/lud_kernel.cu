#include <cuda.h>
#include <stdio.h>
#include "lud_kernel.cuh"
#include "helper_cuda.h"

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

#ifdef __cplusplus
extern "C" {
#endif
__global__ void lud_diagonal(float *m, int size, int offset) {
  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  int array_offset = offset * size + offset;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    shadow[i][threadIdx.x] = m[array_offset + threadIdx.x];
    array_offset += size;
  }
  __syncthreads();

  for (int i = 0; i < BLOCK_SIZE - 1; i++) {
    if (threadIdx.x > i) {
      for (int j = 0; j < i; j++) {
        shadow[threadIdx.x][i] -= shadow[threadIdx.x][j] * shadow[j][i];
      }
      shadow[threadIdx.x][i] /= shadow[i][i];
    }

    __syncthreads();

    if (threadIdx.x > i) {
      for (int j = 0; j < i + 1; j++)
        shadow[i + 1][threadIdx.x] -= shadow[i + 1][j] * shadow[j][threadIdx.x];
    }
    __syncthreads();
  }

  /*
   The first row is not modified, it
   is no need to write it back to the
   global memory

   */
  array_offset = (offset + 1) * size + offset;
  for (int i = 1; i < BLOCK_SIZE; i++) {
    m[array_offset + threadIdx.x] = shadow[i][threadIdx.x];
    array_offset += size;
  }
}

__global__ void lud_perimeter(float *m, int size, int offset) {
  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int idx = threadIdx.x;

  if (idx < BLOCK_SIZE) {
    int array_offset = offset * size + offset;
    for (int i = 0; i < BLOCK_SIZE / 2; i++) {
      dia[i][idx] = m[array_offset + idx];
      array_offset += size;
    }

    array_offset = offset * size + offset;
    for (int i = 0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx] = m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx];
      array_offset += size;
    }
  } else {
    idx -= BLOCK_SIZE;

    int array_offset = (offset + BLOCK_SIZE / 2) * size + offset;
    for (int i = BLOCK_SIZE / 2; i < BLOCK_SIZE; i++) {
      dia[i][idx] = m[array_offset + idx];
      array_offset += size;
    }

    array_offset = (offset + (blockIdx.x + 1) * BLOCK_SIZE) * size + offset;
    for (int i = 0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset + idx];
      array_offset += size;
    }

  }
  __syncthreads();

  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    for (int i = 1; i < BLOCK_SIZE; i++) {
      for (int j = 0; j < i; j++)
        peri_row[i][idx] -= dia[i][j] * peri_row[j][idx];
    }
  } else { //peri-col
    for (int i = 0; i < BLOCK_SIZE; i++) {
      for (int j = 0; j < i; j++)
        peri_col[idx][i] -= peri_col[idx][j] * dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }
  }

  __syncthreads();

  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    int array_offset = (offset + 1) * size + offset;
    for (int i = 1; i < BLOCK_SIZE; i++) {
      m[array_offset + (blockIdx.x + 1) * BLOCK_SIZE + idx] = peri_row[i][idx];
      array_offset += size;
    }
  } else { //peri-col
    int array_offset = (offset + (blockIdx.x + 1) * BLOCK_SIZE) * size + offset;
    for (int i = 0; i < BLOCK_SIZE; i++) {
      m[array_offset + idx] = peri_col[i][idx];
      array_offset += size;
    }
  }

}

__global__ void lud_internal(float *m, int size, int offset) {
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int global_row_id = offset + (blockIdx.y + 1) * BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x + 1) * BLOCK_SIZE;

  peri_row[threadIdx.y][threadIdx.x] = m[(offset + threadIdx.y) * size + global_col_id
      + threadIdx.x];
  peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id + threadIdx.y) * size + offset
      + threadIdx.x];

  __syncthreads();

  float sum = 0;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
  }
  m[(global_row_id + threadIdx.y) * size + global_col_id + threadIdx.x] -= sum;

}

#ifdef __cplusplus
}
#endif
