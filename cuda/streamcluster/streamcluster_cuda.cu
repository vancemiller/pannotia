/***********************************************
 streamcluster_cuda.cu
 : parallelized code of streamcluster

 - original code from PARSEC Benchmark Suite
 - parallelization with CUDA API has been applied by

 Shawn Sang-Ha Lee - sl4ge@virginia.edu
 University of Virginia
 Department of Electrical and Computer Engineering
 Department of Computer Science

 ***********************************************/
#include "string.h"
#include "errno.h"
#include "streamcluster_header.cu"
#include "helper_cuda.h"
using namespace std;

#define THREADS_PER_BLOCK 512
#define MAXBLOCKS 65536
#define CUDATIME

#define TIMESTAMP(NAME) \
 struct timespec NAME; \
  if (clock_gettime(CLOCK_MONOTONIC, &NAME)) { \
    fprintf(stderr, "Failed to get time: %s\n", strerror(errno)); \
  }

#define ELAPSED(start, end) \
  (uint64_t) 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec

#define VPRINT(verbose, format, ...) \
  if (verbose) {\
    fprintf(stdout, format, ## __VA_ARGS__);\
  }

// host memory
float* work_mem_h;
float* coord_h;

// device memory
float* work_mem_d;
float* coord_d;
int* center_table_d;
bool* switch_membership_d;
Point* p;

static int iter = 0;		// counter for total# of iteration

//=======================================
// Euclidean Distance
//=======================================
__device__ float d_dist(int p1, int p2, int num, int dim, float *coord_d) {
  float retval = 0.0;
  for (int i = 0; i < dim; i++) {
    float tmp = coord_d[(i * num) + p1] - coord_d[(i * num) + p2];
    retval += tmp * tmp;
  }
  return retval;
}

//=======================================
// Kernel - Compute Cost
//=======================================
__global__ void kernel_compute_cost(int num, int dim, long x, Point *p, int K, int stride,
    float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d) {
  // block ID and global thread ID
  const int bid = blockIdx.x + gridDim.x * blockIdx.y;
  const int tid = blockDim.x * bid + threadIdx.x;

  if (tid < num) {
    float *lower = &work_mem_d[tid * stride];

    // cost between this point and point[x]: euclidean distance multiplied by weight
    float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;
    // if computed cost is less then original (it saves), mark it as to reassign
    if (x_cost < p[tid].cost) {
      switch_membership_d[tid] = 1;
      lower[K] += x_cost - p[tid].cost;
    }
    // if computed cost is larger, save the difference
    else {
      lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
    }
  }
}

//=======================================
// Allocate Device Memory
//=======================================
void allocDevMem(int num, int dim, bool unified) {
  if (!unified) {
    checkCudaErrors(cudaMalloc((void**) &center_table_d, num * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**) &switch_membership_d, num * sizeof(bool)));
    checkCudaErrors(cudaMalloc((void**) &p, num * sizeof(Point)));
    checkCudaErrors(cudaMalloc((void**) &coord_d, num * dim * sizeof(float)));
  }
}

//=======================================
// Allocate Host Memory
//=======================================
void allocHostMem(int num, int dim, bool unified) {
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&coord_h, num * dim * sizeof(float)));
  } else {
    coord_h = (float*) malloc(num * dim * sizeof(float));
  }
}

//=======================================
// Free Device Memory
//=======================================
void freeDevMem(bool unified) {
  if (!unified) {
    checkCudaErrors(cudaFree(center_table_d));
    checkCudaErrors(cudaFree(switch_membership_d));
    checkCudaErrors(cudaFree(p));
    checkCudaErrors(cudaFree(coord_d));
  }
}

//=======================================
// Free Host Memory
//=======================================
void freeHostMem(bool unified) {
  if (unified) {
    checkCudaErrors(cudaFree(coord_h));
  } else {
    free(coord_h);
  }
}

//=======================================
// pgain Entry - CUDA SETUP + CUDA CALL
//=======================================
float pgain(long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center,
    int *center_table, bool *switch_membership, bool isCoordChanged, long long *serial_t,
    long long *cpu_to_gpu_t, long long *gpu_to_cpu_t, long long *alloc_t, long long *kernel_t,
    long long *free_t, bool unified) {

  TIMESTAMP(t0);

  int stride = *numcenters + 1;			// size of each work_mem segment
  int K = *numcenters;				// number of centers
  int num = points->num;				// number of points
  int dim = points->dim;				// number of dimension
  int nThread = num;						// number of threads == number of data points
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  //=========================================
  // ALLOCATE HOST MEMORY + DATA PREPARATION
  //=========================================
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&work_mem_h, stride * (nThread + 1) * sizeof(float)));
  } else {
    work_mem_h = (float*) malloc(stride * (nThread + 1) * sizeof(float));
  }
  // Only on the first iteration
  if (iter == 0) {
    allocHostMem(num, dim, unified);
  }

  // build center-index table
  int count = 0;
  for (int i = 0; i < num; i++) {
    if (is_center[i]) {
      center_table[i] = count++;
    }
  }

  // Extract 'coord'
  // Only if first iteration OR coord has changed
  if (isCoordChanged || iter == 0) {
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < num; j++) {
        coord_h[(num * i) + j] = points->p[j].coord[i];
      }
    }
  }
  TIMESTAMP(t1);
  *serial_t += ELAPSED(t0, t1);

  //=======================================
  // ALLOCATE GPU MEMORY
  //=======================================
  if (unified) {
    work_mem_d = work_mem_h;
  } else {
    checkCudaErrors(cudaMalloc((void**) &work_mem_d, stride * (nThread + 1) * sizeof(float)));
  }
  // Only on the first iteration
  if (iter == 0) {
    allocDevMem(num, dim, unified);
  }
  TIMESTAMP(t2);
  *alloc_t += ELAPSED(t1, t2);

  //=======================================
  // CPU-TO-GPU MEMORY COPY
  //=======================================
  // Only if first iteration OR coord has changed
  if (isCoordChanged || iter == 0) {
    if (unified) {
      coord_d = coord_h;
    } else {
      checkCudaErrors(
          cudaMemcpy(coord_d, coord_h, num * dim * sizeof(float), cudaMemcpyHostToDevice));
    }
  }
  if (unified) {
    switch_membership_d = switch_membership;
    center_table_d = center_table;
    p = points->p;
  } else {
    checkCudaErrors(
        cudaMemcpy(center_table_d, center_table, num * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(p, points->p, num * sizeof(Point), cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaMemset(switch_membership_d, 0, num * sizeof(bool)));
  checkCudaErrors(cudaMemset(work_mem_d, 0, stride * (nThread + 1) * sizeof(float)));

  TIMESTAMP(t3);
  *cpu_to_gpu_t += ELAPSED(t2, t3);

  //=======================================
  // KERNEL: CALCULATE COST
  //=======================================
  // Determine the number of thread blocks in the x- and y-dimension
  int num_blocks = (int) ((float) (num + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);
  int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1) / (float) MAXBLOCKS);
  int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1) / (float) num_blocks_y);
  dim3 grid_size(num_blocks_x, num_blocks_y, 1);

  kernel_compute_cost<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(
      num,					// in:	# of data
      dim,// in:	dimension of point coordinates
      x,  // in:	point to open a center at
      p,  // in:	data point array
      K,  // in:	number of centers
      stride,// in:  size of each work_mem segment
      coord_d,// in:	array of point coordinates
      work_mem_d,// out:	cost and lower field array
      center_table_d,// in:	center index table
      switch_membership_d// out:  changes in membership
  );
  checkCudaErrors(cudaStreamSynchronize(stream));

  TIMESTAMP(t4);
  *kernel_t += ELAPSED(t3, t4);

  //=======================================
  // GPU-TO-CPU MEMORY COPY
  //=======================================
  if (!unified) {
    checkCudaErrors(
        cudaMemcpy(work_mem_h, work_mem_d, stride * (nThread + 1) * sizeof(float),
            cudaMemcpyDeviceToHost));
    checkCudaErrors(
        cudaMemcpy(switch_membership, switch_membership_d, num * sizeof(bool),
            cudaMemcpyDeviceToHost));
  }

  TIMESTAMP(t5);
  *gpu_to_cpu_t += ELAPSED(t4, t5);

  //=======================================
  // CPU (SERIAL) WORK
  //=======================================
  int number_of_centers_to_close = 0;
  float gl_cost_of_opening_x = z;
  float *gl_lower = &work_mem_h[stride * nThread];
  // compute the number of centers to close if we are to open i
  for (int i = 0; i < num; i++) {
    if (is_center[i]) {
      float low = z;
      for (int j = 0; j < num; j++) {
        low += work_mem_h[j * stride + center_table[i]];
      }

      gl_lower[center_table[i]] = low;

      if (low > 0) {
        ++number_of_centers_to_close;
        work_mem_h[i * stride + K] -= low;
      }
    }
    gl_cost_of_opening_x += work_mem_h[i * stride + K];
  }

  //if opening a center at x saves cost (i.e. cost is negative) do so; otherwise, do nothing
  if (gl_cost_of_opening_x < 0) {
    for (int i = 0; i < num; i++) {
      bool close_center = gl_lower[center_table[points->p[i].assign]] > 0;
      if (switch_membership[i] || close_center) {
        points->p[i].cost = dist(points->p[i], points->p[x], dim) * points->p[i].weight;
        points->p[i].assign = x;
      }
    }

    for (int i = 0; i < num; i++) {
      if (is_center[i] && gl_lower[center_table[i]] > 0) {
        is_center[i] = false;
      }
    }

    if (x >= 0 && x < num) {
      is_center[x] = true;
    }
    *numcenters = *numcenters + 1 - number_of_centers_to_close;
  } else {
    gl_cost_of_opening_x = 0;
  }

  TIMESTAMP(t6);
  *serial_t += ELAPSED(t5, t6);

  //=======================================
  // DEALLOCATE HOST MEMORY
  //=======================================
  if (unified) {
    checkCudaErrors(cudaFree(work_mem_h));
  } else {
    free(work_mem_h);
  }

  //=======================================
  // DEALLOCATE GPU MEMORY
  //=======================================
  if (!unified) {
    checkCudaErrors(cudaFree(work_mem_d));
  }

  TIMESTAMP(t7);
  *free_t += ELAPSED(t6, t7);

  iter++;
  return -gl_cost_of_opening_x;
}

