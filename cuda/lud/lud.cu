/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <argp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <errno.h>
#include <unistd.h>

extern "C" {
#include "lud_kernel.cuh"
}
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

#include "../timing.h"

#define VPRINT(verbose, format, ...) \
  if (verbose) {\
    fprintf(stdout, format, ## __VA_ARGS__);\
  }

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                       Argument processing                                      //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

static char doc[] = "Rodinia LUD Benchmark";
static char args_doc[] = "";

static struct argp_option options[] = {
  {"device", 'd', "DEVICE", 0, "CUDA Device ID"},
  {"file", 'f', "FILEPATH", 0, "Path to file containing input data."},
  {"size", 's', "SIZE", 0, "Generate input with SIZE elements. (Ignores file input)"},
  {"unified", 'u', 0, 0, "Use unified memory"},
  {"verbose", 'v', 0, 0, "Verbose output"},
  {0},
};

struct arguments {
  uint8_t device;
  char* file;
  uint32_t size;
  bool unified;
  bool verbose;
};

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
  struct arguments* args = (struct arguments*) state->input;
  switch (key) {
    case 'd':
      args->device = (int) strtol(arg, NULL, 0);
      break;
    case 'f':
      args->file = arg;
      break;
    case 's':
      args->size = (int) strtol(arg, NULL, 0);
      break;
    case 'u':
      args->unified = true;
      break;
    case 'v':
      args->verbose = true;
      break;
    default:
      return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc};

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                      Forward declarations                                      //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

void print_matrix(float *m, int size);
long long lud_cuda(float *m, int size, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                              Main                                              //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  struct arguments args;
  // Defaults
  args.device = 0;
  args.file = NULL;
  args.size = 0;
  args.unified = false;
  args.verbose = false;
  // Parse command line arguments
  argp_parse(&argp, argc, argv, 0, 0, &args);
  if (!args.size && !args.file) {
    fprintf(stderr, "Provide -s or -f flag. Use --help for help\n");
    exit(EXIT_FAILURE);
  }

  VPRINT(args.verbose, "WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  size_t size = args.size;

  float* m;
  float* d_m;

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  float time_pre = 0;
  float time_post = 0;
  float time_serial = 0;
  float time_copy_in = 0;
  float time_copy_out = 0;
  float time_kernel = 0;
  float time_malloc = 0;
  float time_free = 0;

  TIMESTAMP(t0);

  // Initialize data
  if (size) {
    if (args.unified) {
      checkCudaErrors(cudaMallocManaged(&m, sizeof(float) * size * size));
    } else {
      m = (float*) malloc(sizeof(float) * size * size);
    }
    if (!m) {
      fprintf(stderr, "Failed to allocate memory: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }
    TIMESTAMP(t1);
    time_malloc += ELAPSED(t0, t1);

    VPRINT(args.verbose, "Creating matrix internally size=%lu\n", size);
    const float lamda = -0.001;
    float coe[2 * size - 1];
    float coe_i = 0.0;

    for (int i = 0; i < size; i++) {
      coe_i = 10 * exp(lamda * i);
      int j = size - 1 + i;
      coe[j] = coe_i;
      j = size - 1 - i;
      coe[j] = coe_i;
    }

    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        m[i * size + j] = coe[size - 1 - i + j];
      }
    }
    TIMESTAMP(t2);
    time_pre += ELAPSED(t1, t2);
  } else {
    // File input
    VPRINT(args.verbose, "Reading matrix from file %s\n", args.file);

    FILE* fp = fopen(args.file, "rb");
    if (!fp) {
      fprintf(stderr, "Failed to open file: %s. %s\n", args.file, strerror(errno));
      exit(EXIT_FAILURE);
    }
    int ret = fscanf(fp, "%lu\n", &size);
    if (!ret) {
      fprintf(stderr, "Improperly formatted input file: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }

    TIMESTAMP(t1);
    time_pre += ELAPSED(t0, t1);

    if (args.unified) {
      checkCudaErrors(cudaMallocManaged(&m, sizeof(float) * size * size));
    } else {
      m = (float*) malloc(sizeof(float) * size * size);
    }
    if (!m) {
      fprintf(stderr, "Failed to allocate memory: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }
    TIMESTAMP(t2);
    time_malloc += ELAPSED(t1, t2);

    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        ret = fscanf(fp, "%f ", &m[i * size + j]);
        if (!ret) {
          fprintf(stderr, "Improperly formatted input file. Input ended early. %s\n",
          strerror(errno));
          exit(EXIT_FAILURE);
        }
      }
    }
    fclose(fp);
    TIMESTAMP(t3);
    time_pre += ELAPSED(t2, t3);
  }

  TIMESTAMP(t1);
  if (!args.unified) {
    cudaMalloc((void**) &d_m, size * size * sizeof(float));
    assert(d_m);
  }
  TIMESTAMP(t2);
  time_malloc += ELAPSED(t1, t2);

  if (args.unified) {
    d_m = m;
  } else {
    checkCudaErrors(cudaMemcpyAsync(d_m, m, size * size * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  TIMESTAMP(t3);
  time_copy_in += ELAPSED(t2, t3);

  time_kernel = lud_cuda(d_m, size, stream);

  TIMESTAMP(t4);
  if (!args.unified) {
    checkCudaErrors(cudaMemcpyAsync(m, d_m, size * size * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  TIMESTAMP(t5);
  time_copy_out += ELAPSED(t4, t5);

  // TODO something different
  // Access all data to bring it back to the host
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      m[i * size + j] += 1;
    }
  }
  TIMESTAMP(t6);
  time_post += ELAPSED(t5, t6);

  if (args.unified) {
    cudaFree(m);
  } else {
    cudaFree(d_m);
    free(m);
  }
  TIMESTAMP(t7);
  time_free += ELAPSED(t6, t7);

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                        Kernel Caller                                           //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////
long long lud_cuda(float *m, int size, cudaStream_t stream) {
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  float *m_debug = (float*) malloc(size * size * sizeof(float));

  int i;
  TIMESTAMP(start);
  for (i = 0; i < size - BLOCK_SIZE; i += BLOCK_SIZE) {
    lud_diagonal<<<1, BLOCK_SIZE, 0, stream>>>(m, size, i);
    checkCudaErrors(cudaStreamSynchronize(stream));
    lud_perimeter<<<(size - i) / BLOCK_SIZE - 1, BLOCK_SIZE * 2, 0, stream>>>(m, size,
        i);
    checkCudaErrors(cudaStreamSynchronize(stream));
    dim3 dimGrid((size - i) / BLOCK_SIZE - 1, (size - i) / BLOCK_SIZE - 1);
    lud_internal<<<dimGrid, dimBlock, 0, stream>>>(m, size, i);
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  lud_diagonal<<<1, BLOCK_SIZE, 0, stream>>>(m, size, i);
  checkCudaErrors(cudaStreamSynchronize(stream));
  TIMESTAMP(stop);
  return ELAPSED(start, stop);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                        Helper Functions                                        //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

void print_matrix(float *m, int size) {
  int i, j;
  for (i = 0; i < size; i++) {
    for (j = 0; j < size; j++)
      printf("%f ", m[i * size + j]);
    printf("\n");
  }
}
