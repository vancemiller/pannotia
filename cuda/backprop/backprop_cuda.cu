// includes, system
#include <argp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

// includes, kernels
#include "helper_cuda.h"
#include "backprop_cuda_kernel.cu"
#include "backprop.h"

#define TIMESTAMP(NAME) \
  struct timespec NAME; \
  if (clock_gettime(CLOCK_MONOTONIC, &NAME)) { \
    fprintf(stderr, "Failed to get time: %s\n", strerror(errno)); \
  }

#define ELAPSED(start, end) \
  ((uint64_t) 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)

#define VPRINT(verbose, format, ...) \
  if (verbose) {\
    fprintf(stdout, format, ## __VA_ARGS__);\
  }

// global timing vars
long long time_pre = 0;
long long time_post = 0;
long long time_serial = 0;
long long time_copy_in = 0;
long long time_copy_out = 0;
long long time_kernel = 0;
long long time_malloc = 0;
long long time_free = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                       Argument processing                                      //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

static char doc[] = "Rodinia backprop Benchmark";
static char args_doc[] = "";

static struct argp_option options[] = {
  {"size", 's', "SIZE", 0, "Number of input elements"},
  {"device", 'd', "DEVICE", 0, "CUDA Device ID"},
  {"unified", 'u', 0, 0, "Use unified memory"},
  {"verbose", 'v', 0, 0, "Verbose output"},
  {0},
};

struct arguments {
  size_t size;
  int device;
  bool unified;
  bool verbose;
};

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
  struct arguments* args = (struct arguments*) state->input;
  switch (key) {
    case 'd':
      args->device = (int) strtol(arg, NULL, 0);
      break;
    case 's':
      args->size = (size_t) strtol(arg, NULL, 0);
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


void bpnn_layerforward(float* l1, float* l2, float* conn, int n1, int n2);

float bpnn_output_error(float* delta, float* target, float* output, int nj);

float bpnn_hidden_error(float* delta_h, int nh, float* delta_o, int no, float* who,
    float* hidden);

void bpnn_adjust_weights(float* delta, int ndelta, float* ly, int nly, float* w, float* oldw);

float squash(float x);

BPNN* bpnn_create(int n_in, int n_hidden, int n_out, bool unified);

float* alloc_1d_dbl(int n, bool unified);

float* alloc_2d_dbl(int m, int n, bool unified);

void bpnn_randomize_weights(float* w, int m, int n);

void bpnn_randomize_row(float* w, int m);

void bpnn_zero_weights(float* w, int m, int n);

BPNN* bpnn_internal_create(int n_in, int n_hidden, int n_out, bool unified);

void bpnn_free(BPNN* net, bool unified);

void bpnn_train_cuda(BPNN *net, cudaStream_t stream, bool unified);

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                              Main                                              //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  struct arguments args;
  // Defaults
  args.device = 0;
  args.size = 0;
  args.unified = false;
  args.verbose = false;
  // Parse command line arguments
  argp_parse(&argp, argc, argv, 0, 0, &args);
  if (!(args.size)) {
    fprintf(stderr, "Provide -s flag. Use --help for help.\n");
    exit(EXIT_FAILURE);
  }
  if (args.size % 16 != 0) {
    fprintf(stderr, "The number of input points must be divisible by 16\n");
    exit(EXIT_FAILURE);
  }

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  // TODO make argument
  int seed = 7;
  srand(seed);

  TIMESTAMP(t0);
  BPNN* net = bpnn_create(args.size, 16, 1, args.unified); // (16, 1 can not be changed)
  TIMESTAMP(t1);
  time_malloc += ELAPSED(t0, t1);
  VPRINT(args.verbose, "Input layer size : %lu\n", args.size);
  for (int i = 0; i < args.size; i++) {
    net->input_units[i] = (float) rand() / RAND_MAX;
  }
  TIMESTAMP(t2);
  time_pre+= ELAPSED(t1, t2);

  VPRINT(args.verbose, "Starting training kernel\n");
  bpnn_train_cuda(net, stream, args.unified);
  bpnn_free(net, args.unified);
  VPRINT(args.verbose, "Training done\n");
  TIMESTAMP(t3);

  printf("====Timing info====\n");
  printf("time malloc = %f ms\n", time_malloc * 1e-6);
  printf("time pre = %f ms\n", time_pre * 1e-6);
  printf("time CPU to GPU memory copy = %f ms\n", time_copy_in * 1e-6);
  printf("time kernel = %f ms\n", time_kernel * 1e-6);
  printf("time serial = %f ms\n", time_serial * 1e-6);
  printf("time GPU to CPU memory copy back = %f ms\n", time_copy_out * 1e-6);
  printf("time post = %f ms\n", time_post * 1e-6);
  printf("time free = %f ms\n", time_free * 1e-6);
  printf("End-to-end = %f ms\n", ELAPSED(t0, t3) * 1e-6);
  exit(EXIT_SUCCESS);
}

////////////////////////////
// Helper functions
////////////////////////////

void bpnn_layerforward(float* l1, float* l2, float* conn, int n1, int n2) {
  /*** Set up thresholding unit ***/
  l1[0] = 1.0;
  /*** For each unit in second layer ***/
  for (int j = 1; j <= n2; j++) {
    /*** Compute weighted sum of its inputs ***/
    float sum = 0.0;
    for (int k = 0; k <= n1; k++) {
      sum += conn[k * n2 + j] * l1[k];
    }
    l2[j] = squash(sum);
  }
}

float bpnn_output_error(float* delta, float* target, float* output, int nj) {
  float errsum = 0.0;
  for (int j = 1; j <= nj; j++) {
    float o = output[j];
    float t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    errsum += abs(delta[j]);
  }
  return errsum;
}

float bpnn_hidden_error(float* delta_h, int nh, float* delta_o, int no, float* who,
    float* hidden) {
  float errsum = 0.0;
  for (int j = 1; j <= nh; j++) {
    float h = hidden[j];
    float sum = 0.0;
    for (int k = 1; k <= no; k++) {
      sum += delta_o[k] * who[j * no + k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    errsum += abs(delta_h[j]);
  }
  return errsum;
}

void bpnn_adjust_weights(float* delta, int ndelta, float* ly, int nly, float* w, float* oldw) {
  ly[0] = 1.0;
  for (int j = 1; j <= ndelta; j++) {
    for (int k = 0; k <= nly; k++) {
      float new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k * ndelta + j]));
      w[k * ndelta + j] += new_dw;
      oldw[k * ndelta + j] = new_dw;
    }
  }
}

/*** The squashing function.  Currently, it's a sigmoid. ***/
float squash(float x) {
  return 1.0 / (1.0 + exp(-x));
}

/*** Allocate 1d array of floats ***/
float* alloc_1d_dbl(int n, bool unified) {
  float* mem;
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&mem, n * sizeof(float)));
  } else {
    mem = (float*) malloc((n * sizeof(float)));
  }
  if (!mem) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of floats\n");
    exit(EXIT_FAILURE);
  }
  return mem;
}

/*** Allocate 2d array of floats ***/
float* alloc_2d_dbl(int m, int n, bool unified) {
  float* mem;
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&mem, n * m * sizeof(float)));
  } else {
    mem = (float*) malloc(n * m * sizeof(float));
  }
  if (!mem) {
    fprintf(stderr, "ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    exit(EXIT_FAILURE);
  }
  return mem;
}

void bpnn_randomize_weights(float* w, int m, int n) {
  for (int i = 0; i <= m; i++) {
    for (int j = 0; j <= n; j++) {
      w[i* n + j] = (float) rand() / RAND_MAX;
    }
  }
}

void bpnn_randomize_row(float* w, int m) {
  for (int i = 0; i <= m; i++) {
    w[i] = 0.1;
  }
}


void bpnn_zero_weights(float* w, int m, int n) {
  for (int i = 0; i <= m; i++) {
    for (int j = 0; j <= n; j++) {
      w[i * n + j] = 0.0;
    }
  }
}

BPNN* bpnn_internal_create(int n_in, int n_hidden, int n_out, bool unified) {
  BPNN* newnet;
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&newnet, sizeof(BPNN)));
  } else {
    newnet = (BPNN*) malloc(sizeof(BPNN));
  }
  if (!newnet) {
    fprintf(stderr, "BPNN_CREATE: Couldn't allocate neural network\n");
    exit(EXIT_FAILURE);
  }

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in + 1, unified);
  newnet->hidden_units = alloc_1d_dbl(n_hidden + 1, unified);
  newnet->output_units = alloc_1d_dbl(n_out + 1, unified);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden + 1, unified);
  newnet->output_delta = alloc_1d_dbl(n_out + 1, unified);
  newnet->target = alloc_1d_dbl(n_out + 1, unified);

  newnet->input_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1, unified);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1, unified);

  newnet->input_prev_weights = alloc_2d_dbl(n_in + 1, n_hidden + 1, unified);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden + 1, n_out + 1, unified);

  return newnet;
}

void bpnn_free(BPNN* net, bool unified) {
  TIMESTAMP(t0);
  if (unified) {
    checkCudaErrors(cudaFree(net->input_units));
    checkCudaErrors(cudaFree(net->hidden_units));
    checkCudaErrors(cudaFree(net->output_units));
    checkCudaErrors(cudaFree(net->hidden_delta));
    checkCudaErrors(cudaFree(net->output_delta));
    checkCudaErrors(cudaFree(net->target));
    checkCudaErrors(cudaFree(net->input_weights));
    checkCudaErrors(cudaFree(net->input_prev_weights));
    checkCudaErrors(cudaFree(net->hidden_weights));
    checkCudaErrors(cudaFree(net->hidden_prev_weights));
    checkCudaErrors(cudaFree(net));
  } else {
    free(net->input_units);
    free(net->hidden_units);
    free(net->output_units);
    free(net->hidden_delta);
    free(net->output_delta);
    free(net->target);
    free(net->input_weights);
    free(net->input_prev_weights);
    free(net->hidden_weights);
    free(net->hidden_prev_weights);
    free(net);
  }
  TIMESTAMP(t1);
  time_free += ELAPSED(t0, t1);
}

/*** Creates a new fully-connected network from scratch,
     with the given numbers of input, hidden, and output units.
     Threshold units are automatically included.  All weights are
     randomly initialized.

     Space is also allocated for temporary storage (momentum weights,
     error computations, etc).
***/
BPNN* bpnn_create(int n_in, int n_hidden, int n_out, bool unified) {
  BPNN* newnet = bpnn_internal_create(n_in, n_hidden, n_out, unified);

#ifdef INITZERO
  bpnn_zero_weights(newnet->input_weights, n_in, n_hidden);
#else
  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
#endif
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);
  return newnet;
}

void bpnn_train_cuda(BPNN* net, cudaStream_t stream, bool unified) {
  int in = net->input_n;
  int hid = net->hidden_n;
  int out = net->output_n;

  int num_blocks = in / 16;
  dim3  grid(1 , num_blocks);
  dim3  threads(16 , 16);

  float* partial_sum;
  float* input_cuda;
  float* input_hidden_cuda;
  float* output_hidden_cuda;
  float* hidden_partial_sum;
  TIMESTAMP(t0);
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&partial_sum, num_blocks * WIDTH * sizeof(float)));
  } else {
    partial_sum = (float*) malloc(num_blocks * WIDTH * sizeof(float));
    checkCudaErrors(cudaMalloc(&input_cuda, (in + 1) * sizeof(float)));
    checkCudaErrors(cudaMalloc(&input_hidden_cuda, (in + 1) * (hid + 1) * sizeof(float)));
    checkCudaErrors(cudaMalloc(&hidden_partial_sum, num_blocks * WIDTH * sizeof(float)));
  }
  checkCudaErrors(cudaMalloc(&output_hidden_cuda, (hid + 1) * sizeof(float)));
  TIMESTAMP(t1);
  time_malloc += ELAPSED(t0, t1);


  if (unified) {
    input_cuda = net->input_units;
    input_hidden_cuda = net->input_weights;
    hidden_partial_sum = partial_sum;
  } else {
    checkCudaErrors(cudaMemcpy(input_cuda, net->input_units, (in + 1) * sizeof(float),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(input_hidden_cuda, net->input_weights, (in + 1) * (hid + 1) *
      sizeof(float), cudaMemcpyHostToDevice));
  }
  TIMESTAMP(t2);
  time_copy_in += ELAPSED(t1, t2);

  printf("Performing GPU computation\n");

  bpnn_layerforward_CUDA<<<grid, threads, 0, stream>>>(input_cuda, output_hidden_cuda,
      input_hidden_cuda, hidden_partial_sum, in, hid);
  checkCudaErrors(cudaStreamSynchronize(stream));
  TIMESTAMP(t3);
  time_kernel += ELAPSED(t2, t3);

  if (!unified) {
    checkCudaErrors(cudaMemcpy(partial_sum, hidden_partial_sum, num_blocks * WIDTH * sizeof(float),
        cudaMemcpyDeviceToHost));
  }
  TIMESTAMP(t4);
  time_copy_out += ELAPSED(t3, t4);

  for (int j = 1; j <= hid; j++) {
    float sum = 0.0;
    for (int k = 0; k < num_blocks; k++) {
      sum += partial_sum[k * hid + j - 1] ;
    }
    sum += net->input_weights[j];
    net->hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
  }

  bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
  float out_err = bpnn_output_error(net->output_delta, net->target, net->output_units, out);
  float hid_err = bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
      net->hidden_weights, net->hidden_units);
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights,
      net->hidden_prev_weights);

  float* hidden_delta_cuda;
  float* input_prev_weights_cuda;

  TIMESTAMP(t5);
  time_post += ELAPSED(t4, t5);

  if (!unified) {
    checkCudaErrors(cudaMalloc(&hidden_delta_cuda, (hid + 1) * sizeof(float)));
    checkCudaErrors(cudaMalloc(&input_prev_weights_cuda, (in + 1) * (hid + 1) * sizeof(float)));
  }
  TIMESTAMP(t6);
  time_malloc += ELAPSED(t5, t6);

  if (unified) {
    hidden_delta_cuda = net->hidden_delta;
    input_prev_weights_cuda = net->input_prev_weights;
  } else {
    checkCudaErrors(cudaMemcpy(hidden_delta_cuda, net->hidden_delta, (hid + 1) * sizeof(float),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(input_prev_weights_cuda, net->input_prev_weights,
        (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(input_hidden_cuda, net->input_weights,
        (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyHostToDevice));
  }
  TIMESTAMP(t7);
  time_copy_in += ELAPSED(t6, t7);

  bpnn_adjust_weights_cuda<<<grid, threads, 0, stream>>>(hidden_delta_cuda, hid, input_cuda, in,
      input_hidden_cuda, input_prev_weights_cuda);
  checkCudaErrors(cudaStreamSynchronize(stream));
  TIMESTAMP(t8);
  time_kernel += ELAPSED(t7, t8);

  if (!unified) {
    checkCudaErrors(cudaMemcpy(net->input_units, input_cuda, (in + 1) * sizeof(float),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(net->input_weights, input_hidden_cuda,
        (in + 1) * (hid + 1) * sizeof(float), cudaMemcpyDeviceToHost));
  }
  TIMESTAMP(t9);
  time_copy_out += ELAPSED(t8, t9);

  if (unified) {
    checkCudaErrors(cudaFree(partial_sum));
  } else {
    checkCudaErrors(cudaFree(input_cuda));
    checkCudaErrors(cudaFree(output_hidden_cuda));
    checkCudaErrors(cudaFree(input_hidden_cuda));
    checkCudaErrors(cudaFree(hidden_partial_sum));
    checkCudaErrors(cudaFree(input_prev_weights_cuda));
    checkCudaErrors(cudaFree(hidden_delta_cuda));
    free(partial_sum);
  }
  TIMESTAMP(t10);
  time_free += ELAPSED(t9, t10);
}
