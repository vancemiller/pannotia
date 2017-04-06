#include <argp.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <unistd.h>

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

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;


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

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                       Argument processing                                      //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

static char doc[] = "Rodinia LUD Benchmark";
static char args_doc[] = "";

static struct argp_option options[] = {
  {"base", 'b', "BASE", 0, "Number of rows/columns"},
  {"height", 'h', "HEIGHT", 0, "Pyramid height"},
  {"iterations", 'i', "ITERATIONS", 0, "Number of iterations"},
  {"temperature", 't', "FILEPATH", 0, "File containing initial temperature values of each cell"},
  {"power", 'p', "FILEPATH", 0, "File containing the dissipated power values of each cell"},
  {"output", 'o', "FILEPATH", 0, "Path to output file"},
  {"device", 'd', "DEVICE", 0, "CUDA Device ID"},
  {"unified", 'u', 0, 0, "Use unified memory"},
  {"verbose", 'v', 0, 0, "Verbose output"},
  {0},
};

struct arguments {
  size_t base;
  size_t height;
  size_t iterations;
  char* temperature;
  char* power;
  char* output;
  int device;
  bool unified;
  bool verbose;
};

static error_t parse_opt(int key, char* arg, struct argp_state* state) {
  struct arguments* args = (struct arguments*) state->input;
  switch (key) {
    case 'b':
      args->base = (int) strtol(arg, NULL, 0);
      break;
    case 'd':
      args->device = (int) strtol(arg, NULL, 0);
      break;
    case 'h':
      args->height = (int) strtol(arg, NULL, 0);
      break;
    case 'i':
      args->iterations= (int) strtol(arg, NULL, 0);
      break;
    case 'o':
      args->output = arg;
      break;
    case 'p':
      args->power = arg;
      break;
    case 't':
      args->temperature = arg;
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

void readinput(float* vect, int rows, int cols, char *file);
void writeoutput(float* vect, int rows, int cols, char* file);
float* compute_tran_temp(float* d_power, float* d_temperature[2], int col, int row, int iterations,
    int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows,
    cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                              Main                                              //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
  struct arguments args;
  // Defaults
  args.device = 0;
  args.temperature = NULL;
  args.power = NULL;
  args.output = NULL;
  args.base = 0;
  args.height = 0;
  args.iterations = 1;
  args.unified = false;
  args.verbose = false;
  // Parse command line arguments
  argp_parse(&argp, argc, argv, 0, 0, &args);
  if (!(args.base && args.height)) {
    fprintf(stderr, "Provide -b and -h flag. Use --help for help.\n");
    exit(EXIT_FAILURE);
  }
  if (!(args.power && args.temperature && args.output)) {
    fprintf(stderr, "Provide -p -t and -o flags for file locations. Use --help for help.\n");
    exit(EXIT_FAILURE);
  }

  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  size_t size = args.base * args.base;

  VPRINT(args.verbose, "WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);

  // add one iteration will extend the pyramid base by 2 per each borderline
# define EXPAND_RATE 2

  const int borderCols = (args.height) * EXPAND_RATE / 2;
  const int borderRows = borderCols;
  const int smallBlockCol = BLOCK_SIZE - (args.height) * EXPAND_RATE;
  const int smallBlockRow = smallBlockCol;
  const int blockCols = (args.base + smallBlockCol - 1) / smallBlockCol; // round up
  const int blockRows = blockCols;

  float* temperature;
  float* power;
  float* output;
  float* d_temperature[2];
  float* d_power;

  if (args.unified) {
    checkCudaErrors(cudaMallocManaged(&temperature, size * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&power, size * sizeof(float)));
    checkCudaErrors(cudaMallocManaged(&output, size * sizeof(float)));
  } else {
    temperature = (float *) malloc(size * sizeof(float));
    power = (float *) malloc(size * sizeof(float));
    output = (float *) calloc(size, sizeof(float));

    checkCudaErrors(cudaMalloc(&d_temperature[0], sizeof(float) * size));
    checkCudaErrors(cudaMalloc(&d_temperature[1], sizeof(float) * size));
    checkCudaErrors(cudaMalloc(&d_power, sizeof(float) * size));
  }

  if (!(power && temperature && output)) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(EXIT_FAILURE);
  }

  VPRINT(args.verbose, "pyramidHeight: %lu\ngridSize: [%lu, %lu]\nborder:[%d, %d]\n"
      "blockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n", args.height, args.base, args.base,
      borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);

  TIMESTAMP(start);

  readinput(temperature, args.base, args.base, args.temperature);
  readinput(power, args.base, args.base, args.power);

  if (args.unified) {
    d_temperature[0] = temperature;
    d_temperature[1] = output;
    d_power = power;
  } else {
    checkCudaErrors(cudaMemcpyAsync(d_temperature[0], temperature, sizeof(float) * size,
          cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_power, power, sizeof(float) * size, cudaMemcpyHostToDevice,
          stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
  }

  VPRINT(args.verbose, "Start computing the transient temperature\n");
  TIMESTAMP(kernel_start);
  float* result = compute_tran_temp(d_power, d_temperature, args.base, args.base,
      args.iterations, args.height, blockCols, blockRows, borderCols, borderRows, stream);
  TIMESTAMP(kernel_stop);
  VPRINT(args.verbose, "Ending simulation\n");

  if (!args.unified) {
    checkCudaErrors(cudaMemcpyAsync(output, result, sizeof(float) * size, cudaMemcpyDeviceToHost,
        stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    // copy to host memory but reassign pointer so we can use the same name for unified/normal
    result = output;
  }

  writeoutput(result, args.base, args.base, args.output);

  if (args.unified) {
    checkCudaErrors(cudaFree(temperature));
    checkCudaErrors(cudaFree(power));
    checkCudaErrors(cudaFree(output));
  } else {
    checkCudaErrors(cudaFree(d_power));
    checkCudaErrors(cudaFree(d_temperature[0]));
    checkCudaErrors(cudaFree(d_temperature[1]));
    free(temperature);
    free(power);
    free(output);
  }

  TIMESTAMP(stop);
  long long total_time = ELAPSED(start, stop);
  long long kernel_time = ELAPSED(kernel_start, kernel_stop);
  printf("\nTime total (including memory transfers)\t%f ms\n", (double) total_time * 1e-6);
  printf("Time for CUDA kernels:\t%f ms\n", (double) kernel_time * 1e-6);
}

void writeoutput(float* vect, int rows, int cols, char* file) {
  FILE* fp = fopen(file, "w");;
  if (!fp) {
    fprintf(stderr, "The file was not opened\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      fprintf(fp, "%d\t%g\n", i * cols + j, vect[i * cols + j]);
    }
  }
  fclose(fp);
}

void readinput(float* vect, int rows, int cols, char *file) {
  FILE* fp = fopen(file, "r");
  if (!fp) {
    fprintf(stderr, "The file was not opened\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i <= rows - 1; i++) {
    for (int j = 0; j <= cols - 1; j++) {
      float val;
      int result = fscanf(fp, "%f", &val);
      if (!result) {
        fprintf(stderr, "Not enough lines in file");
        exit(EXIT_FAILURE);
      }
      vect[i * cols + j] = val;
    }
  }
  fclose(fp);
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )

__global__ void calculate_temp(int iteration,  //number of iterations
    float *power,   //power input
    float *temp_src,    //temperature input/output
    float *temp_dst,    //temperature input/output
    int cols,  //Col of grid
    int rows,  //Row of grid
    int border_cols,  // border offset
    int border_rows,  // border offset
    float Cap,      //Capacitance
    float Rx, float Ry, float Rz, float step, float time_elapsed) {

  __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

  const float amb_temp = 80.0;
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const float step_div_Cap = step / Cap;
  const float Rx_1 = 1 / Rx;
  const float Ry_1 = 1 / Ry;
  const float Rz_1 = 1 / Rz;

  // each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  // calculate the small block size
  const int small_block_rows = BLOCK_SIZE - iteration * 2;  //EXPAND_RATE
  const int small_block_cols = BLOCK_SIZE - iteration * 2;  //EXPAND_RATE

  // calculate the boundary for the block according to
  // the boundary of its small block
  const int blkY = small_block_rows * by - border_rows;
  const int blkX = small_block_cols * bx - border_cols;
  const int blkYmax = blkY + BLOCK_SIZE - 1;
  const int blkXmax = blkX + BLOCK_SIZE - 1;

  // calculate the global thread coordination
  const int yidx = blkY + ty;
  const int xidx = blkX + tx;

  // load data if it is within the valid input range
  const int loadYidx = yidx, loadXidx = xidx;
  const int index = cols * loadYidx + loadXidx;

  if (IN_RANGE(loadYidx, 0, rows - 1) && IN_RANGE(loadXidx, 0, cols - 1)) {
    // Load the temperature data from global memory to shared memory
    temp_on_cuda[ty][tx] = temp_src[index];
    // Load the power data from global memory to shared memory
    power_on_cuda[ty][tx] = power[index];
  }
  __syncthreads();

  // effective range within this block that falls within
  // the valid range of the input data
  // used to rule out computation outside the boundary.
  const int validYmin = (blkY < 0) ? -blkY : 0;
  const int validYmax = (blkYmax > rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - rows + 1) :
      BLOCK_SIZE - 1;
  const int validXmin = (blkX < 0) ? -blkX : 0;
  const int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) :
      BLOCK_SIZE - 1;

  const int N = max(validYmin, ty - 1);
  const int S = min(validYmax, ty + 1);
  const int W = max(validXmin, tx - 1);
  const int E = min(validXmax, tx + 1);

  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && IN_RANGE(ty, i+1, BLOCK_SIZE - i - 2) &&
        IN_RANGE(tx, validXmin, validXmax) && IN_RANGE(ty, validYmin, validYmax)) {
      computed = true;
      temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx]
          + (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0 * temp_on_cuda[ty][tx]) * Ry_1
          + (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0 * temp_on_cuda[ty][tx]) * Rx_1
          + (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
    }
    __syncthreads();
    if (i == iteration - 1) {
      break;
    }
    if (computed) {
      //Assign the computation range
      temp_on_cuda[ty][tx] = temp_t[ty][tx];
    }
    __syncthreads();
  }
  // update the global memory
  // after the last iteration, only threads coordinated within the
  // small block perform the calculation and switch on ``computed''
  if (computed) {
    temp_dst[index] = temp_t[ty][tx];
  }
}

float* compute_tran_temp(float* d_power, float* d_temperature[2], int col, int row, int iterations,
    int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows,
    cudaStream_t stream) {
  const dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  const dim3 dimGrid(blockCols, blockRows);

  const float height = chip_height / row;
  const float width = chip_width / col;

  const float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * width * height;
  const float Rx = width / (2.0 * K_SI * t_chip * height);
  const float Ry = height / (2.0 * K_SI * t_chip * width);
  const float Rz = t_chip / (K_SI * height * width);

  const float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  const float step = PRECISION / max_slope;
  const float time_elapsed = 0.001;

  float* src = d_temperature[0];
  float* dst = d_temperature[1];

  for (int t = 0; t < iterations; t += num_iterations) {
    calculate_temp<<<dimGrid, dimBlock, 0, stream>>>(min(num_iterations, iterations - t),
        d_power, src, dst, col, row, borderCols, borderRows, Cap, Rx, Ry, Rz, step, time_elapsed);
    cudaStreamSynchronize(stream);
    float* tmp = src;
    src = dst;
    dst = tmp;
  }

  return dst;
}
