/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.
 **   The sequential version is gaussian.c.  This parallel
 **   implementation converts three independent for() loops
 **   into three Fans.  Use the data file ge_3.dat to verify
 **   the correction of the output.
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 ** Updated by Vance Miller for CUDA 8.0, 04/2017
 **-----------------------------------------------------------
 */
#include <argp.h>
#include <assert.h>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda.h"
#include "helper_cuda.h"

#ifdef RD_WG_SIZE_0_0
#define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define MAXBLOCKSIZE RD_WG_SIZE
#else
#define MAXBLOCKSIZE 512
#endif

//2D defines. Go from specific to general
#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE_XY RD_WG_SIZE
#else
#define BLOCK_SIZE_XY 4
#endif

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

static char doc[] = "Rodinia Gaussian Benchmark";
static char args_doc[] = "The program is to solve a linear system Ax = b"
"  by using Gaussian Elimination. The algorithm on page 101"
"  (\"Foundations of Parallel Programming\") is used."
"  The sequential version is gaussian.c.  This parallel"
"  implementation converts three independent for() loops"
"  into three Fans.  Use the data file ge_3.dat to verify"
"the correction of the output.";

static struct argp_option options[] = {
  {"device", 'd', "DEVICE", 0, "CUDA Device ID"},
  {"file", 'f', "FILEPATH", 0, "Path to file containing input data.\n"
    "The first line of the file contains the dimension of the matrix, n."
      "The second line of the file is a newline.\n"
      "The next n lines contain n tab separated values for the matrix."
      "The next line of the file is a newline.\n"
      "The next line of the file is a 1xn vector with tab separated values.\n"
      "The next line of the file is a newline. (optional)\n"
      "The final line of the file is the pre-computed solution. (optional)\n"
      "Example: matrix4.txt:\n4\n\n"
      "-0.6	-0.5	0.7	0.3\n"
      "-0.3	-0.9	0.3	0.7\n"
      "-0.4	-0.5	-0.3	-0.8\n"
      "0.0	-0.1	0.2	0.9\n\n"
      "-0.85	-0.68	0.24	-0.53\n\n"
      "0.7	0.0	-0.4	-0.5\n"},
  {"size", 's', "SIZE", 0, "Generate a matrix with SIZE elements. (Ignores file input)"},
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
//                                     Forward declarations                                       //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////
void InitFromFile(char *filename, bool unified, float** matrix_a, float** vector_b,
    float** matrix_m);
long long ForwardSub(int size, bool unified, float* matrix_a, float* vector_b, float* matrix_m);
void BackSub(int size, float** result, float* matrix_a, float* vector_b, float* matrix_m);
__global__ void Fan1(float *m, float *a, int size, int t);
__global__ void Fan2(float *m, float *a, float *b,int size, int t);
void InitMatrix(FILE* fp, float* matrix, int rows, int cols);
void InitVector(FILE* fp, float* vector, int size);
void PrintMatrix(float* matrix, int rows, int cols);
void PrintVector(float* vector, int size);

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                //
//                                          Data Creation                                         //
//                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void create_matrix(float *m, int size) {
  const float lamda = -0.01;
  float coe_i = 0.0;
  float coe[2 * size - 1];

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
}

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
    fprintf(stderr, "Provide -s or -f flag. Use -h for help\n");
    exit(EXIT_FAILURE);
  }

  VPRINT(args.verbose, "WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", MAXBLOCKSIZE,
      BLOCK_SIZE_XY, BLOCK_SIZE_XY);

  float* matrix_a;
  float* vector_b;
  float* matrix_m;

  //begin timing
  TIMESTAMP(start);

  if (args.size) {
    VPRINT(args.verbose, "Create matrix internally in parse, size = %d \n", (int) args.size);
    if (args.unified) {
      checkCudaErrors(cudaMallocManaged(&matrix_a, args.size * args.size * sizeof(float),
            cudaMemAttachHost));
      checkCudaErrors(cudaMallocManaged(&vector_b, args.size * sizeof(float), cudaMemAttachHost));
      checkCudaErrors(cudaMallocManaged(&matrix_m, args.size * args.size * sizeof(float),
            cudaMemAttachHost));
    } else {
      matrix_a = (float*) malloc(args.size * args.size * sizeof(float));
      vector_b = (float*) malloc(args.size * sizeof(float));
      matrix_m = (float*) malloc(args.size * args.size * sizeof(float));
    }
    assert(matrix_a && vector_b && matrix_m);

    // initialize data
    create_matrix(matrix_a, args.size);
    memset(vector_b, 1.0, args.size);
    memset(matrix_m, 0.0, args.size * args.size);
  } else {
    // file input
    VPRINT(args.verbose, "Read file from %s \n", args.file);
    InitFromFile(args.file, args.unified, &matrix_a, &vector_b, &matrix_m);
  }


  // run kernels
  long long kernel_time = ForwardSub(args.unified, args.unified, matrix_a, vector_b, matrix_m);

  if (args.verbose) {
    printf("Matrix m is: \n");
    PrintMatrix(matrix_m, args.size, args.size);
    printf("Matrix a is: \n");
    PrintMatrix(matrix_a, args.size, args.size);
    printf("Vector b is: \n");
    PrintVector(vector_b, args.size);
  }

  float* result;
  BackSub(args.size, &result, matrix_a, vector_b, matrix_m);

  //end timing
  TIMESTAMP(stop);
  long long total_time = ELAPSED(start, stop);

  if (args.verbose) {
    printf("The final solution is: \n");
    PrintVector(result, args.size);
  }

  printf("\nTime total (including memory transfers)\t%f ms\n", (double) total_time * 1e-6);
  printf("Time for CUDA kernels:\t%f ms\n", (double) kernel_time * 1e-6);

  if (args.unified) {
    checkCudaErrors(cudaFree(matrix_m));
    checkCudaErrors(cudaFree(matrix_a));
    checkCudaErrors(cudaFree(vector_b));
  } else {
    free(matrix_m);
    free(matrix_a);
    free(vector_b);
  }
  free(result);
}


/*------------------------------------------------------
 ** InitFromFile -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **------------------------------------------------------
 */
void InitFromFile(char *filename, bool unified, float** matrix_a, float** vector_b, float** matrix_m) {
  FILE* fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Invalid input file: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t size;
  int ret = fscanf(fp, "%lu", &size);
  if (!ret) {
    fprintf(stderr, "Improperly formatted input file: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (unified) {
    checkCudaErrors(cudaMallocManaged(matrix_a, size * size * sizeof(float),
          cudaMemAttachHost));
    checkCudaErrors(cudaMallocManaged(vector_b, size * sizeof(float), cudaMemAttachHost));
    checkCudaErrors(cudaMallocManaged(matrix_m, size * size * sizeof(float),
          cudaMemAttachHost));
  } else {
    *matrix_a = (float*) malloc(size * size * sizeof(float));
    *vector_b = (float*) malloc(size * sizeof(float));
    *matrix_m = (float*) malloc(size * size * sizeof(float));
  }
  assert(*matrix_a && *vector_b && *matrix_m);

  InitMatrix(fp, *matrix_a, size, size);
  InitVector(fp, *vector_b, size);
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float* d_matrix_m, float* d_matrix_a, int size, int t) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= size - 1 - t)
    return;
  d_matrix_m[size * (index + t + 1) + t] = d_matrix_a[size * (index + t + 1) + t] /
      d_matrix_a[size * t + t];
  // TODO shouldn't the kernel iterate over t instead of having the host launch each t-th kernel?
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */

__global__ void Fan2(float *d_matrix_m, float *d_matrix_a, float *d_vector_b,int size, int t)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= size - 1 - t || y >= size - t)
    return;

  d_matrix_a[size * (x + 1 + t) + (y + t)] -= d_matrix_m[size * (x + 1 + t) + t] *
      d_matrix_a[size * t + (y + t)];
  if (y == 0) {
    d_vector_b[x + 1 + t] -= d_matrix_m[size * (x + 1 + t) + (y + t)] * d_vector_b[t];
  }
  // TODO shouldn't the kernel iterate over t instead of having the host launch each t-th kernel?
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
long long ForwardSub(int size, bool unified, float* matrix_a, float* vector_b, float* matrix_m) {
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));

  float* d_matrix_m;
  float* d_matrix_a;
  float* d_vector_b;

  // allocate memory on GPU
  if (unified) {
    d_matrix_a = matrix_a;
    d_vector_b = vector_b;
    d_matrix_m = matrix_m;
  } else {
    cudaMalloc((void **) &d_matrix_m, size * size * sizeof(float));
    cudaMalloc((void **) &d_matrix_a, size * size * sizeof(float));
    cudaMalloc((void **) &d_vector_b, size * sizeof(float));
    // copy memory to GPU
    cudaMemcpy(d_matrix_m, matrix_m, size * size * sizeof(float),cudaMemcpyHostToDevice );
    cudaMemcpy(d_matrix_a, matrix_a, size * size * sizeof(float),cudaMemcpyHostToDevice );
    cudaMemcpy(d_vector_b, vector_b, size * sizeof(float),cudaMemcpyHostToDevice );
  }

  int block_size = MAXBLOCKSIZE;
  int grid_size = (size + block_size - 1) / block_size; // round up

  dim3 dimBlock(block_size);
  dim3 dimGrid(grid_size);

  int blockSize2d = BLOCK_SIZE_XY;
  int gridSize2d = (size + blockSize2d - 1) /blockSize2d; // round up

  dim3 dimBlockXY(blockSize2d, blockSize2d);
  dim3 dimGridXY(gridSize2d, gridSize2d);

  // begin timing kernels
  TIMESTAMP(start);
  for (int i = 0; i < size - 1; i++) {
    Fan1<<<dimGrid, dimBlock, 0, stream>>>(d_matrix_m, d_matrix_a, size, i);
    checkCudaErrors(cudaStreamSynchronize(stream));
    Fan2<<<dimGridXY, dimBlockXY>>>(d_matrix_m, d_matrix_a, d_vector_b, size, i);
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  // end timing kernels
  TIMESTAMP(stop);

  if (!unified) {
    // copy memory back to CPU
    cudaMemcpy(matrix_m, d_matrix_m, size * size * sizeof(float),cudaMemcpyDeviceToHost );
    cudaMemcpy(matrix_a, d_matrix_a, size * size * sizeof(float),cudaMemcpyDeviceToHost );
    cudaMemcpy(vector_b, d_vector_b, size * sizeof(float),cudaMemcpyDeviceToHost );
    cudaFree(d_matrix_m);
    cudaFree(d_matrix_a);
    cudaFree(d_vector_b);
  }
  return ELAPSED(start, stop);
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub(int size, float** result, float* matrix_a, float* vector_b, float* matrix_m) {
  // create a new vector to hold the final answer
  *result = (float*) malloc(size * sizeof(float));
  // solve "bottom up"
  for(int i = 0; i < size; i++) {
    (*result)[size - i - 1] = vector_b[size - i - 1];
    for(int j = 0; j < i; j++) {
      (*result)[size - i - 1] -= matrix_a[size * (size - i - 1) + (size - j - 1)] *
          (*result)[size - j - 1];
    }
    (*result)[size - i - 1] = (*result)[size - i - 1] / matrix_a[size * (size - i - 1) +
        (size - i - 1)];
  }
}

/*------------------------------------------------------
 ** InitMatrix() -- Initialize the matrix by reading
 ** data from the file
 **------------------------------------------------------
 */
void InitMatrix(FILE* fp, float* matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int result = fscanf(fp, "%f", matrix + rows * i + j);
      if (!result) {
        fprintf(stderr, "Improperly formatted input file. Matrix ended early. %s\n",
            strerror(errno));
        exit(EXIT_FAILURE);
      }
    }
  }
}

/*------------------------------------------------------
 ** PrintMatrix() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMatrix(float* matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%8.2f ", *(matrix + rows * i + j));
    }
    printf("\n");
  }
  printf("\n");
}

/*------------------------------------------------------
 ** InitVector() -- Initialize the vector by reading
 ** data from the file
 **------------------------------------------------------
 */
void InitVector(FILE* fp, float* vector, int size) {
  for (int i = 0; i < size; i++) {
    int result = fscanf(fp, "%f",  &vector[i]);
    if (!result) {
      fprintf(stderr, "Improperly formatted input file. Vector ended early. %s\n",
          strerror(errno));
      exit(EXIT_FAILURE);
    }
  }
}

/*------------------------------------------------------
 ** PrintVector() -- Print the contents of the vector
 **------------------------------------------------------
 */
void PrintVector(float* vector, int size) {
  for (int i = 0; i < size; i++) {
    printf("%.2f ", vector[i]);
  }
  printf("\n\n");
}

