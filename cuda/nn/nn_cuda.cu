/*
 * nn.cu
 * Nearest Neighbor
 *
 */

#include <stdio.h>
#include <time.h>
#include <float.h>
#include <vector>
#include "cuda.h"
#include <assert.h>
#include <errno.h>

#define min( a, b )     a > b ? b : a
#define ceilDiv( a, b )   ( a + b - 1 ) / b
#define print( x )      printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG       false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28 // character position of the latitude value in each record
#define OPEN 10000  // initial value of nearest neighbors

#include "../timing.h"

float time_pre = 0;
float time_post = 0;
float time_serial = 0;
float time_copy_in = 0;
float time_copy_out = 0;
float time_kernel = 0;
float time_malloc = 0;
float time_free = 0;

typedef struct latLong {
  float lat;
  float lng;
} LatLong;

typedef struct record {
  char recString[REC_LENGTH];
  float distance;
} Record;

int loadData(char *filename, std::vector<Record> &records, LatLong** locations, bool unified);
void findLowest(std::vector<Record> &records, float *distances, int numRecords, int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename, int *r, float *lat, float *lng, int *q,
    int *t, int *p, int *d, bool* u);

/**
 * Kernel
 * Executed on GPU
 * Calculates the Euclidean distance from each record in the database to the target position
 */
__global__ void euclid(LatLong* locations, float* distances, int numRecords, float lat, float lng) {
  int globalId = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
  LatLong* latLong = locations + globalId;
  if (globalId < numRecords) {
    float* dist = distances + globalId;
    *dist = (float) sqrt(
        (lat - latLong->lat) * (lat - latLong->lat) + (lng - latLong->lng) * (lng - latLong->lng));
  }
}

/**
 * This program finds the k-nearest neighbors
 **/
int main(int argc, char* argv[]) {
  int i = 0;
  float lat, lng;
  int quiet = 0, timing = 0, platform = 0, device = 0;
  bool unified = false;

  std::vector<Record> records;
  LatLong* locations;
  char filename[100];
  int resultsCount = 10;

  // parse command line
  if (parseCommandline(argc, argv, filename, &resultsCount, &lat, &lng, &quiet, &timing, &platform,
        &device, &unified)) {
    printUsage();
    return 0;
  }

  TIMESTAMP(t0)
  int numRecords = loadData(filename, records, &locations, unified);
  if (resultsCount > numRecords)
    resultsCount = numRecords;

  //Pointers to host memory
  float *distances;
  //Pointers to device memory
  LatLong *d_locations;
  float *d_distances;

  // Scaling calculations - added by Sam Kauffman
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cudaDeviceSynchronize();
  unsigned long maxGridX = deviceProp.maxGridSize[0];
  unsigned long threadsPerBlock = min(deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK);
  size_t totalDeviceMemory;
  size_t freeDeviceMemory;
  cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);
  cudaDeviceSynchronize();
  unsigned long usableDeviceMemory = freeDeviceMemory * 85 / 100; // 85% arbitrary throttle to compensate for known CUDA bug
  unsigned long maxThreads = usableDeviceMemory / 12; // 4 bytes in 3 vectors per thread
  if (numRecords > maxThreads) {
    fprintf(stderr, "Error: Input too large.\n");
    exit(1);
  }
  unsigned long blocks = ceilDiv(numRecords, threadsPerBlock); // extra threads will do nothing
  unsigned long gridY = ceilDiv(blocks, maxGridX);
  unsigned long gridX = ceilDiv(blocks, gridY);
  // There will be no more than (gridY - 1) extra blocks
  dim3 gridDim(gridX, gridY);

  if (DEBUG) {
    print(totalDeviceMemory); // 804454400
    print(freeDeviceMemory);
    print(usableDeviceMemory);
    print(maxGridX); // 65535
    print(deviceProp.maxThreadsPerBlock); // 1024
    print(threadsPerBlock);
    print(maxThreads);
    print(blocks); // 130933
    print(gridY);
    print(gridX);
  }

  TIMESTAMP(t1);
  time_pre += ELAPSED(t0, t1);
  /**
   * Allocate memory on host and device
   */
  if (!unified) {
    distances = (float *) malloc(sizeof(float) * numRecords);
    cudaMalloc((void **) &d_locations, sizeof(LatLong) * numRecords);
    cudaMalloc((void **) &d_distances, sizeof(float) * numRecords);
  } else {
    assert(cudaMallocManaged(&distances, sizeof(float) * numRecords) == cudaSuccess);
  }

  TIMESTAMP(t2);
  time_malloc += ELAPSED(t1, t2);

  /**
   * Transfer data from host to device
   */
  if (!unified) {
    cudaMemcpy(d_locations, locations, sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice);
  }
  TIMESTAMP(t3);
  time_copy_in += ELAPSED(t2, t3);

  /**
   * Execute kernel
   */
  if (!unified) {
    euclid<<<gridDim, threadsPerBlock>>>(d_locations, d_distances, numRecords, lat, lng);
  } else {
    euclid<<<gridDim, threadsPerBlock>>>(locations, distances, numRecords, lat, lng);
  }
  TIMESTAMP(t4);
  time_kernel += ELAPSED(t3, t4);

  /**
   * Copy data from device memory to host memory
   */
  if (!unified) {
    cudaMemcpy(distances, d_distances, sizeof(float) * numRecords, cudaMemcpyDeviceToHost);
  }
  TIMESTAMP(t5);
  time_copy_out += ELAPSED(t4, t5);

  // find the resultsCount least distances
  findLowest(records, distances, numRecords, resultsCount);

  // print out results
  if (!quiet) {
    for (i = 0; i < resultsCount; i++) {
      printf("%s --> Distance=%f\n", records[i].recString, records[i].distance);
    }
  }
  TIMESTAMP(t6);
  time_post += ELAPSED(t5, t6);

  if (!unified) {
    free(locations);
    free(distances);
    //Free memory
    cudaFree(d_locations);
    cudaFree(d_distances);
  } else {
    cudaFree(locations);
    cudaFree(distances);
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

int loadData(char *filename, std::vector<Record> &records, LatLong** loc, bool unified) {
  FILE *flist, *fp;
  int i = 0;
  char dbname[64];
  int recNum = 0;
  std::vector<LatLong> locations;

  /**Main processing **/

  TIMESTAMP(t0);
  flist = fopen(filename, "r");
  while (!feof(flist)) {
    /**
     * Read in all records of length REC_LENGTH
     * If this is the last file in the filelist, then done
     * else open next file to be read next iteration
     */
    if (fscanf(flist, "%s\n", dbname) != 1) {
      fprintf(stderr, "error reading filelist\n");
      exit(0);
    }
    fp = fopen(dbname, "r");
    if (!fp) {
      printf("error opening a db\n");
      exit(1);
    }
    // read each record
    while (!feof(fp)) {
      Record record;
      LatLong latLong;
      fgets(record.recString, 49, fp);
      fgetc(fp); // newline
      if (feof(fp))
        break;

      // parse for lat and long
      char substr[6];

      for (i = 0; i < 5; i++)
        substr[i] = *(record.recString + i + 28);
      substr[5] = '\0';
      latLong.lat = atof(substr);

      for (i = 0; i < 5; i++)
        substr[i] = *(record.recString + i + 33);
      substr[5] = '\0';
      latLong.lng = atof(substr);

      locations.push_back(latLong);
      records.push_back(record);
      recNum++;
    }
    fclose(fp);
  }
  fclose(flist);

  TIMESTAMP(t1);
  time_pre += ELAPSED(t0, t1);
  if (!unified) {
    *loc = (LatLong*) malloc(sizeof(LatLong) * locations.size());
  } else {
    assert(cudaMallocManaged(loc, sizeof(LatLong) * locations.size()) == cudaSuccess);
  }
  TIMESTAMP(t2);
  time_malloc += ELAPSED(t1, t2);
  for (int i = 0; i < locations.size(); i++) {
    (*loc)[i] = locations[i];
  }
  TIMESTAMP(t3);
  time_pre += ELAPSED(t2, t3);

  return recNum;
}

void findLowest(std::vector<Record> &records, float *distances, int numRecords, int topN) {
  int i, j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for (i = 0; i < topN; i++) {
    minLoc = i;
    for (j = i; j < numRecords; j++) {
      val = distances[j];
      if (val < distances[minLoc])
        minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename, int *r, float *lat, float *lng, int *q,
    int *t, int *p, int *d, bool* u) {
  int i;
  if (argc < 2)
    return 1; // error
  strncpy(filename, argv[1], 100);
  char flag;

  for (i = 1; i < argc; i++) {
    if (argv[i][0] == '-') { // flag
      flag = argv[i][1];
      switch (flag) {
        case 'r': // number of results
          i++;
          *r = atoi(argv[i]);
          break;
        case 'l': // lat or lng
          if (argv[i][2] == 'a') { //lat
            *lat = atof(argv[i + 1]);
          } else { //lng
            *lng = atof(argv[i + 1]);
          }
          i++;
          break;
        case 'h': // help
          return 1;
        case 'q': // quiet
          *q = 1;
          break;
        case 't': // timing
          *t = 1;
          break;
        case 'u': // unified memory
          *u = true;
          break;
        case 'p': // platform
          i++;
          *p = atoi(argv[i]);
          break;
        case 'd': // device
          i++;
          *d = atoi(argv[i]);
          break;
      }
    }
  }
  if ((*d >= 0 && *p < 0) || (*p >= 0 && *d < 0)) // both p and d must be specified if either are specified
    return 1;
  return 0;
}

void printUsage() {
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf(
      "nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
