#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda.h>

#define THREADS_PER_DIM 16
#define BLOCKS_PER_DIM 16
#define THREADS_PER_BLOCK THREADS_PER_DIM*THREADS_PER_DIM

extern "C" {
#include "kmeans.h"
}
#include "kmeans_cuda_kernel.cu"

#define CPU_DELTA_REDUCE
#define CPU_CENTER_REDUCE

#define TIMESTAMP(NAME) \
  struct timespec NAME; \
if (clock_gettime(CLOCK_MONOTONIC, &NAME)) { \
  fprintf(stderr, "Failed to get time: %s\n", strerror(errno)); \
}

#define ELAPSED(start, end) \
  ((long long int) 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)

unsigned int num_threads_perdim = THREADS_PER_DIM; /* sqrt(256) -- see references for this choice */
unsigned int num_blocks_perdim = BLOCKS_PER_DIM; /* temporary */
unsigned int num_threads = num_threads_perdim * num_threads_perdim; /* number of threads */
unsigned int num_blocks = num_blocks_perdim * num_blocks_perdim; /* number of blocks */

/* _d denotes it resides on the device */
int *membership_new; /* newly assignment membership */
float *feature_d; /* inverted data array */
float *feature_flipped_d; /* original (not inverted) data array */
int *membership_d; /* membership on the device */
float *block_new_centers; /* sum of points in a cluster (per block) */
float *clusters_d; /* cluster centers on the device */
float *block_clusters_d; /* per block calculation of cluster centers */
int *block_deltas_d; /* per block calculation of deltas */

/* -------------- allocateMemory() ------------------- */
/* allocate device memory, calculate number of blocks and threads, and invert the data array */
extern "C" void allocateMemory(int npoints, int nfeatures, int nclusters, float **features,
    bool unified) {
}
/* -------------- allocateMemory() end ------------------- */

/* -------------- deallocateMemory() ------------------- */
/* free host and device memory */
extern "C" void deallocateMemory(bool unified) {
  free(block_new_centers);
  cudaFree(feature_d);
  if (unified) {
    cudaFree(membership_new);
  } else {
    cudaFree(feature_flipped_d);
    free(membership_new);
    cudaFree(membership_d);
  }

  cudaFree(clusters_d);
#ifdef BLOCK_CENTER_REDUCE
  cudaFree(block_clusters_d);
#endif
#ifdef BLOCK_DELTA_REDUCE
  cudaFree(block_deltas_d);
#endif
}
/* -------------- deallocateMemory() end ------------------- */

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
  char *help = "\nUsage: %s [switches] -i filename\n\n"
    "    -i filename      :file containing data to be clustered\n"
    "    -m max_nclusters :maximum number of clusters allowed    [default=5]\n"
    "    -n min_nclusters :minimum number of clusters allowed    [default=5]\n"
    "    -t threshold     :threshold value                       [default=0.001]\n"
    "    -l nloops        :iteration for each number of clusters [default=1]\n"
    "    -b               :input file is in binary format\n"
    "    -r               :calculate RMSE                        [default=off]\n"
    "    -o               :output cluster center coordinates     [default=off]\n"
    "    -u               :use unified memory                    [default=off]\n";
  fprintf(stderr, help, argv0);
  exit(-1);
}

int main(int argc, char** argv) {
  int opt;
  extern char *optarg;
  char *filename = 0;
  float *buf;
  char line[1024];
  int isBinaryFile = 0;

  float threshold = 0.001; /* default value */
  int max_nclusters = 5; /* default value */
  int min_nclusters = 5; /* default value */
  int best_nclusters = 0;
  int nfeatures = 0;
  int npoints = 0;
  float len;

  float **features;
  float **cluster_centres = NULL;
  int i, j, index;
  int nloops = 1; /* default value */

  int isRMSE = 0;
  float rmse;

  int isOutput = 0;
  int unified = 0;
  //float	cluster_timing, io_timing;

  /* obtain command line arguments and change appropriate options */
  while ((opt = getopt(argc, argv, "i:t:m:n:l:brou")) != EOF) {
    switch (opt) {
      case 'i':
        filename = optarg;
        break;
      case 'b':
        isBinaryFile = 1;
        break;
      case 't':
        threshold = atof(optarg);
        break;
      case 'm':
        max_nclusters = atoi(optarg);
        break;
      case 'n':
        min_nclusters = atoi(optarg);
        break;
      case 'r':
        isRMSE = 1;
        break;
      case 'o':
        isOutput = 1;
        break;
      case 'l':
        nloops = atoi(optarg);
        break;
      case 'u':
        unified = 1;
        break;
      case '?':
        usage(argv[0]);
        break;
      default:
        usage(argv[0]);
        break;
    }
  }

  if (filename == 0)
    usage(argv[0]);

  long long time_pre = 0;
  long long time_post = 0;
  long long time_serial = 0;
  long long time_copy_in = 0;
  long long time_copy_out = 0;
  long long time_kernel = 0;
  long long time_malloc = 0;
  long long time_free = 0;

  TIMESTAMP(t0);
  /* ============== I/O begin ==============*/
  /* get nfeatures and npoints */
  if (isBinaryFile) {		//Binary file input
    int infile;
    if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    read(infile, &npoints, sizeof(int));
    read(infile, &nfeatures, sizeof(int));

    TIMESTAMP(t1);
    time_pre += ELAPSED(t0, t1);

    /* allocate space for features[][] and read attributes of all objects */
    buf = (float*) malloc(npoints * nfeatures * sizeof(float));
    features = (float**) malloc(npoints * sizeof(float*));
    if (unified) {
      cudaMallocManaged(&features[0], npoints * nfeatures * sizeof(float));
    } else {
      features[0] = (float*) malloc(npoints * nfeatures * sizeof(float));
    }
    TIMESTAMP(t2);
    time_malloc += ELAPSED(t1, t2);

    for (i = 1; i < npoints; i++)
      features[i] = features[i - 1] + nfeatures;

    read(infile, buf, npoints * nfeatures * sizeof(float));
    close(infile);
    TIMESTAMP(t3);
    time_pre += ELAPSED(t2, t3);
  } else {
    FILE *infile;
    if ((infile = fopen(filename, "r")) == NULL) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
      if (strtok(line, " \t\n") != 0)
        npoints++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") != 0) {
        /* ignore the id (first attribute): nfeatures = 1; */
        while (strtok(NULL, " ,\t\n") != NULL)
          nfeatures++;
        break;
      }
    }
    TIMESTAMP(t1);
    time_pre += ELAPSED(t0, t1);

    /* allocate space for features[] and read attributes of all objects */
    buf = (float*) malloc(npoints * nfeatures * sizeof(float));
    features = (float**) malloc(npoints * sizeof(float*));
    if (unified) {
      cudaMallocManaged(&features[0], npoints * nfeatures * sizeof(float));
    } else {
      features[0] = (float*) malloc(npoints * nfeatures * sizeof(float));
    }
    TIMESTAMP(t2);
    time_malloc += ELAPSED(t1, t2);

    for (i = 1; i < npoints; i++)
      features[i] = features[i - 1] + nfeatures;
    rewind(infile);
    i = 0;
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") == NULL)
        continue;
      for (j = 0; j < nfeatures; j++) {
        buf[i] = atof(strtok(NULL, " ,\t\n"));
        i++;
      }
    }
    fclose(infile);
    TIMESTAMP(t3);
    time_pre += ELAPSED(t2, t3);
  }
  TIMESTAMP(t1);

  printf("\nI/O completed\n");
  printf("\nNumber of objects: %d\n", npoints);
  printf("Number of features: %d\n", nfeatures);
  /* ============== I/O end ==============*/

  // error check for clusters
  if (npoints < min_nclusters) {
    printf("Error: min_nclusters(%d) > npoints(%d) -- cannot proceed\n", min_nclusters, npoints);
    exit(0);
  }

  srand(7); /* seed for future random number generator */
  memcpy(features[0], buf, npoints * nfeatures * sizeof(float)); /* now features holds 2-dimensional array of features */
  free(buf);

  /* ======================= core of the clustering ===================*/

  cluster_centres = NULL;

  index = 0; /* number of iteration to reach the best RMSE */
  int nclusters; /* number of clusters k */
  int *membership; /* which cluster a data point belongs to */
  float **tmp_cluster_centres; /* hold coordinates of cluster centers */
  float min_rmse_ref = FLT_MAX; /* reference min_rmse value */
  float min_rmse = 0;

  TIMESTAMP(t2);
  time_pre += ELAPSED(t1, t2);

  /* allocate memory for membership */
  membership = (int*) malloc(npoints * sizeof(int));
  TIMESTAMP(t3);
  time_malloc += ELAPSED(t2, t3);

  /* sweep k from min to max_nclusters to find the best number of clusters */
  for (nclusters = min_nclusters; nclusters <= max_nclusters; nclusters++) {
    TIMESTAMP(t4);
    if (nclusters > npoints)
      break; /* cannot have more clusters than points */

    /* allocate device memory, invert data array (@ kmeans_cuda.cu) */
    num_blocks = npoints / num_threads;
    if (npoints % num_threads > 0) /* defeat truncation */
      num_blocks++;

    num_blocks_perdim = sqrt((double) num_blocks);
    while (num_blocks_perdim * num_blocks_perdim < num_blocks)  // defeat truncation (should run once)
      num_blocks_perdim++;

    num_blocks = num_blocks_perdim * num_blocks_perdim;

    TIMESTAMP(t5);
    time_pre += ELAPSED(t4, t5);

    /* allocate memory for block_new_centers[] (host) */
    block_new_centers = (float *) malloc(nclusters * nfeatures * sizeof(float));
    /* allocate memory for memory_new[] and initialize to -1 (host) */
    if (unified) {
      cudaMallocManaged(&membership_new, npoints * sizeof(int));
    } else {
      membership_new = (int*) malloc(npoints * sizeof(int));
      /* allocate memory for feature_flipped_d[][], feature_d[][] (device) */
      cudaMalloc((void**) &feature_flipped_d, npoints * nfeatures * sizeof(float));
      /* allocate memory for membership_d[] and clusters_d[][] (device) */
      cudaMalloc((void**) &membership_d, npoints * sizeof(int));
    }
    cudaMalloc((void**) &clusters_d, nclusters * nfeatures * sizeof(float));
    cudaMalloc((void**) &feature_d, npoints * nfeatures * sizeof(float));
    TIMESTAMP(t6);
    time_malloc += ELAPSED(t5, t6);

    for (i = 0; i < npoints; i++) {
      membership_new[i] = -1;
    }
    TIMESTAMP(t7);
    time_pre += ELAPSED(t6, t7);

    if (unified) {
      feature_flipped_d = features[0];
    } else {
      cudaMemcpy(feature_flipped_d, features[0], npoints * nfeatures * sizeof(float),
          cudaMemcpyHostToDevice);
    }
    TIMESTAMP(t8);
    time_copy_in += ELAPSED(t7, t8);

    /* invert the data array (kernel execution) */
    invert_mapping<<<num_blocks,num_threads>>>(feature_flipped_d,feature_d,npoints,nfeatures);

    TIMESTAMP(t9);
    time_kernel += ELAPSED(t8, t9);

#ifdef BLOCK_DELTA_REDUCE
    // allocate array to hold the per block deltas on the gpu side

    if (unified) {
      cudaMallocManaged(&block_deltas_d, num_block_perdim * num_blocks_perdim * sizeof(int));
    }
    cudaMalloc((void**) &block_deltas_d, num_blocks_perdim * num_blocks_perdim * sizeof(int));
    TIMESTAMP(t10);
    time_malloc += ELAPSED(t9, t10);
#endif

#ifdef BLOCK_CENTER_REDUCE
    // allocate memory and copy to card cluster  array in which to accumulate center points for the next iteration
    TIMESTAMP(t11);
    if (unified) {
      cudaMallocManaged(&block_clusters_d, num_blocks_perdim * num_blocks_perdim * nclusters *
          nfeatures * sizeof(float));
    } else {
      cudaMalloc((void**) &block_clusters_d,
          num_blocks_perdim * num_blocks_perdim *
          nclusters * nfeatures * sizeof(float));
    }
    TIMESTAMP(t12);
    time_malloc += ELAPSED(t11, t12);
    //cudaMemcpy(new_clusters_d, new_centers[0], nclusters*nfeatures*sizeof(float), cudaMemcpyHostToDevice);
#endif

    /* iterate nloops times for each number of clusters */
    for (i = 0; i < nloops; i++) {
      /* initialize initial cluster centers, CUDA calls (@ kmeans_cuda.cu) */
      TIMESTAMP(t13);
      tmp_cluster_centres = kmeans_clustering(features, nfeatures, npoints, nclusters, threshold,
          membership, unified);
      TIMESTAMP(t14);
      time_serial += ELAPSED(t13, t14);
      if (cluster_centres) {
        free((cluster_centres)[0]);
        free(cluster_centres);
      }
      TIMESTAMP(t15);
      time_free += ELAPSED(t14, t15);

      cluster_centres = tmp_cluster_centres;

      /* find the number of clusters with the best RMSE */
      if (isRMSE) {
        rmse = rms_err(features, nfeatures, npoints, tmp_cluster_centres, nclusters);

        if (rmse < min_rmse_ref) {
          min_rmse_ref = rmse;			//update reference min RMSE
          min_rmse = min_rmse_ref;		//update return min RMSE
          best_nclusters = nclusters;	//update optimum number of clusters
          index = i;						//update number of iteration to reach best RMSE
        }
      }
      TIMESTAMP(t16);
      time_serial += ELAPSED(t15, t16);
    }

    TIMESTAMP(t17);
    deallocateMemory(unified); /* free device memory (@ kmeans_cuda.cu) */
    TIMESTAMP(t18);
    time_free += ELAPSED(t17, t18);
  }

  TIMESTAMP(t19);
  free(membership);
  TIMESTAMP(t20);
  time_free += ELAPSED(t19, t20);

  /* =============== Command Line Output =============== */

  /* cluster center coordinates
     :displayed only for when k=1*/
  if ((min_nclusters == max_nclusters) && (isOutput == 1)) {
    printf("\n================= Centroid Coordinates =================\n");
    for (i = 0; i < max_nclusters; i++) {
      printf("%d:", i);
      for (j = 0; j < nfeatures; j++) {
        printf(" %.2f", cluster_centres[i][j]);
      }
      printf("\n\n");
    }
  }

  len = (float) ((max_nclusters - min_nclusters + 1) * nloops);

  printf("Number of Iteration: %d\n", nloops);

  if (min_nclusters != max_nclusters) {
    if (nloops != 1) {									//range of k, multiple iteration
      printf("Best number of clusters is %d\n", best_nclusters);
    } else {												//range of k, single iteration
      printf("Best number of clusters is %d\n", best_nclusters);
    }
  } else {
    if (nloops != 1) {									// single k, multiple iteration
      if (isRMSE)										// if calculated RMSE
        printf("Number of trials to approach the best RMSE of %.3f is %d\n", rmse, index + 1);
    } else {												// single k, single iteration
      if (isRMSE)										// if calculated RMSE
        printf("Root Mean Squared Error: %.3f\n", rmse);
    }
  }

  TIMESTAMP(t21);
  time_post += ELAPSED(t20, t21);
  /* free up memory */
  if (unified) {
    cudaFree(features[0]);
  } else {
    free(features[0]);
  }
  free(features);
  TIMESTAMP(t22);
  time_free += ELAPSED(t21, t22);

  printf("====Timing info====\n");
  printf("time malloc = %f ms\n", time_malloc * 1e-6);
  printf("time pre = %f ms\n", time_pre * 1e-6);
  printf("time CPU to GPU memory copy = %f ms\n", time_copy_in * 1e-6);
  printf("time kernel = %f ms\n", time_kernel * 1e-6);
  printf("time serial = %f ms\n", time_serial * 1e-6);
  printf("time GPU to CPU memory copy back = %f ms\n", time_copy_out * 1e-6);
  printf("time post = %f ms\n", time_post * 1e-6);
  printf("time free = %f ms\n", time_free * 1e-6);
  printf("End-to-end = %f ms\n", ELAPSED(t0, t22) * 1e-6);
  exit(EXIT_SUCCESS);
}

/* ------------------- kmeansCuda() ------------------------ */
extern "C" int // delta -- had problems when return value was of float type
kmeansCuda(float **feature, /* in: [npoints][nfeatures] */
    int nfeatures, /* number of attributes for each point */
    int npoints, /* number of data points */
    int nclusters, /* number of clusters */
    int *membership, /* which cluster the point belongs to */
    float **clusters, /* coordinates of cluster centers */
    int *new_centers_len, /* number of elements in each cluster */
    float **new_centers, /* sum of elements in each cluster */
    int unified) {
  int delta = 0; /* if point has moved */
  int i, j; /* counters */

  if (unified) {
    membership_d = membership_new;
  } else {
    /* copy membership (host to device) */
    cudaMemcpy(membership_d, membership_new, npoints * sizeof(int), cudaMemcpyHostToDevice);
    /* copy clusters (host to device) */
  }
  cudaMemcpy(clusters_d, clusters[0], nclusters * nfeatures * sizeof(float),
      cudaMemcpyHostToDevice);

  /* set up texture */
  cudaChannelFormatDesc chDesc0 = cudaCreateChannelDesc<float>();
  t_features.filterMode = cudaFilterModePoint;
  t_features.normalized = false;
  t_features.channelDesc = chDesc0;

  if (cudaBindTexture(NULL, &t_features, feature_d, &chDesc0, npoints * nfeatures * sizeof(float))
      != CUDA_SUCCESS)
    printf("Couldn't bind features array to texture!\n");

  cudaChannelFormatDesc chDesc1 = cudaCreateChannelDesc<float>();
  t_features_flipped.filterMode = cudaFilterModePoint;
  t_features_flipped.normalized = false;
  t_features_flipped.channelDesc = chDesc1;

  if (cudaBindTexture(NULL, &t_features_flipped, feature_flipped_d, &chDesc1,
        npoints * nfeatures * sizeof(float)) != CUDA_SUCCESS)
    printf("Couldn't bind features_flipped array to texture!\n");

  cudaChannelFormatDesc chDesc2 = cudaCreateChannelDesc<float>();
  t_clusters.filterMode = cudaFilterModePoint;
  t_clusters.normalized = false;
  t_clusters.channelDesc = chDesc2;

  if (cudaBindTexture(NULL, &t_clusters, clusters_d, &chDesc2,
        nclusters * nfeatures * sizeof(float)) != CUDA_SUCCESS)
    printf("Couldn't bind clusters array to texture!\n");

  /* copy clusters to constant memory */
  cudaMemcpyToSymbol("c_clusters", clusters[0], nclusters * nfeatures * sizeof(float), 0,
      cudaMemcpyHostToDevice);

  /* setup execution parameters.
     changed to 2d (source code on NVIDIA CUDA Programming Guide) */
  dim3 grid(num_blocks_perdim, num_blocks_perdim);
  dim3 threads(num_threads_perdim * num_threads_perdim);

  /* execute the kernel */
  kmeansPoint<<< grid, threads >>>( feature_d,
      nfeatures,
      npoints,
      nclusters,
      membership_d,
      clusters_d,
      block_clusters_d,
      block_deltas_d);

  cudaThreadSynchronize();

  /* copy back membership (device to host) */
  if (!unified)
    cudaMemcpy(membership_new, membership_d, npoints * sizeof(int), cudaMemcpyDeviceToHost);

#ifdef BLOCK_CENTER_REDUCE
  /*** Copy back arrays of per block sums ***/
  float * block_clusters_h;
  if (unified) {
    block_clusters_h = block_clusters_d;
  } else {
    block_clusters_h = (float *) malloc(
        num_blocks_perdim * num_blocks_perdim *
        nclusters * nfeatures * sizeof(float));
    cudaMemcpy(block_clusters_h, block_clusters_d,
        num_blocks_perdim * num_blocks_perdim *
        nclusters * nfeatures * sizeof(float),
        cudaMemcpyDeviceToHost);
  }

#endif
#ifdef BLOCK_DELTA_REDUCE
  int* block_deltas_h;
  if (unified) {
    block_deltas_h = block_deltas_d;
  } else {
    block_deltas_h = (int *) malloc(
        num_blocks_perdim * num_blocks_perdim * sizeof(int));

    cudaMemcpy(block_deltas_h, block_deltas_d,
        num_blocks_perdim * num_blocks_perdim * sizeof(int),
        cudaMemcpyDeviceToHost);
  }
#endif

  /* for each point, sum data points in each cluster
     and see if membership has changed:
     if so, increase delta and change old membership, and update new_centers;
     otherwise, update new_centers */
  delta = 0;
  for (i = 0; i < npoints; i++) {
    int cluster_id = membership_new[i];
    new_centers_len[cluster_id]++;
    if (membership_new[i] != membership[i]) {
#ifdef CPU_DELTA_REDUCE
      delta++;
#endif
      membership[i] = membership_new[i];
    }
#ifdef CPU_CENTER_REDUCE
    for (j = 0; j < nfeatures; j++) {
      new_centers[cluster_id][j] += feature[i][j];
    }
#endif
  }

#ifdef BLOCK_DELTA_REDUCE
  /*** calculate global sums from per block sums for delta and the new centers ***/

  //debug
  //printf("\t \t reducing %d block sums to global sum \n",num_blocks_perdim * num_blocks_perdim);
  for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
    //printf("block %d delta is %d \n",i,block_deltas_h[i]);
    delta += block_deltas_h[i];
  }

#endif
#ifdef BLOCK_CENTER_REDUCE

  for(int j = 0; j < nclusters;j++) {
    for(int k = 0; k < nfeatures;k++) {
      block_new_centers[j*nfeatures + k] = 0.f;
    }
  }

  for(i = 0; i < num_blocks_perdim * num_blocks_perdim; i++) {
    for(int j = 0; j < nclusters;j++) {
      for(int k = 0; k < nfeatures;k++) {
        block_new_centers[j*nfeatures + k] += block_clusters_h[i * nclusters*nfeatures + j * nfeatures + k];
      }
    }
  }

#ifdef CPU_CENTER_REDUCE
  //debug
  /*for(int j = 0; j < nclusters;j++) {
    for(int k = 0; k < nfeatures;k++) {
    if(new_centers[j][k] >  1.001 * block_new_centers[j*nfeatures + k] || new_centers[j][k] < 0.999 * block_new_centers[j*nfeatures + k]) {
    printf("\t \t for %d:%d, normal value is %e and gpu reduced value id %e \n",j,k,new_centers[j][k],block_new_centers[j*nfeatures + k]);
    }
    }
    }*/
#endif

#ifdef BLOCK_CENTER_REDUCE
  for(int j = 0; j < nclusters;j++) {
    for(int k = 0; k < nfeatures;k++)
      new_centers[j][k]= block_new_centers[j*nfeatures + k];
  }
#endif

#endif

  return delta;

}
/* ------------------- kmeansCuda() end ------------------------ */

