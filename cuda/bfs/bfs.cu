/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad.
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for
  educational purpose is hereby granted without fee, provided that the above copyright
  notice and this permission notice appear in all copies of this software and that you do
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/

#include <argp.h>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda.h>
#include <errno.h>

#include "helper_cuda.h"

#define MAX_THREADS_PER_BLOCK 512

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

static char doc[] = "Rodinia Breadth-First Search Benchmark";
static char args_doc[] = "";

static struct argp_option options[] = {
  {"device", 'd', "DEVICE", 0, "CUDA Device ID"},
  {"file", 'f', "FILEPATH", 0, "Path to file containing input data."},
  {"unified", 'u', 0, 0, "Use unified memory"},
  {"verbose", 'v', 0, 0, "Verbose output"},
  {0},
};

struct arguments {
  uint8_t device;
  char* file;
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

//Structure to hold a node information
struct Node {
  int starting;
  int n_edges;
};

__global__ void Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask,
    bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int n_nodes);

__global__ void Kernel2(bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited,
    bool* g_over, int n_nodes);

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
  args.unified = false;
  args.verbose = false;
  // Parse command line arguments
  argp_parse(&argp, argc, argv, 0, 0, &args);
  if (!args.file) {
    fprintf(stderr, "Provide -f flag. Use -h for help\n");
    exit(EXIT_FAILURE);
  }

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

  VPRINT(args.verbose, "Reading File\n");
  //Read in Graph from a file
  FILE* fp = fopen(args.file, "r");
  if(!fp) {
    fprintf(stderr, "Error Reading graph file: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t n_nodes;

  int ret = fscanf(fp, "%lu", &n_nodes);
  if (!ret) {
    fprintf(stderr, "Invalid file format. %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  int n_blocks = 1;
  int n_threads_per_block = n_nodes;

  //Make execution Parameters according to the number of nodes
  //Distribute threads across multiple Blocks if necessary
  if (n_nodes > MAX_THREADS_PER_BLOCK) {
    n_blocks = (int) ceil(n_nodes / (double) MAX_THREADS_PER_BLOCK);
    n_threads_per_block = MAX_THREADS_PER_BLOCK;
  }

  TIMESTAMP(t1);
  time_pre += ELAPSED(t0, t1);
  // allocate host memory
  Node* h_graph_nodes;
  bool* h_graph_mask;
  bool* h_updating_graph_mask;
  bool* h_graph_visited;
  if (args.unified) {
    checkCudaErrors(cudaMallocManaged(&h_graph_nodes, sizeof(Node) * n_nodes));
    checkCudaErrors(cudaMallocManaged(&h_graph_mask, sizeof(bool) * n_nodes));
    checkCudaErrors(cudaMallocManaged(&h_updating_graph_mask, sizeof(bool) * n_nodes));
    checkCudaErrors(cudaMallocManaged(&h_graph_visited, sizeof(bool) * n_nodes));
  } else {
    h_graph_nodes = (Node*) malloc(sizeof(Node) * n_nodes);
    h_graph_mask = (bool*) malloc(sizeof(bool) * n_nodes);
    h_updating_graph_mask = (bool*) malloc(sizeof(bool) * n_nodes);
    h_graph_visited = (bool*) malloc(sizeof(bool) * n_nodes);
  }
  assert(h_graph_nodes && h_graph_mask && h_updating_graph_mask && h_graph_visited);

  TIMESTAMP(t2);
  time_malloc += ELAPSED(t1, t2);

  // initalize the memory
  for (unsigned int i = 0; i < n_nodes; i++) {
    int start;
    int edgeno;
    int ret = fscanf(fp, "%d %d", &start, &edgeno);
    if (!ret) {
      fprintf(stderr, "Invalid file format. %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }
    h_graph_nodes[i].starting = start;
    h_graph_nodes[i].n_edges = edgeno;
    h_graph_mask[i] = false;
    h_updating_graph_mask[i] = false;
    h_graph_visited[i] = false;
  }

  //read the source node from the file
  int source = 0;
  ret = fscanf(fp, "%d", &source);
  if (!ret) {
    fprintf(stderr, "Invalid file format. %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  // set the source node as true in the mask
  h_graph_mask[source] = true;
  h_graph_visited[source] = true;

  size_t edge_list_size;
  ret = fscanf(fp,"%lu", &edge_list_size);
  if (!ret) {
    fprintf(stderr, "Invalid file format. %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  TIMESTAMP(t3);
  time_pre += ELAPSED(t2, t3);

  int* h_graph_edges;
  if (args.unified) {
    checkCudaErrors(cudaMallocManaged(&h_graph_edges, sizeof(int) * edge_list_size));
  } else {
    h_graph_edges = (int*) malloc(sizeof(int) * edge_list_size);
  }
  assert(h_graph_edges);
  TIMESTAMP(t4);
  time_malloc += ELAPSED(t3, t4);

  for (int i = 0; i < edge_list_size; i++) {
    int id, cost;
    ret = fscanf(fp,"%d %d", &id, &cost);
    if (!ret) {
      fprintf(stderr, "Invalid file format. %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }
    h_graph_edges[i] = id;
  }

  TIMESTAMP(t5);
  time_pre += ELAPSED(t4, t5);

  // Copy the node list, edge list, masks, and visited nodes to device memory
  Node* d_graph_nodes;
  int* d_graph_edges;
  bool* d_graph_mask;
  bool* d_updating_graph_mask;
  bool* d_graph_visited;
  if (!args.unified) {
    checkCudaErrors(cudaMalloc(&d_graph_nodes, sizeof(Node) * n_nodes));
    checkCudaErrors(cudaMalloc(&d_graph_edges, sizeof(int) * edge_list_size));
    checkCudaErrors(cudaMalloc(&d_graph_mask, sizeof(bool) * n_nodes));
    checkCudaErrors(cudaMalloc(&d_updating_graph_mask, sizeof(bool) * n_nodes));
    checkCudaErrors(cudaMalloc(&d_graph_visited, sizeof(bool) * n_nodes));
  }

  TIMESTAMP(t6);
  time_malloc += ELAPSED(t5, t6);

  if (args.unified) {
    d_graph_nodes = h_graph_nodes;
    d_graph_edges = h_graph_edges;
    d_graph_mask = h_graph_mask;
    d_updating_graph_mask = h_updating_graph_mask;
    d_graph_visited = h_graph_visited;
  } else {
    checkCudaErrors(cudaMemcpyAsync(d_graph_nodes, h_graph_nodes, sizeof(Node) * n_nodes,
        cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size,
        cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_graph_mask, h_graph_mask, sizeof(bool) * n_nodes,
          cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool) *
          n_nodes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_graph_visited, h_graph_visited, sizeof(bool) * n_nodes,
          cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  TIMESTAMP(t7);
  time_copy_in += ELAPSED(t6, t7);

  // allocate mem for the result
  int* h_cost;
  int* d_cost;
  bool* h_done;
  bool* d_done;
  if (args.unified) {
    checkCudaErrors(cudaMallocManaged(&h_cost, sizeof(int) * n_nodes));
    checkCudaErrors(cudaMallocManaged(&h_done, sizeof(bool)));
    assert(h_cost && h_done);
  } else {
    h_cost = (int*) malloc(sizeof(int) * n_nodes);
    checkCudaErrors(cudaMalloc(&d_cost, sizeof(int) * n_nodes));
    h_done = (bool*) malloc(sizeof(bool));
    assert(h_cost && h_done);
    checkCudaErrors(cudaMalloc(&d_done, sizeof(bool)));
    assert(d_cost && d_done);
  }
  TIMESTAMP(t8);
  time_malloc += ELAPSED(t7, t8);

  if (args.unified) {
    memset(h_cost, -1, n_nodes);
    h_cost[source] = 0;
    d_cost = h_cost;
    d_done = h_done;
  } else {
    memset(h_cost, -1, n_nodes);
    h_cost[source] = 0;
    checkCudaErrors(cudaMemcpyAsync(d_cost, h_cost, sizeof(int) * n_nodes, cudaMemcpyHostToDevice,
        stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  TIMESTAMP(t9);
  time_copy_in += ELAPSED(t8, t9);

  // setup execution parameters
  const dim3 grid(n_blocks, 1, 1);
  const dim3 threads(n_threads_per_block, 1, 1);

  int k = 0;
  //Call the Kernel until all the elements of Frontier are not false
  do {
    *h_done = false;
    // if no thread changes this value then the loop stops
    if (!args.unified) {
      TIMESTAMP(t10);
      checkCudaErrors(cudaMemcpyAsync(d_done, h_done, sizeof(bool), cudaMemcpyHostToDevice,
          stream));
      checkCudaErrors(cudaStreamSynchronize(stream));
      TIMESTAMP(t11);
      time_copy_in += ELAPSED(t10, t11);
    }

    TIMESTAMP(t12);
    Kernel<<<grid, threads, 0, stream>>>(d_graph_nodes, d_graph_edges, d_graph_mask,
        d_updating_graph_mask, d_graph_visited, d_cost, n_nodes);
    checkCudaErrors(cudaStreamSynchronize(stream));
    Kernel2<<<grid, threads, 0, stream>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited,
        d_done, n_nodes);
    checkCudaErrors(cudaStreamSynchronize(stream));
    TIMESTAMP(t13);
    time_kernel += ELAPSED(t12, t13);

    if (!args.unified) {
      checkCudaErrors(cudaMemcpyAsync(h_done, d_done, sizeof(bool), cudaMemcpyDeviceToHost,
          stream));
      checkCudaErrors(cudaStreamSynchronize(stream));
      TIMESTAMP(t14);
      time_copy_out += ELAPSED(t13, t14);
    }
    k++;
  } while (*h_done);

  VPRINT(args.verbose, "Kernel Executed %d times\n", k);

  // copy result from device to host
  if (!args.unified) {
    TIMESTAMP(t15);
    checkCudaErrors(cudaMemcpyAsync(h_cost, d_cost, sizeof(int) * n_nodes, cudaMemcpyDeviceToHost,
        stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    TIMESTAMP(t16);
    time_copy_out += ELAPSED(t15, t16);
  }

  TIMESTAMP(t17);
  //Store the result into a file
  FILE* fpo = fopen("result.txt","w");
  if (!fpo) {
    fprintf(stderr, "Failed to open output. %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < n_nodes; i++) {
    fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
  }
  fclose(fpo);
  printf("Result stored in result.txt\n");

  TIMESTAMP(t18);
  time_post += ELAPSED(t17, t18);

  // cleanup memory
  if (args.unified) {
    checkCudaErrors(cudaFree(h_graph_nodes));
    checkCudaErrors(cudaFree(h_graph_edges));
    checkCudaErrors(cudaFree(h_graph_mask));
    checkCudaErrors(cudaFree(h_updating_graph_mask));
    checkCudaErrors(cudaFree(h_graph_visited));
    checkCudaErrors(cudaFree(h_cost));
    checkCudaErrors(cudaFree(h_done));
  } else {
    free(h_graph_nodes);
    free(h_graph_edges);
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
    free(h_done);
    checkCudaErrors(cudaFree(d_graph_nodes));
    checkCudaErrors(cudaFree(d_graph_edges));
    checkCudaErrors(cudaFree(d_graph_mask));
    checkCudaErrors(cudaFree(d_updating_graph_mask));
    checkCudaErrors(cudaFree(d_graph_visited));
    checkCudaErrors(cudaFree(d_cost));
    checkCudaErrors(cudaFree(d_done));
  }
  TIMESTAMP(t19);
  time_free += ELAPSED(t18, t19);

  printf("====Timing info====\n");
  printf("time malloc = %f ms\n", time_malloc);
  printf("time pre = %f ms\n", time_pre);
  printf("time copyIn = %f ms\n", time_copy_in);
  printf("time kernel = %f ms\n", time_kernel);
  printf("time serial = %f ms\n", time_serial);
  printf("time copyOut = %f ms\n", time_copy_out);
  printf("time post = %f ms\n", time_post);
  printf("time free = %f ms\n", time_free);
  printf("time end-to-end = %f ms\n", ELAPSED(t0, t19));
  exit(EXIT_SUCCESS);
}

__global__ void Kernel(Node* g_graph_nodes, int* g_graph_edges, bool* g_graph_mask,
    bool* g_updating_graph_mask, bool *g_graph_visited, int* g_cost, int n_nodes) {
	int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
	if (tid < n_nodes && g_graph_mask[tid]) {
		g_graph_mask[tid] = false;
		for (int i = g_graph_nodes[tid].starting; i < (g_graph_nodes[tid].n_edges +
        g_graph_nodes[tid].starting); i++) {
			int id = g_graph_edges[i];
			if (!g_graph_visited[id]) {
				g_cost[id]=g_cost[tid] + 1;
				g_updating_graph_mask[id] = true;
      }
    }
	}
}

__global__ void Kernel2(bool* g_graph_mask, bool *g_updating_graph_mask, bool* g_graph_visited,
    bool* g_over, int n_nodes) {
	int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
	if (tid < n_nodes && g_updating_graph_mask[tid]) {
		g_graph_mask[tid] = true;
		g_graph_visited[tid] = true;
		*g_over = true;
		g_updating_graph_mask[tid] = false;
	}
}

