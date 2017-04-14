/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA. Main file.
 *
 * Copyright (C) 2009 Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 *
 */

#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "print_helpers.h"
#include "comparison_helpers.h"
#include "stats_logger.h"
#include "load_data.h"
#include <sys/time.h>
#include "vlc_kernel_sm64huff.cu"
#include "scan.cu"
#include "pack_kernels.cu"
#include "cpuencode.h"

long long get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000000) + tv.tv_usec;
}
void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks=1, bool unified=false);

extern "C" void cpu_vlc_encode(unsigned int* indata, unsigned int num_elements, unsigned int* outdata, unsigned int *outsize, unsigned int *codewords, unsigned int* codewordlens);

int main(int argc, char* argv[]){
  unsigned int num_block_threads = 256;
  if (argc > 2) {
    bool unified = argv[1];
    for (int i=2; i<argc; i++) {
      runVLCTest(argv[i], num_block_threads, unified);
    }
  } else {
    bool unified = false;
    if (argc == 2) {
      unified = argv[1];
    }
    runVLCTest(NULL, num_block_threads, 1024, unified);
  }
  checkCudaErrors(cudaThreadExit());
  return 0;
}

void runVLCTest(char *file_name, uint num_block_threads, uint num_blocks, bool unified) {
  printf("CUDA! Starting VLC Tests!\n");
  unsigned int num_elements; //uint num_elements = num_blocks * num_block_threads;
  unsigned int mem_size; //uint mem_size = num_elements * sizeof(int);
  unsigned int symbol_type_size = sizeof(int);
  //////// LOAD DATA ///////////////
  double H; // entropy
  initParams(file_name, num_block_threads, num_blocks, num_elements, mem_size, symbol_type_size);
  printf("Parameters: num_elements: %d, num_blocks: %d, num_block_threads: %d\n----------------------------\n", num_elements, num_blocks, num_block_threads);
  ////////LOAD DATA ///////////////
  uint	*sourceData;
  uint	*destData;
  uint	*crefData;
  crefData=	(uint*) malloc(mem_size);
  uint	*codewords;
  uint	*codewordlens;
  if (unified) {
    checkCudaErrors(cudaMallocManaged(&sourceData, mem_size));
    checkCudaErrors(cudaMallocManaged(&destData, mem_size));
    checkCudaErrors(cudaMallocManaged(&codewords, NUM_SYMBOLS * symbol_type_size));
    checkCudaErrors(cudaMallocManaged(&codewordlens, NUM_SYMBOLS * symbol_type_size));
  } else {
    sourceData =	(uint*) malloc(mem_size);
    destData =	(uint*) malloc(mem_size);
    codewords = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
    codewordlens = (uint*) malloc(NUM_SYMBOLS*symbol_type_size);
  }

  uint	*cw32 =		(uint*) malloc(mem_size);
  uint	*cw32len =	(uint*) malloc(mem_size);
  uint	*cw32idx =	(uint*) malloc(mem_size);

  uint	*cindex2=	(uint*) malloc(num_blocks*sizeof(int));

  memset(sourceData,   0, mem_size);
  memset(destData,     0, mem_size);
  memset(crefData,     0, mem_size);
  memset(cw32,         0, mem_size);
  memset(cw32len,      0, mem_size);
  memset(cw32idx,      0, mem_size);
  memset(codewords,    0, NUM_SYMBOLS*symbol_type_size);
  memset(codewordlens, 0, NUM_SYMBOLS*symbol_type_size);
  memset(cindex2, 0, num_blocks*sizeof(int));
  //////// LOAD DATA ///////////////
  loadData(file_name, sourceData, codewords, codewordlens, num_elements, mem_size, H);

  //////// LOAD DATA ///////////////

  unsigned int	*d_sourceData, *d_destData, *d_destDataPacked;
  unsigned int	*d_codewords, *d_codewordlens;
  unsigned int	*d_cw32, *d_cw32len, *d_cw32idx, *d_cindex, *d_cindex2;

  if (unified) {
    checkCudaErrors(cudaMallocManaged((void**) &d_destDataPacked,	  mem_size));
  } else {
    checkCudaErrors(cudaMalloc((void**) &d_sourceData,		  mem_size));
    checkCudaErrors(cudaMalloc((void**) &d_destData,			  mem_size));
    checkCudaErrors(cudaMalloc((void**) &d_destDataPacked,	  mem_size));

    checkCudaErrors(cudaMalloc((void**) &d_codewords,		  NUM_SYMBOLS*symbol_type_size));
    checkCudaErrors(cudaMalloc((void**) &d_codewordlens,		  NUM_SYMBOLS*symbol_type_size));
  }

  checkCudaErrors(cudaMalloc((void**) &d_cw32,				  mem_size));
  checkCudaErrors(cudaMalloc((void**) &d_cw32len,			  mem_size));
  checkCudaErrors(cudaMalloc((void**) &d_cw32idx,			  mem_size));

  checkCudaErrors(cudaMalloc((void**)&d_cindex,         num_blocks*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void**)&d_cindex2,        num_blocks*sizeof(unsigned int)));

  if (unified) {
    d_sourceData = sourceData;
    d_codewords = codewords;
    d_codewordlens = codewordlens;
    d_destData = destData;
  } else {
    checkCudaErrors(cudaMemcpy(d_sourceData,		sourceData,		mem_size,		cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_codewords,		codewords,		NUM_SYMBOLS*symbol_type_size,	cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_codewordlens,	codewordlens,	NUM_SYMBOLS*symbol_type_size,	cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_destData,		destData,		mem_size,		cudaMemcpyHostToDevice));
  }

  dim3 grid_size(num_blocks,1,1);
  dim3 block_size(num_block_threads, 1, 1);
  unsigned int sm_size;


  unsigned int NT = 10; //number of runs for each execution time

  //////////////////* CPU ENCODER *///////////////////////////////////
  unsigned int refbytesize;
  long long timer = get_time();
  cpu_vlc_encode((unsigned int*)sourceData, num_elements, (unsigned int*)crefData,  &refbytesize, codewords, codewordlens);
  float msec = (float)((get_time() - timer)/1000.0);
  printf("CPU Encoding time (CPU): %f (ms)\n", msec);
  printf("CPU Encoded to %d [B]\n", refbytesize);
  unsigned int num_ints = refbytesize/4 + ((refbytesize%4 ==0)?0:1);
  //////////////////* END CPU *///////////////////////////////////

  //////////////////* SM64HUFF KERNEL *///////////////////////////////////
  grid_size.x		= num_blocks;
  block_size.x	= num_block_threads;
  sm_size			= block_size.x*sizeof(unsigned int);
#ifdef CACHECWLUT
  sm_size			= 2*NUM_SYMBOLS*sizeof(int) + block_size.x*sizeof(unsigned int);
#endif
  cudaEvent_t     start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord( start, 0 );
  for (int i=0; i<NT; i++) {
    vlc_encode_kernel_sm64huff<<<grid_size, block_size, sm_size>>>(d_sourceData, d_codewords, d_codewordlens,
#ifdef TESTING
        d_cw32, d_cw32len, d_cw32idx,
#endif
        d_destData, d_cindex); //testedOK2
  }
  cudaThreadSynchronize();
  cudaEventRecord( stop, 0 ) ;
  cudaEventSynchronize( stop ) ;
  float   elapsedTime;
  cudaEventElapsedTime( &elapsedTime,
      start, stop ) ;

  printf("GPU Encoding time (SM64HUFF): %f (ms)\n", elapsedTime/NT);
  //////////////////* END KERNEL *///////////////////////////////////

#ifdef TESTING
  unsigned int num_scan_elements = grid_size.x;
  preallocBlockSums(num_scan_elements);
  cudaMemset(d_destDataPacked, 0, mem_size);
  printf("Num_blocks to be passed to scan is %d.\n", num_scan_elements);
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  prescanArray(d_cindex2, d_cindex, num_scan_elements, stream);

  pack2<<< num_scan_elements/16, 16, 0, stream>>>((unsigned int*)d_destData, d_cindex, d_cindex2, (unsigned int*)d_destDataPacked, num_elements/num_scan_elements);
  checkCudaErrors(cudaStreamSynchronize(stream));
  deallocBlockSums();

  checkCudaErrors(cudaMemcpy(destData, d_destDataPacked, mem_size, cudaMemcpyDeviceToHost));
  compare_vectors((unsigned int*)crefData, (unsigned int*)destData, num_ints);
#endif

  if (unified) {
    checkCudaErrors(cudaFree(sourceData));
    checkCudaErrors(cudaFree(destData));
    checkCudaErrors(cudaFree(codewords));
    checkCudaErrors(cudaFree(codewordlens));
  } else {
    free(sourceData); free(destData);  	free(codewords);  	free(codewordlens); free(cw32);  free(cw32len); free(crefData);
    checkCudaErrors(cudaFree(d_sourceData)); 	checkCudaErrors(cudaFree(d_destData)); checkCudaErrors(cudaFree(d_destDataPacked));
    checkCudaErrors(cudaFree(d_codewords)); 		checkCudaErrors(cudaFree(d_codewordlens));
  }
  checkCudaErrors(cudaFree(d_cw32)); 		checkCudaErrors(cudaFree(d_cw32len)); 	checkCudaErrors(cudaFree(d_cw32idx));
  checkCudaErrors(cudaFree(d_cindex)); checkCudaErrors(cudaFree(d_cindex2));
  free(cindex2);
}

