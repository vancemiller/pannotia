/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

// includes, kernels
#include <cassert>
#include <cmath>
#include <cstdio>

#include "helper_cuda.h"
#include "scanLargeArray_kernel.cu"

inline bool isPowerOfTwo(int n) {
  return ((n & (n - 1)) == 0);
}

inline int floorPow2(int n) {
  int exp;
  frexp((float) n, &exp);
  return 1 << (exp - 1);
}

#define BLOCK_SIZE 256

static uint32_t** g_scanBlockSums;
static uint32_t g_numEltsAllocated = 0;
static uint32_t g_numLevelsAllocated = 0;

static void preallocBlockSums(uint32_t maxNumElements) {
  assert(!g_numEltsAllocated); // shouldn't be called

  g_numEltsAllocated = maxNumElements;

  uint32_t blockSize = BLOCK_SIZE; // max size of the thread blocks
  uint32_t numElts = maxNumElements;
  int level = 0;

  do {
    uint32_t numBlocks = max(1, (int) ceil((float) numElts / (2.f * blockSize)));
    if (numBlocks > 1)
      level++;
    numElts = numBlocks;
  } while (numElts > 1);

  g_scanBlockSums = (uint32_t**) malloc(level * sizeof(uint32_t*));
  g_numLevelsAllocated = level;
  numElts = maxNumElements;
  level = 0;

  do {
    uint32_t numBlocks = max(1, (int) ceil((float) numElts / (2.f * blockSize)));
    if (numBlocks > 1)
      checkCudaErrors(cudaMalloc((void**) &g_scanBlockSums[level++], numBlocks * sizeof(uint32_t)));
    numElts = numBlocks;
  } while (numElts > 1);
}

static void deallocBlockSums() {
  for (uint32_t i = 0; i < g_numLevelsAllocated; i++) {
    checkCudaErrors(cudaFree(g_scanBlockSums[i]));
  }

  free((void**) g_scanBlockSums);

  g_scanBlockSums = 0;
  g_numEltsAllocated = 0;
  g_numLevelsAllocated = 0;
}

static void prescanArrayRecursive(uint32_t *outArray, const uint32_t *inArray,
    int numElements, int level, cudaStream_t stream) {
  const uint32_t blockSize = BLOCK_SIZE; // max size of the thread blocks
  const uint32_t numBlocks = max(1, (int) ceil((float) numElements / (2.f * blockSize)));
  uint32_t numThreads;

  if (numBlocks > 1)
    numThreads = blockSize;
  else if (isPowerOfTwo(numElements))
    numThreads = numElements / 2;
  else
    numThreads = floorPow2(numElements);

  uint32_t numEltsPerBlock = numThreads * 2;

  // if this is a non-power-of-2 array, the last block will be non-full
  // compute the smallest power of 2 able to compute its scan.
  uint32_t numEltsLastBlock = numElements - (numBlocks - 1) * numEltsPerBlock;
  uint32_t numThreadsLastBlock = max(1, numEltsLastBlock / 2);
  uint32_t np2LastBlock = 0;
  uint32_t sharedMemLastBlock = 0;

  if (numEltsLastBlock != numEltsPerBlock) {
    np2LastBlock = 1;

    if (!isPowerOfTwo(numEltsLastBlock))
      numThreadsLastBlock = floorPow2(numEltsLastBlock);

    uint32_t extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
    sharedMemLastBlock = sizeof(uint32_t) * (2 * numThreadsLastBlock + extraSpace);
  }

  // padding space is used to avoid shared memory bank conflicts
  uint32_t extraSpace = numEltsPerBlock / NUM_BANKS;
  uint32_t sharedMemSize = sizeof(uint32_t) * (numEltsPerBlock + extraSpace);

  // setup execution parameters
  // if NP2, we process the last block separately
  dim3 grid(max(1, numBlocks - np2LastBlock), 1, 1);
  dim3 threads(numThreads, 1, 1);

  // execute the scan
  if (numBlocks > 1) {
    prescan<true, false><<<grid, threads, sharedMemSize, stream>>>(outArray, inArray,
        g_scanBlockSums[level], numThreads * 2, 0, 0);
    checkCudaErrors(cudaStreamSynchronize(stream));
    if (np2LastBlock) {
      prescan<true, true><<<1, numThreadsLastBlock, sharedMemLastBlock, stream>>>(outArray,
          inArray, g_scanBlockSums[level], numEltsLastBlock, numBlocks - 1,
          numElements - numEltsLastBlock);
      checkCudaErrors(cudaStreamSynchronize(stream));
    }

    // After scanning all the sub-blocks, we are mostly done.  But now we
    // need to take all of the last values of the sub-blocks and scan those.
    // This will give us a new value that must be added to each block to
    // get the final results.
    // recursive (CPU) call
    prescanArrayRecursive(g_scanBlockSums[level], g_scanBlockSums[level], numBlocks, level + 1,
        stream);

    uniformAdd<<<grid, threads, 0, stream>>>(outArray, g_scanBlockSums[level],
        numElements - numEltsLastBlock, 0, 0);
    checkCudaErrors(cudaStreamSynchronize(stream));

    if (np2LastBlock) {
      uniformAdd<<<1, numThreadsLastBlock, 0, stream>>>(outArray, g_scanBlockSums[level],
          numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
      checkCudaErrors(cudaStreamSynchronize(stream));
    }
  } else if (isPowerOfTwo(numElements)) {
    prescan<false, false><<<grid, threads, sharedMemSize, stream>>>(outArray, inArray, 0,
        numThreads * 2, 0, 0);
    checkCudaErrors(cudaStreamSynchronize(stream));
  } else {
    prescan<false, true><<<grid, threads, sharedMemSize, stream>>>(outArray, inArray, 0,
        numElements, 0, 0);
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
}

static void prescanArray(uint32_t* outArray, uint32_t* inArray, int numElements,
    cudaStream_t stream) {
  prescanArrayRecursive(outArray, inArray, numElements, 0, stream);
}

