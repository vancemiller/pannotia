/*
 * PAVLE - Parallel Variable-Length Encoder for CUDA
 *
 * Copyright (C) 2009 Tjark Bringewat <golvellius@gmx.net>, Ana Balevic <ana.balevic@gmail.com>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * MIT License. Read the full licence: http://www.opensource.org/licenses/mit-license.php
 *
 * If you find this program useful, please contact me and reference PAVLE home page in your work.
 *
 */


#ifndef _PACK_KERNELS_H_
#define _PACK_KERNELS_H_
#include <cstdint>
#include "parameters.h"

__global__ static void pack2(uint32_t* srcData, uint32_t* cindex, uint32_t* cindex2,
    uint32_t* dstData, uint32_t original_num_block_elements) {
  uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  // source index
  uint32_t offset = tid * original_num_block_elements;//DPB,
  uint32_t bitsize = cindex[tid];

  // destination index
  uint32_t pos = cindex2[tid];
  uint32_t dword = pos / 32;
  uint32_t bit = pos % 32;

  uint32_t dw = srcData[offset];      // load the first dword from srcData[]
  uint32_t tmp = dw >> bit;           // cut off those bits that do not fit into the initial location in destData[]
  atomicOr(&dstData[dword], tmp);	    // fill up this initial location
  tmp = dw << 32-bit;                 // save the remaining bits that were cut off earlier in tmp
  uint32_t i;
  for (i = 1; i < bitsize / 32; i++) {	// from now on, we have exclusive access to destData[]
    dw = srcData[offset+i];           // load next dword from srcData[]
    tmp |= dw >> bit;                 // fill up tmp
    dstData[dword+i] = tmp;           // write complete dword to destData[]
    tmp = dw << 32-bit;               // save the remaining bits in tmp (like before)
  }
  // exclusive access to dstData[] ends here
  // the remaining block can, or rather should be further optimized
  // write the remaining bits in tmp, UNLESS bit is 0 and bitsize is divisible by 32, in this case do nothing
  if (bit != 0 || bitsize % 32 != 0)
    atomicOr(&dstData[dword + i], tmp);
  if (bitsize % 32 != 0) {
    dw = srcData[offset+i];
    atomicOr(&dstData[dword+i], dw >> bit);
    atomicOr(&dstData[dword+i+1], dw << 32-bit);
  }
}

#endif
