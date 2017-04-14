#include <stdint.h>
#include <stdlib.h>
#include <cmath>

#include "print_helpers.h"
#include "cpuencode.h"

using namespace std;

// The max. codeword length for each byte symbol is 32-bits

extern "C"
void cpu_vlc_encode(uint32_t* indata, uint32_t num_elements, uint32_t* outdata,
    uint32_t* outsize, uint32_t* codewords, uint32_t* codewordlens) {
  uint32_t* currentBytes = (uint32_t*) outdata;
  *currentBytes = 0x00000000U;
  uint32_t startbit = 0;
  uint32_t totalBytes = 0;

  for (uint32_t k = 0; k < num_elements; k++) {
    uint32_t cw = 0;
    uint32_t val = indata[k];
    uint32_t numbits = 0;
    uint32_t mask;

    for (uint32_t i = 0; i < 4; i++) {
      uint8_t byte = (uint8_t) (val >> (8 * (3 - i)));
      cw = codewords[byte];
      numbits = codewordlens[byte];

      while (numbits > 0) {
        int writebits = 32 - startbit < numbits ? 32 - startbit : numbits;
        if (numbits == writebits) {
          mask = (cw & ((1 << numbits) - 1)) << (32 - startbit - numbits);
          //first make sure that the start of the word is clean, then shift to the left as many places as you need
        } else {
          mask = cw >> (numbits - writebits); //shift out the bits that can not fit
        }
        *currentBytes = (*currentBytes) | mask;
        numbits = numbits - writebits;
        startbit = (startbit + writebits) % 32;
        if (startbit == 0) {
          currentBytes++;
          *currentBytes = 0x00000000;
          totalBytes += 4;
        }
      }
    }
  }
  totalBytes += (startbit + 8 - 1) / 8; // round up to nearest 8-bits
  *outsize = totalBytes;
}

