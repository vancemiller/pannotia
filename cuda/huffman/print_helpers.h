#ifndef _PRINT_HELPERS_H_
#define _PRINT_HELPERS_H_

#include <stdint.h>
#include <stdlib.h>

#include "parameters.h"
#include <stdio.h>

inline void printdbg_data_bin(const char* filename, uint32_t* data, uint32_t num_ints) {
  FILE* dump = fopen((const char*) filename, "wt");
  for (uint32_t i = 0; i < num_ints; i++) {
    uint32_t mask = 0x80000000;
    for (uint32_t j = 0; j < 32; j++) {
      if (data[i] & mask)
        fprintf(dump, "1");
      else
        fprintf(dump, "0");
      mask = mask >> 1;
    }
    fprintf(dump, "\n");
  }
  fclose(dump);
}

inline void printdbg_data_int(const char* filename, uint32_t* data, uint32_t num_ints) {
  FILE* dump = fopen((const char*) filename, "wt");
  for (uint32_t i = 0; i < num_ints; i++) {
    fprintf(dump, "%d: %d\n", i, data[i]);
  }
  fclose(dump);
}

inline void printdbg_gpu_data_detailed(FILE* gpudump, uint32_t* cw32, uint32_t* cw32len,
    uint32_t* cw32idx, uint32_t num_elements) {
  for (uint32_t i = 0; i < num_elements; i++) {
    fprintf(gpudump, "bp: %d, kc: %d, startbit: %d, cwlen: %d, cw:\t\t", cw32idx[i],
        cw32idx[i] / 32, cw32idx[i] % 32, cw32len[i]);
    //print codeword:
    uint32_t mask = 0x80000000;
    mask = mask >> (32 - cw32len[i]);
    for (uint32_t j = 0; j < cw32len[i]; j++) {
      if (cw32[i] & mask)
        fprintf(gpudump, "1");
      else
        fprintf(gpudump, "0");
      mask = mask >> 1;
    }
    fprintf(gpudump, "\n");
  }
}

inline void printdbg_gpu_data_detailed2(const char* filename, uint32_t* cw32,
    uint32_t* cw32len, uint32_t* cw32idx, uint32_t num_elements) {
  FILE* gpudump = fopen((const char*) filename, "wt");
  for (uint32_t i = 0; i < num_elements; i++) {
    fprintf(gpudump, "bp: %d, kc: %d, startbit: %d, cwlen: %d, cw:\t\t", cw32idx[i],
        cw32idx[i] / 32, cw32idx[i] % 32, cw32len[i]);
    //print codeword:
    uint32_t mask = 0x80000000;
    mask = mask >> (32 - cw32len[i]);
    for (uint32_t j = 0; j < cw32len[i]; j++) {
      if (cw32[i] & mask)
        fprintf(gpudump, "1");
      else
        fprintf(gpudump, "0");
      mask = mask >> 1;
    }
    fprintf(gpudump, "\n");
  }
  fclose(gpudump);
}

/************************************************************************/
/* BIT PRINTS                                                         */
/************************************************************************/
inline void printBits(uint8_t number) {
  uint8_t mask = 0x80;
  for (uint32_t j = 0; j < 8; j++) {
    if (number & mask)
      printf("1");
    else
      printf("0");
    mask = mask >> 1;
  }
  printf(" ");
}

inline void print32Bits(uint32_t number) {
  uint32_t mask = 0x80000000;
  for (uint32_t j = 0; j < 32; j++) {
    if (number & mask)
      printf("1");
    else
      printf("0");
    mask = mask >> 1;
  }
  printf("\n");
}

inline void print32BitsM(uint32_t marker) {
  for (uint32_t j = 0; j < 32; j++) {
    if (marker == (j + 1))
      printf("|");
    else
      printf(".");
  }
  printf("\n");
}

inline void print_array_char_as_bits(uint8_t* a, uint32_t len) {
  printf(" ========================= Printing vector =======================\n");
  printf("Total number of elements is %d\n", len);
  for (uint32_t i = 0; i < len; i++) {
    printf("a[%d]=%d \t", i, a[i]);
    printBits(a[i]);
    printf("\n");
  }
  printf("\n");
  printf(" ==================================================================\n");
}

inline void print_array_ints_as_bits(uint32_t* a, uint32_t len) {
  printf(" ========================= Printing vector =======================\n");
  for (uint32_t i = 0; i < len; i++) {
    print32Bits(a[i]);
    printf("\n");
  }
  printf("\n");
  printf(" ==================================================================\n");
}

inline void print_compare_array_ints_as_bits(uint32_t* a, uint32_t* b, uint32_t len) {
  printf(" ========================= Printing vector =======================\n");
  for (uint32_t i = 0; i < len; i++) {
    print32Bits(a[i]);
    print32Bits(b[i]);
    printf("\n");
  }
  printf("\n");
  printf(" ==================================================================\n");
}

inline void print_array_in_hex(uint32_t* a, uint32_t len) {
  printf(" ========================= Printing vector =======================\n");
  //printf("Total number of elements is %d\n", len);
  for (uint32_t i = 0; i < len; i++) {
    printf("%#X\t", a[i]);
  }
  printf("\n");
  printf(" ==================================================================\n");
}

/************************************************************************/
/* ARRAY PRINTS                                                        */
/***********************************************************************/

template<typename T> inline void print_array(T* a, uint32_t len) {
  printf(" ========================= Printing vector =======================\n");
  printf("Total number of elements is %d\n", len);
  for (uint32_t i = 0; i < len; i++) {
    printf("a[%d]=%d \t", i, a[i]);
  }
  printf("\n");
  printf(" ==================================================================\n");
}

template<typename ST, typename CT> inline void print_rled_arrays(ST* rle_symbols, CT* rle_counts,
    uint32_t rle_len) {
  ST current_symbol;
  CT current_count;
  printf(" ========================= Printing RLE vector =======================\n");
  printf(" Total number of RL Pairs is %d\n", rle_len);
  for (uint32_t k = 0; k < rle_len; k++) {
    current_symbol = rle_symbols[k];
    current_count = rle_counts[k];
    printf("(%d,%d) ,\t", current_symbol, current_count);
  }
  printf("\n");
}

inline void print_packed_rle_array(uint32_t* rle, uint32_t rle_len) {
  uint16_t current_symbol;
  uint16_t current_count;
  printf(" ========================= Printing RLE vector =======================\n");
  printf(" Total number of RL Pairs is %d\n", rle_len);
  for (uint32_t k = 0; k < rle_len; k++) {
    current_symbol = (uint16_t) (rle[k] >> 16);	//get the higher half-word
    current_count = (uint16_t) rle[k] & 0x0000FFFFF;		//get the shorter half-word
    printf("(%d,%d) ,\t", current_symbol, current_count);
  }
  printf("\n");
}

#endif // _PRINT_HELPERS_H_
