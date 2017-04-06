#ifndef _CE_H_
#define _CE_H_

extern "C" void cpu_vlc_encode(uint32_t* indata, uint32_t num_elements, uint32_t* outdata,
    uint32_t *outsize, uint32_t *codewords, uint32_t* codewordlens);
#endif

