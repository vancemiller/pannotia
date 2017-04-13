//=====================================================================
//	MAIN FUNCTION
//=====================================================================
#include "helper_cuda.h"
void master(float timeinst,
					float* initvalue,
					float* parameter,
					float* finalvalue,
					float* com,

					float* d_initvalue,
					float* d_finalvalue,
					float* d_params,
					float* d_com, bool unified) {

	//=====================================================================
	//	VARIABLES
	//=====================================================================

	// counters
	int i;

	// offset pointers
	int initvalue_offset_ecc;																// 46 points
	int initvalue_offset_Dyad;															// 15 points
	int initvalue_offset_SL;																// 15 points
	int initvalue_offset_Cyt;																// 15 poitns

	// cuda
	dim3 threads;
	dim3 blocks;

	//=====================================================================
	//	execute ECC&CAM kernel - it runs ECC and CAMs in parallel
	//=====================================================================

	int d_initvalue_mem;
	d_initvalue_mem = EQUATIONS * sizeof(float);
	int d_finalvalue_mem;
	d_finalvalue_mem = EQUATIONS * sizeof(float);
	int d_params_mem;
	d_params_mem = PARAMETERS * sizeof(float);
	int d_com_mem;
	d_com_mem = 3 * sizeof(float);

  if (unified) {
    d_initvalue = initvalue;
    d_params = parameter;
    d_finalvalue = finalvalue;
    d_com = com;
  } else {
    checkCudaErrors(cudaMemcpy(d_initvalue, initvalue, d_initvalue_mem, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_params, parameter, d_params_mem, cudaMemcpyHostToDevice));
  }

	threads.x = NUMBER_THREADS;
	threads.y = 1;
	blocks.x = 2;
	blocks.y = 1;
	kernel<<<blocks, threads>>>(	timeinst,
															d_initvalue,
															d_finalvalue,
															d_params,
															d_com);

  if (!unified) {
    checkCudaErrors(cudaMemcpy(finalvalue, d_finalvalue, d_finalvalue_mem, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(com, d_com, d_com_mem, cudaMemcpyDeviceToHost));
  }

	//=====================================================================
	//	FINAL KERNEL
	//=====================================================================

	initvalue_offset_ecc = 0;												// 46 points
	initvalue_offset_Dyad = 46;											// 15 points
	initvalue_offset_SL = 61;											// 15 points
	initvalue_offset_Cyt = 76;												// 15 poitns

	kernel_fin(			initvalue,
								initvalue_offset_ecc,
								initvalue_offset_Dyad,
								initvalue_offset_SL,
								initvalue_offset_Cyt,
								parameter,
								finalvalue,
								com[0],
								com[1],
								com[2]);

	//=====================================================================
	//	COMPENSATION FOR NANs and INFs
	//=====================================================================

	for(i=0; i<EQUATIONS; i++){
		if (isnan(finalvalue[i]) == 1){
			finalvalue[i] = 0.0001;												// for NAN set rate of change to 0.0001
		}
		else if (isinf(finalvalue[i]) == 1){
			finalvalue[i] = 0.0001;												// for INF set rate of change to 0.0001
		}
	}

}
