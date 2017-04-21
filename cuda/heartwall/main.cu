//	DEFINE / INCLUDE

//	LIBRARIES

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <avilib.h>
#include <avimod.h>
#include <cuda.h>

//	STRUCTURES, GLOBAL STRUCTURE VARIABLES

#include "define.c"

#define TIMESTAMP(NAME) \
  struct timespec NAME; \
  if (clock_gettime(CLOCK_MONOTONIC, &NAME)) { \
    fprintf(stderr, "Failed to get time: %s\n", strerror(errno)); \
  }

#define ELAPSED(start, end) \
  ((uint64_t) 1e9 * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec)

params_common_change common_change;
__constant__ params_common_change d_common_change;

params_common* common;
__constant__ params_common d_common;

params_unique unique[ALL_POINTS];	// cannot determine size dynamically so choose more than usually needed
__constant__ params_unique d_unique[ALL_POINTS];

// KERNEL CODE

#include "kernel.cu"

//	WRITE DATA FUNCTION

void write_data(char* filename, int frameNo, int frames_processed, int endoPoints, int* input_a,
    int* input_b, int epiPoints, int* input_2a, int* input_2b) {

  //	VARIABLES

  FILE* fid;
  int i, j;

  //	OPEN FILE FOR READING

  fid = fopen(filename, "w+");
  if (fid == NULL) {
    printf("The file was not opened for writing\n");
    return;
  }

  //	WRITE VALUES TO THE FILE
  fprintf(fid, "Total AVI Frames: %d\n", frameNo);
  fprintf(fid, "Frames Processed: %d\n", frames_processed);
  fprintf(fid, "endoPoints: %d\n", endoPoints);
  fprintf(fid, "epiPoints: %d", epiPoints);
  for (j = 0; j < frames_processed; j++) {
    fprintf(fid, "\n---Frame %d---", j);
    fprintf(fid, "\n--endo--\n", j);
    for (i = 0; i < endoPoints; i++) {
      fprintf(fid, "%d\t", input_a[j + i * frameNo]);
    }
    fprintf(fid, "\n");
    for (i = 0; i < endoPoints; i++) {
      // if(input_b[j*size+i] > 2000) input_b[j*size+i]=0;
      fprintf(fid, "%d\t", input_b[j + i * frameNo]);
    }
    fprintf(fid, "\n--epi--\n", j);
    for (i = 0; i < epiPoints; i++) {
      //if(input_2a[j*size_2+i] > 2000) input_2a[j*size_2+i]=0;
      fprintf(fid, "%d\t", input_2a[j + i * frameNo]);
    }
    fprintf(fid, "\n");
    for (i = 0; i < epiPoints; i++) {
      //if(input_2b[j*size_2+i] > 2000) input_2b[j*size_2+i]=0;
      fprintf(fid, "%d\t", input_2b[j + i * frameNo]);
    }
  }
  //		CLOSE FILE

  fclose(fid);

}

//	MAIN FUNCTION
int main(int argc, char *argv[]) {
  long long time_serial = 0;
  long long time_copy_in = 0;
  long long time_copy_out = 0;
  long long time_kernel = 0;
  long long time_malloc = 0;
  long long time_free = 0;

  printf("WG size of kernel = %d \n", NUMBER_THREADS);
  //	VARIABLES

  // CUDA kernel execution parameters
  dim3 threads;
  dim3 blocks;

  // counter
  int i;
  int frames_processed;

  // frames
  char* video_file_name;
  avi_t* frames;
  fp* frame;

  // 	FRAME

  if (argc < 3) {
    printf("ERROR: usage: heartwall <inputfile> <num of frames>\n");
    exit(1);
  }

  TIMESTAMP(t0);
  // open movie file
  video_file_name = argv[1];
  frames = (avi_t*) AVI_open_input_file(video_file_name, 1);												// added casting
  if (frames == NULL) {
    AVI_print_error((char *) "Error with AVI_open_input_file");
    return -1;
  }

  bool unified = argc == 4;
  TIMESTAMP(t1);
  time_serial += ELAPSED(t0, t1);

  // common
  if (unified) {
    cudaMallocManaged(&common, sizeof(params_common));
  } else {
    common = (params_common*) malloc(sizeof(params_common));
  }
  TIMESTAMP(t2);
  time_malloc += ELAPSED(t1, t2);
  common->no_frames = AVI_video_frames(frames);
  common->frame_rows = AVI_video_height(frames);
  common->frame_cols = AVI_video_width(frames);
  common->frame_elem = common->frame_rows * common->frame_cols;
  common->frame_mem = sizeof(fp) * common->frame_elem;

  TIMESTAMP(t3);
  time_serial += ELAPSED(t2, t3);
  // pointers
  cudaMalloc((void **) &common_change.d_frame, common->frame_mem);
  TIMESTAMP(t4);
  time_malloc += ELAPSED(t3, t4);

  // 	CHECK INPUT ARGUMENTS
  frames_processed = atoi(argv[2]);
  if (frames_processed < 0 || frames_processed > common->no_frames) {
    printf("ERROR: %d is an incorrect number of frames specified, select in the range of 0-%d\n",
        frames_processed, common->no_frames);
    return 0;
  }

  //	HARDCODED INPUTS FROM MATLAB

  //	CONSTANTS

  common->sSize = 40;
  common->tSize = 25;
  common->maxMove = 10;
  common->alpha = 0.87;

  //	ENDO POINTS

  common->endoPoints = ENDO_POINTS;
  common->endo_mem = sizeof(int) * common->endoPoints;
  common->epiPoints = EPI_POINTS;
  common->epi_mem = sizeof(int) * common->epiPoints;
  TIMESTAMP(t5);
  time_serial += ELAPSED(t4, t5);
  if (unified) {
    cudaMallocManaged(&common->endoRow, common->endo_mem);
    cudaMallocManaged(&common->endoCol, common->endo_mem);
    cudaMallocManaged(&common->tEndoRowLoc, common->endo_mem * common->no_frames);
    cudaMallocManaged(&common->tEndoColLoc, common->endo_mem * common->no_frames);
    cudaMallocManaged(&common->epiRow, common->epi_mem);
    cudaMallocManaged(&common->epiCol, common->epi_mem);
    cudaMallocManaged(&common->tEpiRowLoc, common->epi_mem * common->no_frames);
    cudaMallocManaged(&common->tEpiColLoc, common->epi_mem * common->no_frames);
  } else {
    common->endoRow = (int*) malloc(common->endo_mem);
    cudaMalloc((void **) &common->d_endoRow, common->endo_mem);
    common->endoCol = (int *) malloc(common->endo_mem);
    cudaMalloc((void **) &common->d_endoCol, common->endo_mem);
    cudaMalloc((void **) &common->d_tEndoRowLoc, common->endo_mem * common->no_frames);
    common->tEndoRowLoc = (int *) malloc(common->endo_mem * common->no_frames);
    common->tEndoColLoc = (int *) malloc(common->endo_mem * common->no_frames);
    cudaMalloc((void **) &common->d_tEndoColLoc, common->endo_mem * common->no_frames);
    common->epiRow = (int *) malloc(common->epi_mem);
    cudaMalloc((void **) &common->d_epiRow, common->epi_mem);
    common->epiCol = (int *) malloc(common->epi_mem);
    cudaMalloc((void **) &common->d_epiCol, common->epi_mem);
    common->tEpiRowLoc = (int *) malloc(common->epi_mem * common->no_frames);
    common->tEpiColLoc = (int *) malloc(common->epi_mem * common->no_frames);
    cudaMalloc((void **) &common->d_tEpiColLoc, common->epi_mem * common->no_frames);
  }

  TIMESTAMP(t6);
  time_malloc += ELAPSED(t5, t6);
  common->endoRow[0] = 369;
  common->endoRow[1] = 400;
  common->endoRow[2] = 429;
  common->endoRow[3] = 452;
  common->endoRow[4] = 476;
  common->endoRow[5] = 486;
  common->endoRow[6] = 479;
  common->endoRow[7] = 458;
  common->endoRow[8] = 433;
  common->endoRow[9] = 404;
  common->endoRow[10] = 374;
  common->endoRow[11] = 346;
  common->endoRow[12] = 318;
  common->endoRow[13] = 294;
  common->endoRow[14] = 277;
  common->endoRow[15] = 269;
  common->endoRow[16] = 275;
  common->endoRow[17] = 287;
  common->endoRow[18] = 311;
  common->endoRow[19] = 339;

  common->endoCol[0] = 408;
  common->endoCol[1] = 406;
  common->endoCol[2] = 397;
  common->endoCol[3] = 383;
  common->endoCol[4] = 354;
  common->endoCol[5] = 322;
  common->endoCol[6] = 294;
  common->endoCol[7] = 270;
  common->endoCol[8] = 250;
  common->endoCol[9] = 237;
  common->endoCol[10] = 235;
  common->endoCol[11] = 241;
  common->endoCol[12] = 254;
  common->endoCol[13] = 273;
  common->endoCol[14] = 300;
  common->endoCol[15] = 328;
  common->endoCol[16] = 356;
  common->endoCol[17] = 383;
  common->endoCol[18] = 401;
  common->endoCol[19] = 411;

  common->epiRow[0] = 390;
  common->epiRow[1] = 419;
  common->epiRow[2] = 448;
  common->epiRow[3] = 474;
  common->epiRow[4] = 501;
  common->epiRow[5] = 519;
  common->epiRow[6] = 535;
  common->epiRow[7] = 542;
  common->epiRow[8] = 543;
  common->epiRow[9] = 538;
  common->epiRow[10] = 528;
  common->epiRow[11] = 511;
  common->epiRow[12] = 491;
  common->epiRow[13] = 466;
  common->epiRow[14] = 438;
  common->epiRow[15] = 406;
  common->epiRow[16] = 376;
  common->epiRow[17] = 347;
  common->epiRow[18] = 318;
  common->epiRow[19] = 291;
  common->epiRow[20] = 275;
  common->epiRow[21] = 259;
  common->epiRow[22] = 256;
  common->epiRow[23] = 252;
  common->epiRow[24] = 252;
  common->epiRow[25] = 257;
  common->epiRow[26] = 266;
  common->epiRow[27] = 283;
  common->epiRow[28] = 305;
  common->epiRow[29] = 331;
  common->epiRow[30] = 360;

  common->epiCol[0] = 457;
  common->epiCol[1] = 454;
  common->epiCol[2] = 446;
  common->epiCol[3] = 431;
  common->epiCol[4] = 411;
  common->epiCol[5] = 388;
  common->epiCol[6] = 361;
  common->epiCol[7] = 331;
  common->epiCol[8] = 301;
  common->epiCol[9] = 273;
  common->epiCol[10] = 243;
  common->epiCol[11] = 218;
  common->epiCol[12] = 196;
  common->epiCol[13] = 178;
  common->epiCol[14] = 166;
  common->epiCol[15] = 157;
  common->epiCol[16] = 155;
  common->epiCol[17] = 165;
  common->epiCol[18] = 177;
  common->epiCol[19] = 197;
  common->epiCol[20] = 218;
  common->epiCol[21] = 248;
  common->epiCol[22] = 276;
  common->epiCol[23] = 304;
  common->epiCol[24] = 333;
  common->epiCol[25] = 361;
  common->epiCol[26] = 391;
  common->epiCol[27] = 415;
  common->epiCol[28] = 434;
  common->epiCol[29] = 448;
  common->epiCol[30] = 455;

  TIMESTAMP(t7);
  time_serial += ELAPSED(t6, t7);
  if (unified) {
    common->d_endoRow = common->endoRow;
    common->d_endoCol = common->endoCol;
    common->d_tEndoRowLoc = common->tEndoRowLoc;
    common->d_tEndoColLoc = common->tEndoColLoc;
    common->d_epiRow = common->epiRow;
    common->d_epiCol = common->epiCol;
    common->d_tEpiRowLoc = common->tEpiRowLoc;
    common->d_tEpiColLoc = common->tEpiColLoc;
  } else {
    cudaMemcpy(common->d_endoRow, common->endoRow, common->endo_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(common->d_endoCol, common->endoCol, common->endo_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(common->d_epiRow, common->epiRow, common->epi_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(common->d_epiCol, common->epiCol, common->epi_mem, cudaMemcpyHostToDevice);
    cudaMalloc((void **) &common->d_tEpiRowLoc, common->epi_mem * common->no_frames);
  }
  TIMESTAMP(t8);
  time_copy_in += ELAPSED(t7, t8);
  //	ALL POINTS

  common->allPoints = ALL_POINTS;

  // 	TEMPLATE SIZES

  // common
  common->in_rows = common->tSize + 1 + common->tSize;
  common->in_cols = common->in_rows;
  common->in_elem = common->in_rows * common->in_cols;
  common->in_mem = sizeof(fp) * common->in_elem;

  // 	CREATE ARRAY OF TEMPLATES FOR ALL POINTS

  TIMESTAMP(t9);
  time_serial += ELAPSED(t8, t9);

  // common
  cudaMalloc((void **) &common->d_endoT, common->in_mem * common->endoPoints);
  cudaMalloc((void **) &common->d_epiT, common->in_mem * common->epiPoints);

  TIMESTAMP(t10);
  time_malloc += ELAPSED(t9, t10);

  //	SPECIFIC TO ENDO OR EPI TO BE SET HERE

  for (i = 0; i < common->endoPoints; i++) {
    unique[i].point_no = i;
    unique[i].d_Row = common->d_endoRow;
    unique[i].d_Col = common->d_endoCol;
    unique[i].d_tRowLoc = common->d_tEndoRowLoc;
    unique[i].d_tColLoc = common->d_tEndoColLoc;
    unique[i].d_T = common->d_endoT;
  }
  for (i = common->endoPoints; i < common->allPoints; i++) {
    unique[i].point_no = i - common->endoPoints;
    unique[i].d_Row = common->d_epiRow;
    unique[i].d_Col = common->d_epiCol;
    unique[i].d_tRowLoc = common->d_tEpiRowLoc;
    unique[i].d_tColLoc = common->d_tEpiColLoc;
    unique[i].d_T = common->d_epiT;
  }

  // 	RIGHT TEMPLATE 	FROM 	TEMPLATE ARRAY

  // pointers
  for (i = 0; i < common->allPoints; i++) {
    unique[i].in_pointer = unique[i].point_no * common->in_elem;
  }

  // 	AREA AROUND POINT		FROM	FRAME

  // common
  common->in2_rows = 2 * common->sSize + 1;
  common->in2_cols = 2 * common->sSize + 1;
  common->in2_elem = common->in2_rows * common->in2_cols;
  common->in2_mem = sizeof(float) * common->in2_elem;

  // 	CONVOLUTION

  // common
  common->conv_rows = common->in_rows + common->in2_rows - 1;									// number of rows in I
  common->conv_cols = common->in_cols + common->in2_cols - 1;							// number of columns in I
  common->conv_elem = common->conv_rows * common->conv_cols;									// number of elements
  common->conv_mem = sizeof(float) * common->conv_elem;
  common->ioffset = 0;
  common->joffset = 0;

  // 	CUMULATIVE SUM

  // 	PADDING OF ARRAY, VERTICAL CUMULATIVE SUM

  // common
  common->in2_pad_add_rows = common->in_rows;
  common->in2_pad_add_cols = common->in_cols;

  common->in2_pad_cumv_rows = common->in2_rows + 2 * common->in2_pad_add_rows;
  common->in2_pad_cumv_cols = common->in2_cols + 2 * common->in2_pad_add_cols;
  common->in2_pad_cumv_elem = common->in2_pad_cumv_rows * common->in2_pad_cumv_cols;
  common->in2_pad_cumv_mem = sizeof(float) * common->in2_pad_cumv_elem;

  // 	SELECTION

  // common
  common->in2_pad_cumv_sel_rowlow = 1 + common->in_rows;													// (1 to n+1)
  common->in2_pad_cumv_sel_rowhig = common->in2_pad_cumv_rows - 1;
  common->in2_pad_cumv_sel_collow = 1;
  common->in2_pad_cumv_sel_colhig = common->in2_pad_cumv_cols;
  common->in2_pad_cumv_sel_rows = common->in2_pad_cumv_sel_rowhig - common->in2_pad_cumv_sel_rowlow
    + 1;
  common->in2_pad_cumv_sel_cols = common->in2_pad_cumv_sel_colhig - common->in2_pad_cumv_sel_collow
    + 1;
  common->in2_pad_cumv_sel_elem = common->in2_pad_cumv_sel_rows * common->in2_pad_cumv_sel_cols;
  common->in2_pad_cumv_sel_mem = sizeof(float) * common->in2_pad_cumv_sel_elem;

  // 	SELECTION	2, SUBTRACTION, HORIZONTAL CUMULATIVE SUM

  // common
  common->in2_pad_cumv_sel2_rowlow = 1;
  common->in2_pad_cumv_sel2_rowhig = common->in2_pad_cumv_rows - common->in_rows - 1;
  common->in2_pad_cumv_sel2_collow = 1;
  common->in2_pad_cumv_sel2_colhig = common->in2_pad_cumv_cols;
  common->in2_sub_cumh_rows = common->in2_pad_cumv_sel2_rowhig - common->in2_pad_cumv_sel2_rowlow
    + 1;
  common->in2_sub_cumh_cols = common->in2_pad_cumv_sel2_colhig - common->in2_pad_cumv_sel2_collow
    + 1;
  common->in2_sub_cumh_elem = common->in2_sub_cumh_rows * common->in2_sub_cumh_cols;
  common->in2_sub_cumh_mem = sizeof(float) * common->in2_sub_cumh_elem;

  // common
  common->in2_sub_cumh_sel_rowlow = 1;
  common->in2_sub_cumh_sel_rowhig = common->in2_sub_cumh_rows;
  common->in2_sub_cumh_sel_collow = 1 + common->in_cols;
  common->in2_sub_cumh_sel_colhig = common->in2_sub_cumh_cols - 1;
  common->in2_sub_cumh_sel_rows = common->in2_sub_cumh_sel_rowhig - common->in2_sub_cumh_sel_rowlow
    + 1;
  common->in2_sub_cumh_sel_cols = common->in2_sub_cumh_sel_colhig - common->in2_sub_cumh_sel_collow
    + 1;
  common->in2_sub_cumh_sel_elem = common->in2_sub_cumh_sel_rows * common->in2_sub_cumh_sel_cols;
  common->in2_sub_cumh_sel_mem = sizeof(float) * common->in2_sub_cumh_sel_elem;

  //	SELECTION 2, SUBTRACTION

  // common
  common->in2_sub_cumh_sel2_rowlow = 1;
  common->in2_sub_cumh_sel2_rowhig = common->in2_sub_cumh_rows;
  common->in2_sub_cumh_sel2_collow = 1;
  common->in2_sub_cumh_sel2_colhig = common->in2_sub_cumh_cols - common->in_cols - 1;
  common->in2_sub2_rows = common->in2_sub_cumh_sel2_rowhig - common->in2_sub_cumh_sel2_rowlow + 1;
  common->in2_sub2_cols = common->in2_sub_cumh_sel2_colhig - common->in2_sub_cumh_sel2_collow + 1;
  common->in2_sub2_elem = common->in2_sub2_rows * common->in2_sub2_cols;
  common->in2_sub2_mem = sizeof(float) * common->in2_sub2_elem;

  //	CUMULATIVE SUM 2

  //	MULTIPLICATION

  // common
  common->in2_sqr_rows = common->in2_rows;
  common->in2_sqr_cols = common->in2_cols;
  common->in2_sqr_elem = common->in2_elem;
  common->in2_sqr_mem = common->in2_mem;

  //	SELECTION 2, SUBTRACTION

  // common
  common->in2_sqr_sub2_rows = common->in2_sub2_rows;
  common->in2_sqr_sub2_cols = common->in2_sub2_cols;
  common->in2_sqr_sub2_elem = common->in2_sub2_elem;
  common->in2_sqr_sub2_mem = common->in2_sub2_mem;

  //	FINAL

  // common
  common->in_sqr_rows = common->in_rows;
  common->in_sqr_cols = common->in_cols;
  common->in_sqr_elem = common->in_elem;
  common->in_sqr_mem = common->in_mem;

  //	TEMPLATE MASK CREATE

  // common
  common->tMask_rows = common->in_rows + (common->sSize + 1 + common->sSize) - 1;
  common->tMask_cols = common->tMask_rows;
  common->tMask_elem = common->tMask_rows * common->tMask_cols;
  common->tMask_mem = sizeof(float) * common->tMask_elem;

  //	POINT MASK INITIALIZE

  // common
  common->mask_rows = common->maxMove;
  common->mask_cols = common->mask_rows;
  common->mask_elem = common->mask_rows * common->mask_cols;
  common->mask_mem = sizeof(float) * common->mask_elem;

  //	MASK CONVOLUTION

  // common
  common->mask_conv_rows = common->tMask_rows;												// number of rows in I
  common->mask_conv_cols = common->tMask_cols;												// number of columns in I
  common->mask_conv_elem = common->mask_conv_rows * common->mask_conv_cols;		// number of elements
  common->mask_conv_mem = sizeof(float) * common->mask_conv_elem;
  common->mask_conv_ioffset = (common->mask_rows - 1) / 2;
  if ((common->mask_rows - 1) % 2 > 0.5) {
    common->mask_conv_ioffset = common->mask_conv_ioffset + 1;
  }
  common->mask_conv_joffset = (common->mask_cols - 1) / 2;
  if ((common->mask_cols - 1) % 2 > 0.5) {
    common->mask_conv_joffset = common->mask_conv_joffset + 1;
  }

  TIMESTAMP(t11);
  time_serial += ELAPSED(t10, t10);

  // pointers
  for (i = 0; i < common->allPoints; i++) {
    cudaMalloc((void **) &unique[i].d_in2, common->in2_mem);
    cudaMalloc((void **) &unique[i].d_conv, common->conv_mem);
    cudaMalloc((void **) &unique[i].d_in2_pad_cumv, common->in2_pad_cumv_mem);
    cudaMalloc((void **) &unique[i].d_in2_pad_cumv_sel, common->in2_pad_cumv_sel_mem);
    cudaMalloc((void **) &unique[i].d_in2_sub_cumh, common->in2_sub_cumh_mem);
    cudaMalloc((void **) &unique[i].d_in2_sub_cumh_sel, common->in2_sub_cumh_sel_mem);
    cudaMalloc((void **) &unique[i].d_in2_sub2, common->in2_sub2_mem);
    cudaMalloc((void **) &unique[i].d_in2_sqr, common->in2_sqr_mem);
    cudaMalloc((void **) &unique[i].d_in2_sqr_sub2, common->in2_sqr_sub2_mem);
    cudaMalloc((void **) &unique[i].d_in_sqr, common->in_sqr_mem);
    cudaMalloc((void **) &unique[i].d_tMask, common->tMask_mem);
    cudaMalloc((void **) &unique[i].d_mask_conv, common->mask_conv_mem);
  }

  TIMESTAMP(t12);
  time_malloc += ELAPSED(t11, t12);

  //	KERNEL

  //	THREAD BLOCK

  // All kernels operations within kernel use same max size of threads. Size of block size is set to the size appropriate for max size operation (on padded matrix). Other use subsets of that.
  threads.x = NUMBER_THREADS;											// define the number of threads in the block
  threads.y = 1;
  blocks.x = common->allPoints;							// define the number of blocks in the grid
  blocks.y = 1;

  //	COPY ARGUMENTS

  cudaMemcpyToSymbol(d_common, &common, sizeof(params_common));
  cudaMemcpyToSymbol(d_unique, &unique, sizeof(params_unique) * ALL_POINTS);

  TIMESTAMP(t13);
  time_copy_in += ELAPSED(t12, t13);

  //	PRINT FRAME PROGRESS START

  printf("frame progress: ");
  fflush (NULL);

  //	LAUNCH

  for (common_change.frame_no = 0; common_change.frame_no < frames_processed;
      common_change.frame_no++) {
    TIMESTAMP(t14);

    // Extract a cropped version of the first frame from the video file
    frame = get_frame(frames,						// pointer to video file
        common_change.frame_no,				// number of frame that needs to be returned
        0,								// cropped?
        0,								// scaled?
        1);							// converted

    TIMESTAMP(t15);
    time_serial += ELAPSED(t14, t15);
    // copy frame to GPU memory
    cudaMemcpy(common_change.d_frame, frame, common->frame_mem, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_common_change, &common_change, sizeof(params_common_change));
    TIMESTAMP(t16);
    time_copy_in += ELAPSED(t15, t16);
    // launch GPU kernel
    kernel<<<blocks, threads>>>();
    TIMESTAMP(t17);
    time_kernel += ELAPSED(t16, t17);

    // free frame after each loop iteration, since AVI library allocates memory for every frame fetched
    free(frame);

    TIMESTAMP(t18);
    time_free += ELAPSED(t17, t18);

    // print frame progress
    printf("%d ", common_change.frame_no);
    fflush(NULL);

  }

  //	PRINT FRAME PROGRESS END

  printf("\n");
  fflush(NULL);

  //	OUTPUT

  TIMESTAMP(t14);
  if (!unified) {
    cudaMemcpy(common->tEndoRowLoc, common->d_tEndoRowLoc, common->endo_mem * common->no_frames,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(common->tEndoColLoc, common->d_tEndoColLoc, common->endo_mem * common->no_frames,
        cudaMemcpyDeviceToHost);

    cudaMemcpy(common->tEpiRowLoc, common->d_tEpiRowLoc, common->epi_mem * common->no_frames,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(common->tEpiColLoc, common->d_tEpiColLoc, common->epi_mem * common->no_frames,
        cudaMemcpyDeviceToHost);
  }
  TIMESTAMP(t15);
  time_copy_out += ELAPSED(t14, t15);

#ifdef OUTPUT

  //	DUMP DATA TO FILE
  write_data( "result.txt",
      common->no_frames,
      frames_processed,
      common->endoPoints,
      common->tEndoRowLoc,
      common->tEndoColLoc,
      common->epiPoints,
      common->tEpiRowLoc,
      common->tEpiColLoc);

  //	End

#endif

  //	DEALLOCATION

  //	COMMON

  TIMESTAMP(t16);
  time_serial += ELAPSED(t15, t16);

  // frame
  cudaFree(common_change.d_frame);

  // endo points
  if (!unified) {
    free(common->endoRow);
    free(common->endoCol);
    free(common->tEndoRowLoc);
    free(common->tEndoColLoc);

    cudaFree(common->d_endoRow);
    cudaFree(common->d_endoCol);
    cudaFree(common->d_tEndoRowLoc);
    cudaFree(common->d_tEndoColLoc);

    cudaFree(common->d_endoT);

    // epi points
    free(common->epiRow);
    free(common->epiCol);
    free(common->tEpiRowLoc);
    free(common->tEpiColLoc);

    cudaFree(common->d_epiRow);
    cudaFree(common->d_epiCol);
    cudaFree(common->d_tEpiRowLoc);
    cudaFree(common->d_tEpiColLoc);

    cudaFree(common->d_epiT);
  }

  //	POINTERS

  for (i = 0; i < common->allPoints; i++) {
    cudaFree(unique[i].d_in2);

    cudaFree(unique[i].d_conv);
    cudaFree(unique[i].d_in2_pad_cumv);
    cudaFree(unique[i].d_in2_pad_cumv_sel);
    cudaFree(unique[i].d_in2_sub_cumh);
    cudaFree(unique[i].d_in2_sub_cumh_sel);
    cudaFree(unique[i].d_in2_sub2);
    cudaFree(unique[i].d_in2_sqr);
    cudaFree(unique[i].d_in2_sqr_sub2);
    cudaFree(unique[i].d_in_sqr);

    cudaFree(unique[i].d_tMask);
    cudaFree(unique[i].d_mask_conv);
  }

  TIMESTAMP(t17);
  time_free += ELAPSED(t16, t17);

  printf("====Timing info====\n");
  printf("time serial = %f ms\n", time_serial * 1e-6);
  printf("time GPU malloc = %f ms\n", time_malloc * 1e-6);
  printf("time CPU to GPU memory copy = %f ms\n", time_copy_in * 1e-6);
  printf("time kernel = %f ms\n", time_kernel * 1e-6);
  printf("time GPU to CPU memory copy back = %f ms\n", time_copy_out * 1e-6);
  printf("time GPU free = %f ms\n", time_free * 1e-6);
  printf("End-to-end = %f ms\n", ELAPSED(t0, t17) * 1e-6);
}

//	MAIN FUNCTION

