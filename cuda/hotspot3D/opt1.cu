__global__ void hotspotOpt1(float *p, float* tIn, float *tOut, float sdc, int nx, int ny, int nz,
    float ce, float cw, float cn, float cs, float ct, float cb, float cc) {
  float amb_temp = 80.0;

  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  int c = i + j * nx;
  int xy = nx * ny;

  int W = (i == 0) ? c : c - 1;
  int E = (i == nx - 1) ? c : c + 1;
  int N = (j == 0) ? c : c - nx;
  int S = (j == ny - 1) ? c : c + nx;

  float temp1, temp2, temp3;
  temp1 = temp2 = tIn[c];
  temp3 = tIn[c + xy];
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S] + cn * tIn[N] + cb * temp1
    + ct * temp3 + sdc * p[c] + ct * amb_temp;
  c += xy;
  W += xy;
  E += xy;
  N += xy;
  S += xy;

  for (int k = 1; k < nz - 1; ++k) {
    temp1 = temp2;
    temp2 = temp3;
    temp3 = tIn[c + xy];
    tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S] + cn * tIn[N] + cb * temp1
      + ct * temp3 + sdc * p[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;
  }
  temp1 = temp2;
  temp2 = temp3;
  tOut[c] = cc * temp2 + cw * tIn[W] + ce * tIn[E] + cs * tIn[S] + cn * tIn[N] + cb * temp1
    + ct * temp3 + sdc * p[c] + ct * amb_temp;
  return;
}

void hotspot_opt1(float *p, float *tIn, float *tOut, int nx, int ny, int nz, float Cap, float Rx,
    float Ry, float Rz, float dt, int numiter, bool unified) {
  float ce, cw, cn, cs, ct, cb, cc;
  TIMESTAMP(t0);
  float stepDivCap = dt / Cap;
  ce = cw = stepDivCap / Rx;
  cn = cs = stepDivCap / Ry;
  ct = cb = stepDivCap / Rz;

  cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

  size_t s = sizeof(float) * nx * ny * nz;
  TIMESTAMP(t1);
  time_serial += ELAPSED(t0, t1);

  float *tIn_d, *tOut_d, *p_d;
  if (unified) {
    p_d = p;
    tIn_d = tIn;
    tOut_d = tOut;
  } else {
    cudaMalloc((void**) &p_d, s);
    cudaMalloc((void**) &tIn_d, s);
    cudaMalloc((void**) &tOut_d, s);
  }

  TIMESTAMP(t2);
  time_malloc += ELAPSED(t1, t2);

  if (!unified) {
    cudaMemcpy(tIn_d, tIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, s, cudaMemcpyHostToDevice);
  }
  TIMESTAMP(t3);
  time_copy_in += ELAPSED(t2, t3);

  cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

  dim3 block_dim(64, 4, 1);
  dim3 grid_dim(nx / 64, ny / 4, 1);

  TIMESTAMP(t4);
  time_serial += ELAPSED(t3, t4);
  for (int i = 0; i < numiter; ++i) {
    hotspotOpt1<<<grid_dim, block_dim>>>
      (p_d, tIn_d, tOut_d, stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
    float *t = tIn_d;
    tIn_d = tOut_d;
    tOut_d = t;
  }
  cudaDeviceSynchronize();
  TIMESTAMP(t5);
  time_kernel += ELAPSED(t4, t5);

  if (!unified) {
    cudaMemcpy(tOut, tOut_d, s, cudaMemcpyDeviceToHost);
  }
  TIMESTAMP(t6);
  time_copy_out += ELAPSED(t5, t6);

  if (!unified) {
    cudaFree(p_d);
    cudaFree(tOut_d);
    cudaFree(tIn_d);
  }
  TIMESTAMP(t7);
  time_free += ELAPSED(t6, t7);
  return;
}

