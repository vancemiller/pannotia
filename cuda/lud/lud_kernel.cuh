#ifdef __cplusplus
extern "C" {
#endif
__global__ void lud_diagonal(float *m, int matrix_dim, int offset);
__global__ void lud_perimeter(float *m, int matrix_dim, int offset);
__global__ void lud_internal(float *m, int matrix_dim, int offset);
#ifdef __cplusplus
}
#endif
