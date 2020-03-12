#ifndef CUDA_H
#define CUDA_H

#include "darknet.h"

#ifdef GPU
#ifdef __cplusplus
extern "C" {
#endif
	void check_error_extend(cudaError_t status, const char* file, int line, const char* date_time);
#ifdef _WINDOWS
#define check_error(s) check_error_extend(s, __FILE__ " : " __FUNCTION__, __LINE__,  __DATE__ " - " __TIME__ )
#else
#define check_error(s) check_error_extend(s,"linux", 0,  "noTime" )
#endif
	cublasHandle_t blas_handle();
	int* cuda_make_int_array(int* x, size_t n);
	void cuda_random(float* x_gpu, size_t n);
	float cuda_compare(float* x_gpu, float* x, size_t n, char* s);
	dim3 cuda_gridsize(size_t n);
	cudaStream_t get_cuda_stream();
	cudaStream_t get_cuda_memcpy_stream();
	int get_number_of_blocks(int array_size, int block_size);
#ifdef __cplusplus
}
#endif
#ifdef CUDNN
#ifdef __cplusplus
extern "C" {
#endif
	cudnnHandle_t cudnn_handle();
	void checkcudnnerror(cudnnStatus_t status);
#ifdef __cplusplus
}
#endif
#endif

#endif
#endif
