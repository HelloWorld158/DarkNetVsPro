#include "darknet.h"
DARKNET_API int gpu_index = 0;

#ifdef GPU

#include "cuda.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>
int iMallocSize = 0;
int get_number_of_blocks(int array_size, int block_size)
{
    return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
}

void DecGenerateMemory(int iSize)
{
#ifdef MEMORYDEBUG
    iMallocSize -= iSize;
    printf("CudaMemoryFree:%d,AllSize:%d\n", iSize, iMallocSize);
#endif
}
int get_gpu_compute_capability(int i)
{
    typedef struct cudaDeviceProp cudaDeviceProp;
    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, i);
    check_error(status);
    int cc = prop.major * 100 + prop.minor * 10;    // __CUDA_ARCH__ format
    return cc;
}

void cuda_set_device(int n)
{
    gpu_index = n;
    cudaError_t status = cudaSetDevice(n);
    check_error(status);
}

int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    check_error(status);
    return n;
}

void check_error_extend(cudaError_t status, const char* file, int line, const char* date_time)
{
    //cudaDeviceSynchronize();
    cudaError_t statusbackup = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        printf("CUDA status Error: file: %s() : line: %d : build time: %s \n", file, line, date_time);
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (statusbackup != cudaSuccess)
    {   
        printf("CUDA status Error: file: %s() : line: %d : build time: %s \n", file, line, date_time);
        const char *s = cudaGetErrorString(statusbackup);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}
#ifdef CUDNN
void checkcudnnerror(cudnnStatus_t status)
{
    if (status != CUDNN_STATUS_SUCCESS)
    {
        char* s = cudnnGetErrorString(status);
        printf("cudnn Error:%s\n", s);
        assert(0);
        exit(0);
    }
}
#endif

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d = {x, y, 1};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

static cudaStream_t streamsArray[16];    // cudaStreamSynchronize( get_cuda_stream() );
static int streamInit[16] = { 0 };

cudaStream_t get_cuda_stream() {
    int i = cuda_get_device();
    if (!streamInit[i]) {
        //printf("Create CUDA-stream \n");
        cudaError_t status = cudaStreamCreate(&streamsArray[i]);
        //cudaError_t status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            printf(" cudaStreamCreate error: %d \n", status);
            const char* s = cudaGetErrorString(status);
            printf("CUDA Error: %s\n", s);
            status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamDefault);
            check_error(status);
        }
        streamInit[i] = 1;
    }
    return streamsArray[i];
}

static cudaStream_t streamsArrayMem[16];    // cudaStreamSynchronize( get_cuda_memcpy_stream() );
static int streamInitMem[16] = { 0 };

cudaStream_t get_cuda_memcpy_stream() {
    int i = cuda_get_device();
    if (!streamInitMem[i]) {
        cudaError_t status = cudaStreamCreate(&streamsArrayMem[i]);
        //cudaError_t status = cudaStreamCreateWithFlags(&streamsArray2[i], cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            printf(" cudaStreamCreate-Memcpy error: %d \n", status);
            const char* s = cudaGetErrorString(status);
            printf("CUDA Error: %s\n", s);
            status = cudaStreamCreateWithFlags(&streamsArrayMem[i], cudaStreamDefault);
            check_error(status);
        }
        streamInitMem[i] = 1;
    }
    return streamsArrayMem[i];
}

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
        cudnnStatus_t status = cudnnSetStream(handle[i], get_cuda_stream());
        checkcudnnerror(status);
    }
    return handle[i];
}
#endif

cublasHandle_t blas_handle()
{
    static int init[16] = {0};
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
void CheckCudaMemory(int size)
{
    iMallocSize += size;
    printf("CudaMemoryAlloc:%d,AllSize:%d\n", size,iMallocSize);
}
void* cuda_make_byte_array(size_t n)
{
    void* x_gpu;
    size_t size = n;
    cudaError_t status = cudaMalloc((void**)&x_gpu, size);
#ifdef MEMORYDEBUG
    CheckCudaMemory(size );
#endif
    check_error(status);
    return x_gpu;
}
short* cuda_make_short_array(size_t n)
{
    void* x_gpu;
    size_t size = n*sizeof(short);
    cudaError_t status = cudaMalloc((void**)&x_gpu, size);
#ifdef MEMORYDEBUG
    CheckCudaMemory(size);
#endif
    check_error(status);
    return x_gpu;
}
float *cuda_make_array(float *x, size_t n)
{
    float *x_gpu;
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
#ifdef MEMORYDEBUG
    CheckCudaMemory(size);
#endif
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    } else {
        fill_gpu(n, 0, x_gpu, 1);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_random(float *x_gpu, size_t n)
{
    static curandGenerator_t gen[16];
    static int init[16] = {0};
    int i = cuda_get_device();
    if(!init[i]){
        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen[i], time(0));
        init[i] = 1;
    }
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError());
}

float cuda_compare(float *x_gpu, float *x, size_t n, char *s)
{
    float *tmp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    cudaError_t status = cudaMalloc((void **)&x_gpu, size);
    check_error(status);
    if(x){
        status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        check_error(status);
    }
    if(!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}

void cuda_free(float *x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}
void cuda_free_allType(void* x_gpu)
{
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}
void cuda_push_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

void cuda_push_array_async(float* x_gpu, float* x, size_t n)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyHostToDevice,get_cuda_stream());
    check_error(status);
}

void cuda_pull_array(float *x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

void cuda_pull_array_async(float* x_gpu, float* x, size_t n)
{
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMemcpyAsync(x, x_gpu, size, cudaMemcpyDefault, get_cuda_stream());
    check_error(status);
}

float cuda_mag_array(float *x_gpu, size_t n)
{
    float *temp = calloc(n, sizeof(float));
    cuda_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
    free(temp);
    return m;
}
#else
void cuda_set_device(int n){}

#endif
