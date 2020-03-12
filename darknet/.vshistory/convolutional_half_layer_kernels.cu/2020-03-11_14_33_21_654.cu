#include "pch.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_fp16.hpp"
#include "cuda.h"
extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}
half* publicMemory[2] = {0,0};
int pMSize[2] = {0,0};
extern "C" cudnnDataType_t GetDataType();
#ifdef TESTPROGRESS16
float* tempBuffer=0;
float* tempWeight = 0;
int iMaxSize=0;
#endif
void MakeHalfMaxSize(int iGiveSize,int iOutSize)
{
   size_t size[2] = {iGiveSize,iOutSize};
   for (int cnum = 0; cnum < 2; cnum++)
   {
       if (pMSize[cnum] < size[cnum])
       {
           if (publicMemory[cnum])
           {
               DecGenerateMemory(pMSize[cnum] * sizeof(half));
               cuda_free_allType(publicMemory[cnum]);
           }
           pMSize[cnum] = size[cnum];
           publicMemory[cnum]=(half *)cuda_make_short_array(pMSize[cnum]);
       }
#ifdef TESTPROGRESS16
       if (iMaxSize < pMSize[cnum])
       {
           iMaxSize = pMSize[cnum];
           if (tempBuffer) cuda_free(tempBuffer);
           tempBuffer = cuda_make_array(0, iMaxSize);
           if (tempWeight) cuda_free_allType(tempWeight);
           tempWeight = cuda_make_array(0, iMaxSize);
       }
#endif
   }        

}
__global__ void cuda_f32_to_f16(float* input_f32, size_t size, half* output_f16)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_f16[idx] = __float2half(input_f32[idx]);
    //if (idx < size) output_f16[idx] = __float2half_rn(input_f32[idx]); // can't be compiled on Linux without casting
    // __float2half_ru, __float2half_rd, __float2half_rz, __float2half_rn
    //if (idx < size) *((unsigned short *)output_f16 + idx) = __float2half(input_f32[idx]);
}

void cuda_convert_f32_to_f16(float* input_f32, size_t size, half* output_f16) {
    cuda_f32_to_f16 << < cuda_gridsize(size), BLOCK,0,get_cuda_stream() >> > (input_f32, size, (half*)output_f16);
    check_error(cudaPeekAtLastError());
}

__global__ void cuda_f16_to_f32(half* input_f16, size_t size, float* output_f32)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output_f32[idx] = __half2float(input_f16[idx]);
    //if (idx < size) output_f32[idx] = __half2float(*((unsigned short *)input_f16 + idx));
}

void cuda_convert_f16_to_f32(half* input_f16, size_t size, float* output_f32) {
    cuda_f16_to_f32 << < cuda_gridsize(size), BLOCK,0,get_cuda_stream() >> > ((half*)input_f16, size, output_f32);
    check_error(cudaPeekAtLastError());
}
void DealWeightBuffer(convolutional_layer* l)
{
    //return;
#ifdef GETDATATYPE
    if (GetDataType() != CUDNN_DATA_HALF) return;
#endif   
#ifdef DEALWEIGHTBUFFER
    OutPutGPUMemory(l.weights_gpu, l.nweights, 0);
#endif
    half* halfWeights = 0;
    halfWeights=(half *)cuda_make_short_array(l->nweights);
    cuda_convert_f32_to_f16(l->weights_gpu, l->nweights, halfWeights);
#ifdef DEALWEIGHTBUFFER
    float* fResult=0;
    check_error(cudaMalloc((void**)&fResult, l.nweights * sizeof(float)));
    cuda_convert_f16_to_f32(halfWeights, l.nweights, fResult);
    OutPutGPUMemory(fResult, l.nweights, 0);
#endif           
    cuda_free(l->weights_gpu);
    DecGenerateMemory(l->nweights * sizeof(float));
    l->weights_gpu = (float *)halfWeights;
    LAYERDATA* layerdata = (LAYERDATA *)l->layerdata;
    CONVPROP* prop=(CONVPROP *)layerdata->layerData;
    if (prop->bUnSupportBias) return;
    half* bias = (half*)cuda_make_short_array(l->n);
    cuda_convert_f32_to_f16(l->biases_gpu, l->n, bias);
    cuda_free(l->biases_gpu);
    DecGenerateMemory(l->n * sizeof(float));
    l->biases_gpu = (float*)bias;
}
#ifdef GPUHALFABILITY
__global__ void add_bias_half_kernel(half* output, half* biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;
    if (offset >= size) return;
    half a = output[(batch * n + filter) * size + offset];
    output[(batch * n + filter) * size + offset] =__hadd(a, biases[filter]);
}

void add_bias_half_gpu(half* output, half* biases, int batch, int n, int size)
{
    dim3 dimGrid((size - 1) / BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    add_bias_half_kernel << <dimGrid, dimBlock, 0, get_cuda_stream() >> > (output, biases, n, size);
    check_error(cudaPeekAtLastError());
}
__global__ void activate_array_hardtan_halfadd_kernel(half* output, half* biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;
    if (offset >= size) return;
    int iOutDex = (batch * n + filter) * size + offset;
    half a = output[iOutDex];
    half b = __hadd(a, biases[filter]);
    if (__hlt(b, half(-1.0f))) output[iOutDex] = half(-1.0f);
    if (__hgt(b, half(1.0f))) output[iOutDex] = half(1.0f);
    output[iOutDex] = b;
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    //if (index < n) {
    //    float a = x[index];
    //    if (a < -1) a = -1;
    //    if (a > 1) a = 1;
    //    x[index] = a;//hardtan_activate_kernel(x[index]);
    //}
}

__global__ void activate_array_relu_halfadd_kernel(half* output, half* biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;
    if (offset >= size) return;
    int iOutDex = (batch * n + filter) * size + offset;
    half a = output[iOutDex];
    half b = __hadd(a, biases[filter]);    
    if (__hgt(b, half(0.0f))) output[iOutDex] = b;
    else output[iOutDex] = half(0.0f);
    //output[iOutDex] = b;
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    //if (index < n) {
    //    float a = x[index];
    //    x[index] = a * (a > 0);// relu_activate_kernel(x[index]);
    //}
}
__global__ void activate_array_leaky_halfadd_kernel(half* output, half* biases, int n, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;
    if (offset >= size) return;
    int iOutDex = (batch * n + filter) * size + offset;
    half a = output[iOutDex];
    half b = __hadd(a, biases[filter]);
    if (__hgt(b, half(0.0f))) output[iOutDex] = b;
    else output[iOutDex] =__hmul(half(0.1f),b);
    //int index = blockIdx.x * blockDim.x + threadIdx.x;
    //if (index < n) {
    //    float a = x[index];
    //    x[index] = (a > 0) ? a : .1f * a; //leaky_activate_kernel(x[index]);
    //}
}
//__global__ void activate_array_selu_halfadd_kernel(half* output, half* biases, int n, int size)
//{
//    int offset = blockIdx.x * blockDim.x + threadIdx.x;
//    int filter = blockIdx.y;
//    int batch = blockIdx.z;
//    if (offset >= size) return;
//    int iOutDex = (batch * n + filter) * size + offset;
//    half a = output[iOutDex];
//    half b = __hadd(a, biases[filter]);
//    if (__hgt(b, half(0.0f))) output[iOutDex] = b;
//    else output[iOutDex] = __hmul(half(0.1f), b);
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    if (index < n) {
//        float a = x[index];
//        x[index] = (a >= 0) * 1.0507f * a + (a < 0) * 1.0507f * 1.6732f * (expf(a) - 1);
//    }
//}
//
//__global__ void activate_array_logistic_halfadd_kernel(half* output, half* biases, int n, int size)
//{
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    if (index < n) {
//        float a = x[index];
//        x[index] = 1.f / (1.f + expf(-a));
//    }
//}
//
//__global__ void activate_array_tanh_halfadd_kernel(half* output, half* biases, int n, int size)
//{
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    if (index < n) {
//        float a = x[index];
//        x[index] = (2.f / (1 + expf(-2 * a)) - 1);
//    }
//}
#endif

void add_bias_activation_half_gpu(half* output, half* biases, int batch, int n, int size
    ,ACTIVATION act,int bUnSupportAct,int bUnsportBias)
{
#ifdef GPUHALFABILITY
    if (bUnsportBias) return;
    if (bUnSupportAct)
    {
        add_bias_half_gpu(output, biases, batch, n, size);
        return;
    }
    dim3 dimGrid((size - 1) / BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);
    switch (act)
    {
    case RELU:
        activate_array_relu_halfadd_kernel << <dimGrid, dimBlock, 0, get_cuda_stream() >> > (output, biases, n, size);
        break;
    case LINEAR:
        break;
    case LEAKY:
        activate_array_leaky_halfadd_kernel << <dimGrid, dimBlock, 0, get_cuda_stream() >> > (output, biases, n, size);
        break;
    case HARDTAN:
        activate_array_hardtan_halfadd_kernel << <dimGrid, dimBlock, 0, get_cuda_stream() >> > (output, biases, n, size);
        break;
   /* case SELU:
        activate_array_selu_halfadd_kernel << <dimGrid, dimBlock, 0, get_cuda_stream() >> > (output, biases, n, size);
        break;
    case LOGISTIC:
        activate_array_logistic_halfadd_kernel << <dimGrid, dimBlock, 0, get_cuda_stream() >> > (output, biases, n, size);
        break;
    case TANH:
        activate_array_tanh_halfadd_kernel << <dimGrid, dimBlock, 0, get_cuda_stream() >> > (output, biases, n, size);
        break;*/
    }
    check_error(cudaPeekAtLastError());
#endif
}
void forward_convolutional_layer_gpu_predict_Float16(convolutional_layer l, network net)
{
    if (l.binary) {
        binarize_weights_gpu(l.weights_gpu, l.n, l.c / l.groups * l.size * l.size, l.binary_weights_gpu);
        swap_binary(&l);
    }

    if (l.xnor) {
        binarize_weights_gpu(l.weights_gpu, l.n, l.c / l.groups * l.size * l.size, l.binary_weights_gpu);
        swap_binary(&l);
        binarize_gpu(net.input_gpu, l.c * l.h * l.w * l.batch, l.binary_input_gpu);
        net.input_gpu = l.binary_input_gpu;
    }
    float one = 1.0f,zero=0.0f;
#ifdef MEMORYDEBUG
    printf("gpuInput:0x%x,gpuOutput:0x%x bin:%d,xnor:%d\n", (unsigned int)net.input_gpu, (unsigned int)l.output_gpu, l.binary, l.xnor);
    printf("workspace:0x%x,size:%d,", (unsigned int)net.workspace, l.workspace_size);
    printf("inputsize:%d,outputSize:%d\n", net.inputs, l.outputs);
#endif
#ifdef FORWARD_CONVOLUTIONAL_LAYER_GPUHALF
    OutPutGPUMemory(net.input_gpu, net.inputs,0);
#endif 
    LAYERDATA* data = (LAYERDATA *)l.layerdata;
    CONVPROP* prop = (CONVPROP*)data->layerData;
    void* input=0;
    void* output = 0;
    if (prop->bIn32)
    {
        cuda_convert_f32_to_f16(net.input_gpu, net.inputs, publicMemory[0]);
        input = publicMemory[0];
    }
    else
    {
        input = net.input_gpu;
    }
    if (prop->bOut32)
    {
        output = publicMemory[1];
    }
    else
    {
        output = l.output_gpu;
    }
#ifdef GETDATATYPE
    float* fa, *fw;
    fa = cuda_make_array(0, net.inputs);
    fw = cuda_make_array(0, l.nweights);
    cuda_convert_f16_to_f32(publicMemory[0], net.inputs, fa);
    cuda_convert_f16_to_f32((half *)l.weights_gpu, l.nweights, fw);
    OutPutGPUMemory(fa, net.inputs, 0);
    OutPutGPUMemory(fw, l.nweights, 0);
#endif
    cudnnStatus_t stat = cudnnConvolutionForward(cudnn_handle(),
        &one,
        l.srcTensorDesc,        
        input,
        l.weightDesc,
        l.weights_gpu,
        l.convDesc,
        l.fw_algo,
        net.workspace,
        l.workspace_size,
        &zero,
        l.dstTensorDesc,        
        output); 
        checkcudnnerror(stat);
#ifdef GETDATATYPE
    /*if (GetDataType() == CUDNN_DATA_FLOAT)
    {
        OutPutGPUMemory((float *)publicMemory[1], l.outputs, 0);
        cudnnStatus_t stat = cudnnConvolutionForward(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            publicMemory[0],
            l.weightDesc,
            l.weights_gpu,
            l.convDesc,
            l.fw_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dstTensorDesc,
            l.output_gpu);
            publicMemory[0]);
        checkcudnnerror(stat);
        OutPutGPUMemory((float*)publicMemory[0], l.outputs, 0);
        stat = cudnnConvolutionForward(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            publicMemory[0],
            l.weightDesc,
            l.weights_gpu,
            l.convDesc,
            l.fw_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dstTensorDesc,
            l.output_gpu);
            publicMemory[0]);
        checkcudnnerror(stat);
        OutPutGPUMemory((float*)l.output_gpu, l.outputs, 0);
        cuda_convert_f32_to_f16((float *)publicMemory[1], l.outputs, (half*)publicMemory[0]);
        cudaError_t stats = cudaMemcpy(publicMemory[1], publicMemory[0], l.outputs * sizeof(float), cudaMemcpyDeviceToDevice);
    }*/
#endif
#ifdef TESTPROGRESS16
        if (output == l.output_gpu)
        {
            cudaMemcpy(publicMemory[1], l.output_gpu, l.outputs * sizeof(half), cudaMemcpyDeviceToDevice);            
        }
        cuda_convert_f16_to_f32((half*)publicMemory[1], l.outputs, tempBuffer);
        //OutPutGPUMemory(l.output_gpu, l.outputs, 0);
        cuda_convert_f16_to_f32((half*)l.biases_gpu, l.n, tempWeight);
        add_bias_gpu(tempBuffer, tempWeight, l.batch, l.n, l.out_w * l.out_h);
        activate_array_ongpu(tempBuffer, l.outputs * l.batch, l.activation);
        //OutPutGPUMemory(l.output_gpu, l.outputs, 0);
        cuda_convert_f32_to_f16(tempBuffer, l.outputs, publicMemory[1]);
        if (output == l.output_gpu)
        {  
            cudaMemcpy(l.output_gpu, publicMemory[1], l.outputs * sizeof(half),cudaMemcpyDeviceToDevice);
        }

#else
    add_bias_activation_half_gpu((half*)output, (half*)l.biases_gpu, l.batch, l.n, l.out_w* l.out_h,l.activation
        ,prop->bUnSupportActivate,prop->bUnSupportBias);
#endif
    if (prop->bOut32)
    {
        cuda_convert_f16_to_f32((half*)output, l.outputs, l.output_gpu);
    }
#ifdef MEMORYDEBUG
    printf("End Forword Cudnn\n");
    //if (prop->bUnSupportActivate) OutPutGPUMemory(l.output_gpu, l.outputs, 0);
#endif
    if(prop->bUnSupportBias) add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w * l.out_h);
    if(prop->bUnSupportActivate) activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation);
#ifdef MEMORYDEBUG
    //if (prop->bUnSupportActivate) OutPutGPUMemory(l.output_gpu, l.outputs, 0);
#endif
    //if(l.dot > 0) dot_error_gpu(l);
    if (l.binary || l.xnor) swap_binary(&l);
}