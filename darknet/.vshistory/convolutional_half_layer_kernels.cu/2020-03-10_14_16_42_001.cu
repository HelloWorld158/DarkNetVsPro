#include "pch.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"
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
void MakeHalfMaxSize(int iGiveSize,int iOutSize)
{
   size_t size[2] = { sizeof(half) * iGiveSize,iOutSize*sizeof(half)};
   for (int cnum = 0; cnum < 2; cnum++)
   {
       if (pMSize[cnum] < size[cnum])
       {
           if (publicMemory[cnum]) cuda_free_allType(publicMemory[cnum]);
           pMSize[cnum] = size[cnum];
           publicMemory[cnum]=(half *)cuda_make_short_array(pMSize[cnum]);
       }
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
void DealWeightBuffer(convolutional_layer l)
{
    //return;
#ifdef GETDATATYPE
    if (GetDataType() != CUDNN_DATA_HALF) return;
#endif   
#ifdef DEALWEIGHTBUFFER
    OutPutGPUMemory(l.weights_gpu, l.nweights, 0);
#endif
    half* halfWeights = 0;
    halfWeights=(half *)cuda_make_short_array(l.nweights);
    cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, halfWeights);
#ifdef DEALWEIGHTBUFFER
    float* fResult=0;
    check_error(cudaMalloc((void**)&fResult, l.nweights * sizeof(float)));
    cuda_convert_f16_to_f32(halfWeights, l.nweights, fResult);
    OutPutGPUMemory(fResult, l.nweights, 0);
#endif       
    //l.weights_gpu = (float*)halfWeights;
    cuda_free(l.weights_gpu);
    DecGenerateMemory(l.nweights * sizeof(float));
    l.weights_gpu = (float *)halfWeights;   
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
    cuda_convert_f32_to_f16(net.input_gpu, net.inputs, publicMemory[0]);   
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
        publicMemory[0],
        l.weightDesc,
        l.weights_gpu,
        l.convDesc,
        l.fw_algo,
        net.workspace,
        l.workspace_size,
        &zero,
        l.dstTensorDesc,        
        publicMemory[1]); 
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
    cuda_convert_f16_to_f32(publicMemory[1], l.outputs, l.output_gpu);
#ifdef FORWARD_CONVOLUTIONAL_LAYER_GPUHALF
    OutPutGPUMemory(l.output_gpu, l.outputs, 0);
   // exit(0);
#endif 
#ifdef MEMORYDEBUG
    printf("End Forword Cudnn\n");
#endif
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w * l.out_h);
    activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation);

    //if(l.dot > 0) dot_error_gpu(l);
    if (l.binary || l.xnor) swap_binary(&l);
}