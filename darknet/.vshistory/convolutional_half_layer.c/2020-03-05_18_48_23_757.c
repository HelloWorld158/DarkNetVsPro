#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
cudnnDataType_t GetDataType() { return CUDNN_DATA_HALF; }
void cudnn_convolutional_setup16(layer* l)
{
    cudnnDataType_t dataype=GetDataType();
    //CONVPROP* prop=(CONVPROP *)l->layerExtraProperty;
    cudnnStatus_t stat = cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, dataype, l->batch, l->c, l->h, l->w); checkcudnnerror(stat);
    stat = cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, dataype, l->batch, l->out_c, l->out_h, l->out_w); checkcudnnerror(stat);
    /*stat = cudnnSetTensor4dDescriptor(prop->biasTensor, CUDNN_TENSOR_NCHW, dataype, l->batch, l->out_c, 1, 1);
    checkcudnnerror(stat);
    stat = cudnnSetActivationDescriptor(prop->actv, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0);
    checkcudnnerror(stat);*/
    stat = cudnnSetFilter4dDescriptor(l->weightDesc, dataype, CUDNN_TENSOR_NCHW, l->n, l->c / l->groups, l->size, l->size); checkcudnnerror(stat);
    cudnnDataType_t convType=CUDNN_DATA_FLOAT;
    if (GetConvolutionPredictMethod() == FLOAT16_TRUE_PREDICT) convType = CUDNN_DATA_HALF;
    stat = cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, convType); checkcudnnerror(stat);
    stat = cudnnSetConvolutionGroupCount(l->convDesc, l->groups); checkcudnnerror(stat);
    stat = cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
        l->srcTensorDesc,
        l->weightDesc,
        l->convDesc,
        l->dstTensorDesc,
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        2000000000,
        &l->fw_algo); checkcudnnerror(stat);      
}
static size_t get_workspace_size(layer l) {
#ifdef CUDNN
    if (gpu_index >= 0) {
        size_t most = 0;
        size_t s = 0;
        cudnnStatus_t stat = cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
            l.srcTensorDesc,
            l.weightDesc,
            l.convDesc,
            l.dstTensorDesc,
            l.fw_algo,
            &s); checkcudnnerror(stat);
        if (s > most) most = s;        
        return most;
    }
#endif
    return (size_t)l.out_h * l.out_w * l.size * l.size * l.c / l.groups * sizeof(float);
}
convolutional_layer make_convolutional_layer_CV_PREDICT_FLOAT16(CLARE_CONV_PARAM)
{
    int i;
    convolutional_layer l = { 0 };
    l.type = CONVOLUTIONAL;
    /*l.layerExtraProperty = malloc(sizeof(CONVPROP));
    CONVPROP* layerProp = (CONVPROP*)l.layerExtraProperty;
    layerProp->iConvProPertySize = sizeof(CONVPROP);*/
    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.weights = calloc(c / groups * n * size * size, sizeof(float));
    l.weight_updates = calloc(c / groups * n * size * size, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    l.nweights = c / groups * n * size * size;
    l.nbiases = n;

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2. / (size * size * c / l.groups));
    //printf("convscale %f\n", scale);
    //scale = .02;
    //for(i = 0; i < c*n*size*size; ++i) l.weights[i] = scale*rand_uniform(-1, 1);
    for (i = 0; i < l.nweights; ++i) l.weights[i] = scale * rand_normal();
    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    MakeHalfMaxSize(l.inputs, l.outputs);

    l.output = 0;/////CV
    l.delta = 0;/////CV

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if (binary) {
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.cweights = calloc(l.nweights, sizeof(char));
        l.scales = calloc(n, sizeof(float));
    }
    if (xnor) {
        l.binary_weights = calloc(l.nweights, sizeof(float));
        l.binary_input = calloc(l.inputs * l.batch, sizeof(float));
    }

    if (batch_normalize) {
        l.scales = calloc(n, sizeof(float));
        l.scale_updates = calloc(n, sizeof(float));
        for (i = 0; i < n; ++i) {
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.mean_delta = calloc(n, sizeof(float));
        l.variance_delta = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
        l.x = 0;//calloc(l.batch*l.outputs, sizeof(float));
        l.x_norm = 0;//calloc(l.batch*l.outputs, sizeof(float));
    }
    if (adam) {
        l.m = calloc(l.nweights, sizeof(float));
        l.v = calloc(l.nweights, sizeof(float));
        l.bias_m = calloc(n, sizeof(float));
        l.scale_m = calloc(n, sizeof(float));
        l.bias_v = calloc(n, sizeof(float));
        l.scale_v = calloc(n, sizeof(float));
    }

#ifdef GPU
    l.forward_gpu = forward_convolutional_layer_gpu_predict_Float16;
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if (gpu_index >= 0) {
        if (adam) {
            l.m_gpu = cuda_make_array(l.m, l.nweights);
            l.v_gpu = cuda_make_array(l.v, l.nweights);
            l.bias_m_gpu = cuda_make_array(l.bias_m, n);
            l.bias_v_gpu = cuda_make_array(l.bias_v, n);
            l.scale_m_gpu = cuda_make_array(l.scale_m, n);
            l.scale_v_gpu = cuda_make_array(l.scale_v, n);
        }

        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = 0;//CV// cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = 0;//CV//cuda_make_array(l.bias_updates, n);

        l.delta_gpu = 0;////CV
        l.output_gpu = 0;////CV

        if (binary) {
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
        }
        if (xnor) {
            l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.binary_input_gpu = cuda_make_array(0, l.inputs * l.batch);
        }

        if (batch_normalize) {
            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.variance, n);

            l.mean_delta_gpu = 0;//CV// cuda_make_array(l.mean, n);
            l.variance_delta_gpu = 0;//CV//cuda_make_array(l.variance, n);

            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = 0;//CV//cuda_make_array(l.scale_updates, n);

            l.x_gpu = 0;//CV//cuda_make_array(l.output, l.batch*out_h*out_w*n);
            l.x_norm_gpu = 0;//CV//cuda_make_array(l.output, l.batch*out_h*out_w*n);
        }
#ifdef CUDNN
        cudnnStatus_t stat;//=cudnnCreateTensorDescriptor(&l.normTensorDesc); checkcudnnerror(stat);
        stat = cudnnCreateTensorDescriptor(&l.srcTensorDesc); checkcudnnerror(stat);
        stat = cudnnCreateTensorDescriptor(&l.dstTensorDesc); checkcudnnerror(stat);
        stat = cudnnCreateFilterDescriptor(&l.weightDesc); checkcudnnerror(stat);
        stat = cudnnCreateConvolutionDescriptor(&l.convDesc); checkcudnnerror(stat);
        //stat = cudnnCreateTensorDescriptor(&(layerProp->biasTensor)); checkcudnnerror(stat);
        //stat = cudnnCreateActivationDescriptor(&(layerProp->actv)); checkcudnnerror(stat);

        cudnn_convolutional_setup16(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size * l.size * l.c / l.groups * l.out_h * l.out_w) / 1000000000.);

    return l;
}