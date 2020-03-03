#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"
//#define CONVOLUTIONAL_LAYER_DEBUG
#ifdef CONVOLUTIONAL_LAYER_DEBUG
#define FORWARD_CONVOLUTIONAL_LAYER_GPU
#define FORWARD_CONVOLUTIONAL_LAYER_GPUHALF
#define GETDATATYPE
//#define DEALWEIGHTBUFFER
#endif
	typedef layer convolutional_layer;
	///注意添加方法的时候要同时添加Train或者predict，如不满足要求，
	//请修改CONVOLUTION_METHOD GetConvolutionTrainMethod();CONVOLUTION_METHOD GetConvolutionPredictMethod();
	//这俩个函数
	typedef enum 
	{
		FLOAT16_PREDICT,
		FLOAT16_TRUE_PREDICT,
		FLOAT32_PREDICT,
		FLOAT16_TRAIN,
		FLOAT16_TRUE_TRAIN,
		FLOAT32_TRAIN,		
		END_CONV_METHOD
	} CONVOLUTION_METHOD;
#ifdef GPU
	void forward_convolutional_layer(const convolutional_layer layer, network net);
	void backward_convolutional_layer(convolutional_layer layer, network net);
	void update_convolutional_layer(convolutional_layer layer, update_args a);



	void forward_convolutional_layer_gpu(convolutional_layer layer, network net);
	void forward_convolutional_layer_gpu_predict_Float16(convolutional_layer layer, network net);
	void backward_convolutional_layer_gpu(convolutional_layer layer, network net);
	void update_convolutional_layer_gpu(convolutional_layer layer, update_args a);



	void push_convolutional_layer_CV(convolutional_layer layer);
	void push_convolutional_layer(convolutional_layer layer);
	void pull_convolutional_layer(convolutional_layer layer);



	void add_bias_gpu(float* output, float* biases, int batch, int n, int size);
	void backward_bias_gpu(float* bias_updates, float* delta, int batch, int n, int size);
	void adam_update_gpu(float* w, float* d, float* m, float* v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
	void binarize_weights_gpu(float* weights, int n, int size, float* binary);
	void binarize_gpu(float* x, int n, float* binary);
#ifdef CUDNN
	void cudnn_convolutional_setup(layer* l);
#endif
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define FILL_CONV_PARAM batch, h, w, c, n, groups, size, stride, padding, activation,\
 batch_normalize, binary, xnor, adam
#define CLARE_CONV_PARAM  int batch, int h, int w, int c, int n, int groups\
	, int size, int stride,int padding, ACTIVATION activation, int batch_normalize,\
	int binary, int xnor, int adam
	convolutional_layer make_convolutional_layer_CV_METHOD(CONVOLUTION_METHOD method,CLARE_CONV_PARAM);
	convolutional_layer make_convolutional_layer_CV_PREDICT_FLOAT32(CLARE_CONV_PARAM);
	convolutional_layer make_convolutional_layer(CLARE_CONV_PARAM);
	convolutional_layer make_convolutional_layer_CV_PREDICT_FLOAT16(CLARE_CONV_PARAM);
	void MakeHalfMaxSize(int iGiveSize, int iOutSize);

	void resize_convolutional_layer(convolutional_layer* layer, int w, int h);
	
	image* visualize_convolutional_layer(convolutional_layer layer, char* window, image* prev_weights);
	void binarize_weights(float* weights, int n, int size, float* binary);
	void swap_binary(convolutional_layer* l);
	void binarize_weights2(float* weights, int n, int size, char* binary, float* scales);	

	void add_bias(float* output, float* biases, int batch, int n, int size);
	void backward_bias(float* bias_updates, float* delta, int batch, int n, int size);

	image get_convolutional_image(convolutional_layer layer);
	image get_convolutional_delta(convolutional_layer layer);
	image get_convolutional_weight(convolutional_layer layer, int i);

	int convolutional_out_height(convolutional_layer layer);
	int convolutional_out_width(convolutional_layer layer);
	CONVOLUTION_METHOD GetConvolutionTrainMethod();
	CONVOLUTION_METHOD GetConvolutionPredictMethod();
	void SetConvolutionTrainMethod(CONVOLUTION_METHOD method);
	void SetConvolutionPredictMethod(CONVOLUTION_METHOD method);
	void OutPutGPUMemory(float* data, int iSize, char* txt);

	void DealGPUArrayBuffer_CV(convolutional_layer l);
	void DealWeightBuffer(convolutional_layer l);
#ifdef __cplusplus
}
#endif
#endif

