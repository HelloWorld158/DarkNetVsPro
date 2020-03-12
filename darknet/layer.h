#include "darknet.h"
#ifndef __LAYER_H__
#define __LAYER_H__
typedef struct 
{
	int iNetWorkIndex;
	int* iConnectBefore;
	int* iConnectAfter;
	int iConnectBefSize;
	int iConnectAfterSize;
	network* net;
	void* layerData;
}LAYERDATA;
#endif