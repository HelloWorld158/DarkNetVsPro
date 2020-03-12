#include "darknet.h"
#ifndef __LAYER_H__
#define __LAYER_H__
struct LAYERDATA
{
	int iNetWorkIndex;
	int* iConnectBefore;
	int* iConnectAfter;
	int iConnectBefSize;
	int iConnectAfterSize;
	network* net;
	void* layerData;
};
#endif