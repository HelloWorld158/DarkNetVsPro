#ifndef MEMORYMANAGER
#define MEMORYMANAGER
#include "pch.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>


#include "parser.h"
#include "option_list.h"

#include <vector>



using namespace std;
vector<float*> cpuBuffer;
vector<float*> gpuBuffer;
vector<int> memCode;
vector<int> iMemCodeMxSize;
vector<int> iLayerUseSize;
vector< vector<int> > fulgraph, fulbgraph;
vector<LAYERDATA> layerDatas;

void GetLayersFromRoute(list* options,vector< vector<int> >& backGraph)
{    
    char* l = option_find(options, "layers");
    if (!l) error("Route Layer must specify input layers");
    int len = strlen(l);
    int n = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++n;
    }
    vector<int> iGetLayer(n);
    for (i = 0; i < n; ++i)
    {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        if (index < 0) index = backGraph.size() + index;
        iGetLayer[i] = index;
    }
    backGraph.push_back(iGetLayer);
}
void GetLayersFromShortCut(list* options, vector< vector<int> >& backGraph)
{
    char* l = option_find(options, "from");
    int len = strlen(l);
    if (!l) error("Route Layer must specify input layers: from = ...");
    int n = 1;
    int i;
    for (i = 0; i < len; ++i) {
        if (l[i] == ',') ++n;
    }
    vector<int> iGetLayer(n);
    for (i = 0; i < n; ++i) {
        int index = atoi(l);
        l = strchr(l, ',') + 1;
        if (index < 0) index = backGraph.size() + index;
        iGetLayer[i] = index;
    }
    backGraph.push_back(iGetLayer);
}
void GetGraph(vector< vector<int> >& graph,vector< vector<int> >& backGraph,node* n,vector<LAYER_TYPE>& layertype)
{
    graph.clear();
    backGraph.clear();
    vector<int> empty;
    while (n)
    {
        ::section* s = (::section*)n->val;
        list* options = s->options;
        layer l = { 0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        switch (lt)
        {
        default:
        case CONVOLUTIONAL:            
        case YOLO:
        case MAXPOOL:
        case UPSAMPLE:
            backGraph.push_back(empty);
            break;
        case ROUTE:
            GetLayersFromRoute(options, backGraph);
            break;        
        case SHORTCUT:
            GetLayersFromShortCut(options, backGraph);
            break;        
        }
        layertype.push_back(lt);
        n = n->next;
    }
    graph.resize(backGraph.size());
    for (int i = 0; i < backGraph.size(); i++)
    {
        for (int j = 0; j < backGraph[i].size(); j++)
        {
            graph[backGraph[i][j]].push_back(i);
        }
    }
}

convolutional_layer GenerateConvLayer(list* options, size_params& params,int& layersize)
{
    int n = option_find_int(options, "filters",1);
    int size = option_find_int(options, "size",1);
    int stride = option_find_int(options, "stride",1);
    int pad = option_find_int_quiet(options, "pad",0);
    int padding = option_find_int_quiet(options, "padding",0);
    int groups = option_find_int_quiet(options, "groups", 1);
    if(pad) padding = size/2;

    char *activation_s = option_find_str(options, "activation", "logistic");
    ACTIVATION activation = get_activation(activation_s);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before convolutional layer must output image.");
    int batch_normalize = option_find_int_quiet(options, "batch_normalize", 0);
    int binary = option_find_int_quiet(options, "binary", 0);
    int xnor = option_find_int_quiet(options, "xnor", 0);

    convolutional_layer layer = make_convolutional_layer_CV_METHOD(GetConvolutionPredictMethod(),batch,h,w,c,n,groups,size,stride,padding
        ,activation, batch_normalize, binary, xnor, params.net->adam);
    layersize = batch * 1 * layer.outputs;
    layer.flipped = option_find_int_quiet(options, "flipped", 0);
    layer.dot = option_find_float_quiet(options, "dot", 0);

    return layer;
}
extern "C" layer make_yolo_layer_CV(int batch, int w, int h, int n
    , int total, int* mask, int classes);
layer GenerateYoloLayer(list* options, size_params params,int& layersize)
{
    int classes = option_find_int(options, "classes", 20);
    int total = option_find_int(options, "num", 1);
    int num = total;

    char *a = option_find_str(options, "mask", 0);
    int *mask = parse_yolo_mask(a, &num);
    layer l = make_yolo_layer_CV(params.batch, params.w, params.h, num, total, mask, classes);
    //assert(l.outputs == params.inputs);
    layersize = l.outputs * params.batch;

    l.max_boxes = option_find_int_quiet(options, "max",90);
    l.jitter = option_find_float(options, "jitter", .2);

    l.ignore_thresh = option_find_float(options, "ignore_thresh", .5);
    l.truth_thresh = option_find_float(options, "truth_thresh", 1);
    l.random = option_find_int_quiet(options, "random", 0);

    char *map_file = option_find_str(options, "map", 0);
    if (map_file) l.map = read_map(map_file);

    a = option_find_str(options, "anchors", 0);
    if(a){
        int len = strlen(a);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (a[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(a);
            l.biases[i] = bias;
            a = strchr(a, ',')+1;
        }
    }
    return l;
}
extern "C" maxpool_layer make_maxpool_layer_CV(int batch, int h, int w, int c, 
int size, int stride, int padding);
maxpool_layer GenerateMaxPoolLayer(list* options, size_params params,int& layersize)
{
    int stride = option_find_int(options, "stride",1);
    int size = option_find_int(options, "size",stride);
    int padding = option_find_int_quiet(options, "padding", size-1);

    int batch,h,w,c;
    h = params.h;
    w = params.w;
    c = params.c;
    batch=params.batch;
    if(!(h && w && c)) error("Layer before maxpool layer must output image.");

    maxpool_layer layer = make_maxpool_layer_CV(batch,h,w,c,size,stride,padding);
    layersize = batch * layer.outputs;
    return layer;
}
extern "C" route_layer make_route_layer_CV(int batch, int n, int *input_layers, int *input_sizes);
route_layer GenerateRouteLayer(list* options, size_params params,network* net,int& layersize)
{
    char *l = option_find(options, "layers");
    int len = strlen(l);
    if(!l) error("Route Layer must specify input layers");
    int n = 1;
    int i;
    for(i = 0; i < len; ++i){
        if (l[i] == ',') ++n;
    }

    int *layers = (int *)calloc(n, sizeof(int));
    int *sizes = (int *)calloc(n, sizeof(int));
    for(i = 0; i < n; ++i){
        int index = atoi(l);
        l = strchr(l, ',')+1;
        if(index < 0) index = params.index + index;
        layers[i] = index;
        sizes[i] = net->layers[index].outputs;
#ifdef MEMORYDEBUG
        printf("index:%d,outputs:%d,h:%d,w:%d,c:%d\n", layers[i], sizes[i], net->layers[index].h, net->layers[index].w, net->layers[index].c);
#endif
    }
    int batch = params.batch;

    route_layer layer = make_route_layer_CV(batch, n, layers, sizes);
    layersize = layer.outputs * batch;
    convolutional_layer first = net->layers[layers[0]];
    layer.out_w = first.out_w;
    layer.out_h = first.out_h;
    layer.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            layer.out_c += next.out_c;
        }else{
            layer.out_h = layer.out_w = layer.out_c = 0;
        }
    }

    return layer;
}
extern "C" layer make_upsample_layer_CV(int batch, int w, int h, int c, int stride);
layer GenerateUpSampleLayer(list* options, size_params params, network& net,int& layersize)
{
    int stride = option_find_int(options, "stride", 2);
    layer l = make_upsample_layer_CV(params.batch, params.w, params.h, params.c, stride);
    l.scale = option_find_float_quiet(options, "scale", 1);
    layersize = params.batch * l.outputs;
    return l;
}
extern "C"  layer make_shortcut_layer_CV(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
layer GenerateShortCut(list* options, size_params params, network& net,int &layersize)
{
    char *l = option_find(options, "from");
    int index = atoi(l);
    if(index < 0) index = params.index + index;

    int batch = params.batch;
    layer from = net.layers[index];

    layer s = make_shortcut_layer_CV(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);
    layersize = params.batch * s.outputs;
    char *activation_s = option_find_str(options, "activation", "linear");
    ACTIVATION activation = get_activation(activation_s);
    s.activation = activation;
    s.alpha = option_find_float_quiet(options, "alpha", 1);
    s.beta = option_find_float_quiet(options, "beta", 1);
    return s;
}



void GenerateNetWork(size_params& params, node* n, network& net, int& count
    , size_t& workspace_size, size_t& max_inputs, size_t& max_outputs, int& avg_outputs
    , float& bflops, vector<int> & iSize)
{
    while (n)
    {
        params.index = count;
        fprintf(stderr, "%4d ", count);
        ::section* s = (::section*)n->val;
        list* options = s->options;
        layer l = { 0 };
        LAYER_TYPE lt = string_to_layer_type(s->type);
        int iCurLayerSize = -1;
        if (lt == CONVOLUTIONAL) {
            //l = parse_convolutional(options, params);
            l = GenerateConvLayer(options, params, iCurLayerSize);
        }
        else if(lt == DECONVOLUTIONAL){
            l = parse_deconvolutional(options, params);
        }
        else if (lt == LOCAL) {
            l = parse_local(options, params);
        }
        else if (lt == ACTIVE) {
            l = parse_activation(options, params);
        }
        else if(lt == LOGXENT){
            l = parse_logistic(options, params);
        }
        else if(lt == L2NORM){
            l = parse_l2norm(options, params);
        }
        else if (lt == RNN) {
            l = parse_rnn(options, params);
        }
        else if (lt == GRU) {
            l = parse_gru(options, params);
        }
        else if (lt == LSTM) {
            l = parse_lstm(options, params);
        }
        else if (lt == CRNN) {
            l = parse_crnn(options, params);
        }
        else if (lt == CONNECTED) {
            l = parse_connected(options, params);
        }
        else if (lt == CROP) {
            l = parse_crop(options, params);
        }
        else if (lt == COST) {
            l = parse_cost(options, params);
            
        }
        else if (lt == REGION) {
            l = parse_region(options, params);
            
        }
        else if (lt == YOLO) {
            l = GenerateYoloLayer(options, params,iCurLayerSize);
            
        }
        else if(lt == ISEG){
            l = parse_iseg(options, params);
        }
        else if (lt == DETECTION) {
            l = parse_detection(options, params);
        }
        else if (lt == SOFTMAX) {
            l = parse_softmax(options, params);
            net.hierarchy = l.softmax_tree;            
        }
        else if (lt == NORMALIZATION) {
            l = parse_normalization(options, params);
        }
        else if (lt == BATCHNORM) {
            l = parse_batchnorm(options, params);
        }
        else if (lt == MAXPOOL) {
            l = GenerateMaxPoolLayer(options,params,iCurLayerSize);
        }
        else if (lt == AVGPOOL) {
            l = parse_avgpool(options, params);
        }
        else if (lt == REORG) {
            l = parse_reorg(options, params);
        }
        else if (lt == AVGPOOL) {
            l = parse_avgpool(options, params);
        }
        else if (lt == ROUTE) {
            //l = parse_route(options, params);
            l = GenerateRouteLayer(options, params,&net, iCurLayerSize);
            //int k;
           // for (k = 0; k < l.n; ++k) {
           //     net.layers[l.input_layers[k]].use_bin_output = 0;
           //     net.layers[l.input_layers[k]].keep_delta_gpu = 1;
            //}
        }
        else if (lt == UPSAMPLE) {
            //l = parse_upsample(options, params, net);
            l = GenerateUpSampleLayer(options, params, net, iCurLayerSize);
        }
        else if (lt == SHORTCUT)
        {
            l = GenerateShortCut(options, params, net,iCurLayerSize);
            
        }        
        else if (lt == DROPOUT) {
            l = parse_dropout(options, params);
            l.output = net.layers[count - 1].output;
            l.delta = net.layers[count - 1].delta;
#ifdef GPU
            l.output_gpu = net.layers[count - 1].output_gpu;
            l.delta_gpu = net.layers[count - 1].delta_gpu;            
#endif
        }
        else {
            fprintf(stderr, "Type not recognized: %s\n", s->type);
        }
        iSize.push_back(iCurLayerSize);


        l.clip = net.clip;
        l.truth = option_find_int_quiet(options, "truth", 0);
        l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
        l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
        l.dontsave = option_find_int_quiet(options, "dontsave", 0);
        l.dontload = option_find_int_quiet(options, "dontload", 0);
        l.numload = option_find_int_quiet(options, "numload", 0);
        l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
        l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
        l.smooth = option_find_float_quiet(options, "smooth", 0);
        option_unused(options);
        net.layers[count] = l;
        if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
        free_section(s);
        n = n->next;
        ++count;
        if(n){
            params.h = l.out_h;
            params.w = l.out_w;
            params.c = l.out_c;
            params.inputs = l.outputs;
        }
    }
}
void ArrangeMemory(network& net, int layer,vector<int> & allSize)
{
    net.layers[layer].output = (float *)malloc(allSize[layer]*sizeof(float));
#ifdef GPU
    net.layers[layer].output_gpu = cuda_make_array(net.layers[layer].output, allSize[layer]);
#endif
}
int GiveOtherMemory(vector<int>& iMem)
{
    if (iMem.size() < 2)
    {
        iMem.push_back(0);
        return 2;
    }
    for (int cnum = 2; cnum < iMem.size(); cnum++)
    {
        if (iMem[cnum] < 0) return cnum;
    }
    iMem.push_back(-1);
    return iMem.size() - 1;
}
int EncodeMemory(vector<int> & memCode, vector< vector<int> >& graph, vector< vector<int> >& bgraph,network& net)
{
    vector<int> iMem(2);
    memCode.resize(graph.size());
    vector<int> memUseCount(graph.size());
    memset(&memUseCount[0],0, graph.size() * sizeof(int));
    for (int cnum = 0; cnum < graph.size(); cnum++)
    {
        memCode[cnum] = cnum % 2;
        if (graph[cnum].size() > 0)
        {
            int iGetMem = GiveOtherMemory(iMem);
            iMem[iGetMem] = cnum;
            memUseCount[cnum] += graph[cnum].size();
            if (memUseCount[cnum] == 0)
            {
                iMem[iGetMem] = -1;
            }
            memCode[cnum] = iGetMem;
        }
        if (bgraph[cnum].size() > 0)
        {
            for (int rnum = 0; rnum < bgraph[cnum].size(); rnum++)
            {
                memUseCount[bgraph[cnum][rnum]] -= 1;
                if (memUseCount[bgraph[cnum][rnum]] == 0)
                {
                    iMem[memCode[bgraph[cnum][rnum]]] = -1;
                }
            }
        }
        /*printf("Layer:%d type:%d mem:%d\n",cnum ,net.layers[cnum].type,memCode[cnum]);
        printf("sep:");
        for (int rnum = 0; rnum < graph[cnum].size(); rnum++)
        {
            printf("%d ", graph[cnum][rnum]);
        }
        printf("\n");
        printf("MemUse:");
        for (int rnum = 0; rnum < iMem.size(); rnum++)
        {
            printf("%d ", iMem[rnum]);
        }
        printf("\n");*/
    }   
    return iMem.size();
}
/*void ReArrangeShortCutLayer(network& net)
{
    vector<float*> fBuffer,fGPUBuffer;
    for (int i = 0; i < net.n; i++)
    {
        fBuffer.clear();
        fGPUBuffer.clear();
        if (net.layers[i].type != SHORTCUT) continue;
        fBuffer.resize(net.layers[i].n);
        fGPUBuffer.resize(net.layers[i].n);
        for (int j = 0; j < net.layers[i].n; j++)
        {
            fBuffer[j]=net.layers[net.layers[i].input_layers[j]].output;
#ifdef GPU
            fGPUBuffer[j] = net.layers[net.layers[i].input_layers[j]].output_gpu;
#endif
        }
#ifdef GPU
        net.layers[i].layers_output_gpu = (float**)cuda_make_array_pointers((void**)&fGPUBuffer[0], fGPUBuffer.size());
#endif
    }
}*/
void RequestFullGraph(vector< vector<int> >& graph, vector< vector<int> >& bgraph, vector< vector<int> >& fullGraph,
    vector< vector<int> >& fullbGraph,vector<LAYER_TYPE>& layerType)
{
    fullGraph.resize(graph.size());
    for (int cnum = 0; cnum < graph.size() - 1; cnum++)
    {
        if (layerType[cnum] != ROUTE)
            fullGraph[cnum].push_back(cnum + 1);
        for (int rnum = 0; rnum < graph[cnum].size(); rnum++)
        {
            fullGraph[cnum].push_back(graph[cnum][rnum]);
        }
    }
    fullbGraph.resize(graph.size());
    for (int cnum = 0; cnum < fullGraph.size(); cnum++)
    {
        for (int rnum = 0; rnum < fullGraph[cnum].size(); rnum++)
        {
            fullbGraph[fullGraph[cnum][rnum]].push_back(cnum);
        }
    }
}
void RequestFullGraph(vector< vector<int> >& graph, vector< vector<int> >& bgraph, vector< vector<int> >& fullGraph,vector< vector<int> >& fullbGraph,network& net)
{
    fullGraph.resize(graph.size());
    for (int cnum = 0; cnum < graph.size()-1; cnum++)
    {
        if (net.layers[cnum+1].type != ROUTE)
            fullGraph[cnum].push_back(cnum + 1);
        for (int rnum = 0; rnum < graph[cnum].size(); rnum++)
        {
            fullGraph[cnum].push_back(graph[cnum][rnum]);
        }
    }
    fullbGraph.resize(graph.size());
    for (int cnum = 0; cnum < fullGraph.size(); cnum++)
    {
        for (int rnum = 0; rnum < fullGraph[cnum].size(); rnum++)
        {
            fullbGraph[fullGraph[cnum][rnum]].push_back(cnum);
        }
    }
}
void GiveMemoryFromSize(network& net,vector<int>& allSize)
{
    for (int cnum = 0; cnum < fulgraph.size(); cnum++)
    {
        if (fulgraph[cnum].size() == 0)
        {
            memCode[cnum] = -1;
        }
    }
    int iMemsize = iMemCodeMxSize.size();
    vector<int>& iMaxSize = iMemCodeMxSize;
    for (int cnum = 0; cnum < iMemsize; cnum++) iMaxSize[cnum] = -1;
    for (int cnum = 0; cnum < fulgraph.size(); cnum++)
    {
        //printf("%d\n", cnum);
        if (allSize[cnum] == -1 || memCode[cnum] < 0)
        {
            if (allSize[cnum] > 0) ArrangeMemory(net, cnum, allSize);
        }
        else
        {
            iMaxSize[memCode[cnum]] = iMaxSize[memCode[cnum]] > allSize[cnum] ? iMaxSize[memCode[cnum]] : allSize[cnum];
        }
    }
    for (int cnum = 0; cnum < iMaxSize.size(); cnum++)
    {
        float* cpuout = (float*)malloc(iMaxSize[cnum] * sizeof(float));
        cpuBuffer.push_back(cpuout);
#ifdef GPU
#ifdef MEMORYDEBUG
        printf("%d ", cnum);
#endif
        float* gpuout = cuda_make_array(0, iMaxSize[cnum]);
#ifdef MEMORYDEBUG
        printf("0x%x\n", (unsigned int)gpuout);
#endif
        gpuBuffer.push_back(gpuout);
#endif
    }
    int iLoop = 0;
    for (int cnum = 0; cnum < memCode.size(); cnum++)
    {
        if (allSize[cnum] == -1 || memCode[cnum] < 0)
        {
            continue;
        }
        net.layers[cnum].output = cpuBuffer[memCode[cnum]];
#ifdef GPU
        net.layers[cnum].output_gpu = gpuBuffer[memCode[cnum]];
#endif
    }
}
void ArrangeLayer(size_params& params,node* n,network& net,int& count
 ,size_t& workspace_size,size_t& max_inputs ,size_t& max_outputs, int& avg_outputs
    ,float& bflops)
{
    vector< vector<int> > graph,bgraph;
    vector<LAYER_TYPE> layerType;
    GetGraph(graph, bgraph, n,layerType);   
   
    vector<int> allSize;
    GenerateNetWork(params, n, net, count, workspace_size, max_inputs,
        max_outputs, avg_outputs, bflops, allSize);
    //assert(graph.size() == net.n);    
    memCode.resize(graph.size());
    int iMemsize=EncodeMemory(memCode, graph, bgraph,net);
    int iSize = -1;
    fulgraph.clear();
    fulbgraph.clear();
    RequestFullGraph(graph, bgraph, fulgraph,fulbgraph,net);
    //GenerateLayerData(net, fulgraph, fulbgraph);
    iMemCodeMxSize.resize(iMemsize);
    GiveMemoryFromSize(net, allSize);
    allSize.swap(iLayerUseSize);
    //ReArrangeShortCutLayer(net);
}
void FreeMenageMemory()
{
    for (int cnum = 0; cnum < cpuBuffer.size(); cnum++)
    {
        free(cpuBuffer[cnum]);
#ifdef GPU
        cuda_free(gpuBuffer[cnum]);
#endif
    }
}
network NetWorkMemoryConfigerEx(size_params* paramss, section* s, list* options, list *sections,
    network* nets,node* n)
{
    network& net = *nets;
    size_params& params = *paramss;
    int avg_outputs = 0;
    float bflops = 0;
    size_t workspace_size = 0;
    size_t max_inputs = 0;
    size_t max_outputs = 0;
    n = n->next;
    net.train = 0;
    int count = 0;
    free_section(s);
    fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
    ArrangeLayer(params, n, net, count, workspace_size
        , max_inputs, max_outputs, avg_outputs, bflops);
    free_list(sections);
    layer out = get_network_output_layer(&net);
    net.outputs = out.outputs;
    net.truths = out.outputs;
    if(net.layers[net.n-1].truths) net.truths = net.layers[net.n-1].truths;
    net.output = out.output;
    net.input = (float *)calloc(net.inputs*net.batch, sizeof(float));
    net.truth = 0;//calloc(net->truths*net->batch, sizeof(float));
#ifdef GPU
    net.output_gpu = out.output_gpu;
    net.input_gpu = cuda_make_array(net.input, net.inputs*net.batch);
    net.truth_gpu = 0;//cuda_make_array(net->truth, net->truths*net->batch);
#endif
    if(workspace_size){
        //printf("%ld\n", workspace_size);
#ifdef GPU
        if(gpu_index >= 0){
            net.workspace = (float *)cuda_make_byte_array(workspace_size);
        }else {
            net.workspace = (float *)calloc(1, workspace_size);
        }
#else
        net.workspace = (float *)calloc(1, workspace_size);
#endif
    }
    return net;
}
int DealNetLayer(network *net)
{
    int bRetFlag=0;
    int bRet = DealConvlutionFinalStep(net, &iLayerUseSize[0]);
    bRetFlag = bRet>bRetFlag? bRet:bRetFlag;
    for (int cnum = 0; cnum < fulgraph.size(); cnum++)
    {
        if (net->layers[cnum].type != CONVOLUTIONAL) continue;
        LAYERDATA* data =(LAYERDATA *) net->layers[cnum].layerdata;
        CONVPROP* prop = (CONVPROP*)data->layerData;
        printf("%dconv:%d   %d   %d %d\n", cnum,fulbgraph[cnum].size(), fulgraph[cnum].size(),  prop->bIn32, prop->bOut32);
    }
    return bRetFlag;
}
void FillNetWorkData(network* net)
{
    layerDatas.clear();
    layerDatas.resize(net->n);
    for (int i = 0; i < net->n; i++)
    {
        layerDatas[i].iNetWorkIndex = i;
        layerDatas[i].net = net;
        layerDatas[i].layerData = net->layers[i].layerdata;
        net->layers[i].layerdata = &layerDatas[i];
        if (fulbgraph[i].size())
        {
            layerDatas[i].iConnectBefore = &fulbgraph[i][0];             
        }
        layerDatas[i].iConnectBefSize = fulbgraph[i].size();
        if (fulgraph[i].size())
        {
            layerDatas[i].iConnectAfter = &fulgraph[i][0];
        }
        layerDatas[i].iConnectAfterSize = fulgraph[i].size();
    }
    vector<int> allSize;
    int bRefresh = DealNetLayer(net);
    if (bRefresh)
    {
        for (int cnum = 0; cnum <cpuBuffer.size(); cnum++)
        {
            free(cpuBuffer[cnum]);
#ifdef GPU
            cuda_free(gpuBuffer[cnum]);
#endif
        }
        cpuBuffer.clear();
        gpuBuffer.clear();
        GiveMemoryFromSize(*net, iLayerUseSize);
    }
}
#endif