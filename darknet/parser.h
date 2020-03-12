#ifndef PARSER_H
#define PARSER_H
#include "activation_layer.h"
#include "logistic_layer.h"
#include "l2norm_layer.h"
#include "activations.h"
#include "avgpool_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "connected_layer.h"
#include "deconvolutional_layer.h"
#include "convolutional_layer.h"
#include "cost_layer.h"
#include "crnn_layer.h"
#include "crop_layer.h"
#include "detection_layer.h"
#include "dropout_layer.h"
#include "gru_layer.h"
#include "list.h"
#include "local_layer.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "yolo_layer.h"
#include "iseg_layer.h"
#include "reorg_layer.h"
#include "rnn_layer.h"
#include "route_layer.h"
#include "upsample_layer.h"
#include "shortcut_layer.h"
#include "softmax_layer.h"
#include "lstm_layer.h"
#include "utils.h"
#ifdef __cplusplus
extern "C" {
#endif
    typedef struct {
        char* type;
        list* options;
    }section;
    typedef struct size_params {
        int batch;
        int inputs;
        int h;
        int w;
        int c;
        int index;
        int time_steps;
        network *net;
    } size_params;
    LAYER_TYPE string_to_layer_type(char * type);
    void free_section(section *s);
    void parse_data(char *data, float *a, int n);
    local_layer parse_local(list *options, size_params params);
    layer parse_deconvolutional(list *options, size_params params);
    convolutional_layer parse_convolutional(list *options, size_params params);
    layer parse_crnn(list *options, size_params params);
    layer parse_rnn(list *options, size_params params);
    layer parse_gru(list *options, size_params params);
    layer parse_lstm(list *options, size_params params);
    layer parse_connected(list *options, size_params params);
    layer parse_softmax(list *options, size_params params);
    int *parse_yolo_mask(char *a, int *num);
    layer parse_yolo(list *options, size_params params);
    layer parse_iseg(list *options, size_params params);
    layer parse_region(list *options, size_params params);
    detection_layer parse_detection(list *options, size_params params);
    cost_layer parse_cost(list *options, size_params params);
    crop_layer parse_crop(list *options, size_params params);
    layer parse_reorg(list *options, size_params params);
    maxpool_layer parse_maxpool(list *options, size_params params);
    avgpool_layer parse_avgpool(list *options, size_params params);
    dropout_layer parse_dropout(list *options, size_params params);
    layer parse_normalization(list *options, size_params params);
    layer parse_batchnorm(list *options, size_params params);
    layer parse_shortcut(list *options, size_params params, network *net);
    layer parse_l2norm(list *options, size_params params);
    layer parse_logistic(list *options, size_params params);
    layer parse_activation(list *options, size_params params);
    layer parse_upsample(list *options, size_params params, network *net);
    route_layer parse_route(list *options, size_params params, network *net);
    learning_rate_policy get_policy(char *s);
    void parse_net_options(list *options, network *net);
    int is_network(section *s);
    network *parse_network_cfg(char *filename);
    list *read_cfg(char *filename);
    void save_convolutional_weights_binary(layer l, FILE *fp);
    void save_convolutional_weights(layer l, FILE *fp);
    void save_batchnorm_weights(layer l, FILE *fp);
    void save_connected_weights(layer l, FILE *fp);
    void save_weights_upto(network *net, char *filename, int cutoff);
    void transpose_matrix(float *a, int rows, int cols);
    void load_connected_weights(layer l, FILE *fp, int transpose);
    void load_batchnorm_weights(layer l, FILE *fp);
    void load_convolutional_weights_binary(layer l, FILE *fp);
    void load_convolutional_weights(layer l, FILE *fp);
    void load_weights_upto(network *net, char *filename, int start, int cutoff);
    void load_weights(network *net, char *filename);
    network *parse_network_train_cfg(char *filename);
    network NetWorkMemoryConfigerEx(size_params* paramss, section* s, list* options, list *sections,
        network* nets,node* n);
    void FillNetWorkData(network* net);
#ifdef __cplusplus
}
#endif
#endif
