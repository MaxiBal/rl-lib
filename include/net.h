#ifndef NET_H
#define NET_H

#include <stdbool.h>
#include <stdlib.h>
#include <math.h>


float gradient(float* y);

// Activation functions

float sigmoid(float);
float reLU(float);

// Activation function derivatives
float d_sigmoid(float);
float d_reLU(float);


// Error functions

float MSE(float* y, float* y_hat);

struct neuron_t
{
    float actv;
    float* out_weights;
    float bias;
    float z;

    float dactv;
    float* dw;
    float dbias;
    float dz;

    size_t out_size;
};

struct layer_t
{
    size_t n;
    size_t next_layer_size;
    struct neuron_t* neurons;
};

struct network_t
{
    size_t n;
    struct layer_t* layers;
};

static void initialize_neuron_weights(struct neuron_t* neuron, size_t out_weights, bool is_input);


/// @brief Initializes a new neuron
/// @param neuron neuron to initialize
/// @param out_weights number of neurons in the next layer
void initialize_neuron  (struct neuron_t* neuron, size_t out_weights, bool is_input);

/// @brief Initializes a new layer in a network
/// @param layer layer to initialize
/// @param size number of neurons in the layer
/// @param next_layer_size number of neurons in the next layer
void initialize_layer   (struct layer_t* layer, size_t size, size_t next_layer_size, bool is_input);

/// @brief Initializes a new neural network
/// @param network network to initialize
/// @param layer_sizes size of each layer
/// @param num_of_layers number of layers
void initialize_network (struct network_t* network, size_t* layer_sizes, size_t num_of_layers);

void free_neuron    (struct neuron_t* neuron);
void free_layer     (struct layer_t* layer);
void free_network   (struct network_t* network);


void feed_input(struct layer_t* input_layer, float** data, size_t n);

float compute_cost(struct layer_t* output_layer, float** labels, size_t n, size_t n_data);

void forward_propogate(struct network_t* network);

static void backpropogate_output_layer(struct layer_t* layers, const size_t output_layer_n, float** labels, size_t label_n);
static void backpropogate_hidden_layers(struct layer_t* layers, size_t end_layer);

/// @brief Back propogates labels to entire network
/// @param network network to perform back propogation on
/// @param labels output data
/// @param label_n size of ```labels```
void back_propogate(struct network_t* network, float** labels, size_t label_n);

void update_weights(struct network_t* network, float learning_rate);

#endif