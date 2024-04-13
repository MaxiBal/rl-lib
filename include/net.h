#ifndef NET_H
#define NET_H

#include <stdlib.h>


float gradient(float* y);

// Activation function

float sigmoid(float);
float reLU(float);

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

/// @brief Initializes a new neuron
/// @param neuron neuron to initialize
/// @param out_weights number of neurons in the next layer
void initialize_neuron  (struct neuron_t* neuron, size_t out_weights);

/// @brief Initializes a new layer in a network
/// @param layer layer to initialize
/// @param size number of neurons in the layer
/// @param next_layer_size number of neurons in the next layer
void initialize_layer   (struct layer_t* layer, size_t size, size_t next_layer_size);

/// @brief Initializes a new neural network
/// @param network network to initialize
/// @param layer_sizes size of each layer
/// @param num_of_layers number of layers
void initialize_network (struct network_t* network, size_t* layer_sizes, size_t num_of_layers);



void free_neuron    (struct neuron_t* neuron);
void free_layer     (struct layer_t* layer);
void free_network   (struct network_t* network);


#endif