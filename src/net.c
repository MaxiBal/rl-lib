#include "net.h"



void initialize_neuron  (struct neuron_t* neuron, size_t out_weights)
{
    neuron->out_weights = (float*) malloc(out_weights * sizeof(float));
    neuron->dw          = (float*) malloc(out_weights * sizeof(float));
    neuron->out_size    = out_weights;
}

void initialize_layer   (struct layer_t* layer, size_t size, size_t next_layer_size)
{
    layer->n = size;
    layer->next_layer_size = next_layer_size;

    layer->neurons = (struct neuron_t*) malloc(size * sizeof(struct neuron_t));

    // populate neurons
    for (size_t i = 0; i < size; i++)
    {
        initialize_neuron(&layer->neurons[i], next_layer_size);
    }
}

void initialize_network (struct network_t* network, size_t* layer_sizes, size_t num_of_layers)
{
    network->n = num_of_layers;
    network->layers = (struct layer_t*) malloc(num_of_layers * sizeof(struct layer_t));

    for (size_t i = 0; i < num_of_layers - 1; i++)
    {
        initialize_layer(&network->layers[i], layer_sizes[i], layer_sizes[i + 1]);
    }

    // initialize last layer with 0 output weights
    initialize_layer(&network->layers[num_of_layers - 1], layer_sizes[num_of_layers - 1], 0);
}



void free_neuron    (struct neuron_t* neuron)
{
    free(neuron->out_weights);
    free(neuron->dw);
}

void free_layer     (struct layer_t* layer)
{
    for (size_t i = 0; i < layer->n; i++)
    {
        free_neuron(&layer->neurons[i]);
    }
}

void free_network   (struct network_t* network)
{
    for (size_t i = 0; i < network->n; i++)
    {
        free_layer(&network->layers[i]);
    }
}