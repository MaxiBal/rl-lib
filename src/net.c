#include "net.h"

// Activation Functions

float sigmoid(float x)
{
    return 1 / (1 + expf(-x));
}

float d_sigmoid(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

float reLU(float x)
{
    if (x <= 0) return 0;
    return x;
}

float d_reLU(float x)
{
    return (float) x > 0;
}

// Neural Network Initializations, Destructors

static void initialize_neuron_weights(struct neuron_t* neuron, size_t out_weights, bool is_input)
{
    for (size_t i = 0; i < out_weights; i++)
    {
        neuron->out_weights[i] = ((float) rand()) / ((float) RAND_MAX);
        neuron->dw[i] = 0.0;
    }

    if (!is_input)
    {
        neuron->bias = ((float) rand()) / ((float) RAND_MAX);
    }
}

void initialize_neuron  (struct neuron_t* neuron, size_t out_weights, bool is_input)
{
    neuron->out_weights = (float*) malloc(out_weights * sizeof(float));
    neuron->dw          = (float*) malloc(out_weights * sizeof(float));
    neuron->out_size    = out_weights;

    initialize_neuron_weights(neuron, out_weights, is_input);
}

void initialize_layer   (struct layer_t* layer, size_t size, size_t next_layer_size, bool is_input)
{
    layer->n = size;
    layer->next_layer_size = next_layer_size;

    layer->neurons = (struct neuron_t*) malloc(size * sizeof(struct neuron_t));

    // populate neurons
    for (size_t i = 0; i < size; i++)
    {
        initialize_neuron(&layer->neurons[i], next_layer_size, is_input);
    }
}

void initialize_network (struct network_t* network, size_t* layer_sizes, size_t num_of_layers)
{
    network->n = num_of_layers;
    network->layers = (struct layer_t*) malloc(num_of_layers * sizeof(struct layer_t));
    network->learning_rate = 0.15;

    initialize_layer(&network->layers[0], layer_sizes[0], layer_sizes[1], true);

    for (size_t i = 1; i < num_of_layers - 1; i++)
    {
        initialize_layer(&network->layers[i], layer_sizes[i], layer_sizes[i + 1], false);
    }

    // initialize last layer with 0 output weights
    initialize_layer(&network->layers[num_of_layers - 1], layer_sizes[num_of_layers - 1], 0, false);
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

// Neural network functions

void feed_input(struct layer_t* input_layer, float* data)
{
    for (size_t i = 0; i < input_layer->n; i++)
    {
        input_layer->neurons[i].actv = data[i];
    }
}

void forward_propogate(struct network_t* network)
{
    // Iterate through each layer

    // accessor for layers
    struct layer_t* layers = network->layers;

    for (size_t l = 1; l < network->n; l++)
    {
        // for each neuron in layer l
        for (size_t n = 0; n < layers[l].n; n++)
        {
            layers[l].neurons[n].z = layers[l].neurons[n].bias;

            // calculate neuron z based on last layer's output from each neuron
            for (size_t i = 0; i < layers[l - 1].n; i++)
            {
                layers[l].neurons[n].z += 
                    layers[l - 1].neurons[i].actv * 
                    layers[l - 1].neurons[i].out_weights[n]; // out weight for neuron n
            }

            // use ReLU activation function for each hidden layer
            if (l < network->n - 1)
            {
                layers[l].neurons[n].actv = reLU(layers[l].neurons[n].z);
            }

            else // use sigmoid
            {
                layers[l].neurons[n].actv = sigmoid(layers[l].neurons[n].z);
            }
        }
    }
}

void back_propogate(struct network_t* network, float** labels, size_t label_n)
{
    // accessor for layers
    struct layer_t* layers = network->layers;

    // start from output layer
    backpropogate_output_layer(layers, network->n - 1, labels, label_n);

    // backpropogate to all hidden layers
    backpropogate_hidden_layers(layers, network->n - 2);
    

}

static void backpropogate_output_layer(struct layer_t* layers, const size_t output_layer_n, float** labels, size_t label_n)
{

    
    for (size_t n = 0; n < layers[output_layer_n].n; n++)
    {
        
        layers[output_layer_n].neurons[n].dz = 
            (layers[output_layer_n].neurons[n].actv - labels[label_n][n]) * d_sigmoid(layers[output_layer_n].neurons[n].actv);

        for (size_t k = 0; k < layers[output_layer_n - 1].n; k++)
        {
            layers[output_layer_n - 1].neurons[k].dw[n] = 
                layers[output_layer_n - 1].neurons[k].actv * layers[output_layer_n].neurons[n].dz;

            layers[output_layer_n - 1].neurons[k].dactv = 
                layers[output_layer_n - 1].neurons[k].out_weights[n] * layers[output_layer_n].neurons[n].dz;
        }

        layers[output_layer_n].neurons[n].dbias = layers[output_layer_n].neurons[n].dz;
    }
}

static void backpropogate_hidden_layers(struct layer_t* layers, size_t end_layer)
{
    for (size_t l = end_layer; l > 0; l--)
    {
        // for each neuron in layer
        for (size_t n = 0; n < layers[l].n; n++)
        {
            layers[l].neurons[n].dz = (layers[l].neurons[n].z >= 0) ? layers[l].neurons[n].dactv : 0;

            for (size_t k = 0; k < layers[l - 1].n; k++)
            {
                layers[l - 1].neurons[k].dw[n] = layers[l].neurons[n].dz * layers[l - 1].neurons[k].actv;

                if (l > 1)
                {
                    layers[l - 1].neurons[k].dactv = layers[l - 1].neurons[k].out_weights[n] * layers[l].neurons[n].dz;
                }
            }

            layers[l].neurons[n].dbias = layers[l].neurons[n].dz;
        }
    }
}

void update_weights(struct network_t* network)
{
    struct layer_t* layers = network->layers;
    for (size_t l = 0; l < network->n - 1; l++)
    {
        for (size_t n = 0; n < layers[l].n; n++)
        {
            for (size_t k = 0; k < layers[l+1].n; k++)
            {
                layers[l].neurons[n].out_weights[k] = 
                    layers[l].neurons[n].out_weights[k] - (network->learning_rate * layers[l].neurons[n].dw[k]);
            }

            // update bias

            layers[l].neurons[n].bias = layers[l].neurons[n].bias - (network->learning_rate * layers[l].neurons[n].dbias);
        }
    }
}

float compute_cost(struct layer_t* output_layer, float** labels, size_t n, size_t n_data)
{
    float cost = 0.0;

    for (size_t i = 0; i < n; i++)
    {
        cost += powf(labels[n_data][i] - output_layer->neurons[i].actv, 2.0) / 2;
    }

    return cost;
}

void train_model(struct network_t* network, float** training_data, float** labels, size_t data_size, size_t epochs)
{
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        for (size_t n = 0; n < data_size; n++)
        {
            feed_input(&network->layers[0], training_data[n]);
            forward_propogate(network);
            compute_cost(&network->layers[network->n - 1], labels, network->layers[network->n - 1].n, n);
            back_propogate(network, labels, n);
            update_weights(network);
        }
    }
}

void feed(struct network_t* network, float* data)
{
    feed_input(&network->layers[0], data);
    forward_propogate(network);
}