#include <stdlib.h>
#include <stdio.h>

#include <net.h>

void check_for_intitialized_network()
{
    struct network_t net;
    
    const size_t layers = 3;

    size_t layer_sizes[] = {5, 8, 1};

    initialize_network(&net, layer_sizes, layers);

    printf("Initialized neural network with %ld layers.\n", layers);
    
    // print something in layers idk?

    for (size_t i = 0; i < layer_sizes[0]; i++)
    {
        printf("Bias for neuron %ld: %f \n", i+1, net.layers[0].neurons[i].bias);
    }

    free_network(&net);
}

int main()
{
    check_for_intitialized_network();

    return 0;
}