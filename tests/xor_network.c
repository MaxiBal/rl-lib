#include <stdio.h>
#include <stdlib.h>

#include <net.h>

/// @brief Test the neural network using XORd data
void test_network_with_XOR_data()
{
    struct network_t net;

    const float learning_rate = 0.15;

    const size_t layers = 4;

    // 2 number input -- one output
    size_t layer_sizes[] = {2, 4, 4, 1};

    const size_t training_data_n = 4;

    float input_data[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    // rewrite using malloc

    float** xor_truth_table_inputs = (float**) malloc(training_data_n * sizeof(float*));
    for(size_t i=0; i<training_data_n; i++)
    {
        xor_truth_table_inputs[i] = (float*)malloc(layer_sizes[0] * sizeof(float));

        for (size_t c = 0; c < layer_sizes[0]; c++)
        {
            xor_truth_table_inputs[i][c] = input_data[i][c];
        }
    }

    float output_data[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    float** xor_truth_table_labels = (float**) malloc(training_data_n * sizeof(float*));
    for(size_t i=0; i<training_data_n; i++)
    {
        xor_truth_table_labels[i] = (float*)malloc(layer_sizes[layers - 1] * sizeof(float));
        for (size_t c = 0; c < layer_sizes[0]; c++)
        {
            xor_truth_table_labels[i][c] = output_data[i][c];
        }
    }


    initialize_network(&net, layer_sizes, layers);

    size_t epochs = 3000;

    train_model(&net, xor_truth_table_inputs, xor_truth_table_labels, training_data_n, 3000);

    // test truth table

    for (size_t i = 0; i < training_data_n; i++)
    {

        printf("Case: %f, %f:\n", xor_truth_table_inputs[i][0], xor_truth_table_inputs[i][1]);


        feed(&net, xor_truth_table_inputs[i]);
        
        for (size_t out = 0; out < net.layers[net.n - 1].n; out++)
        {
            printf("%ld: %d\n", out, net.layers[net.n - 1].neurons[out].actv > 0.500);
        }

    }

    free_network(&net);

    for(size_t i=0; i<training_data_n; i++)
    {
        free(xor_truth_table_inputs[i]);
        free(xor_truth_table_labels[i]);
    }

    free(xor_truth_table_inputs);
    free(xor_truth_table_labels);
}


int main()
{
    test_network_with_XOR_data();

    return 0;
}