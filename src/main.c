#include <stdio.h>
#include <math.h>
#include <malloc.h>

#include <net.h>

typedef struct {

    float a, b;

} RegressionLine;



void LSRL(RegressionLine* line, float* x, float* y, size_t n)
{

    float x_bar, y_bar;

    float x_sum = 0, y_sum = 0;

    for (int i = 0; i < n; i++)
    {
        x_sum += x[i];
        y_sum += y[i];
    }

    x_bar = x_sum / n;
    y_bar = y_sum / n;

    float x_y_grad = 0;
    float mse = 0;

    for (int i = 0; i < n; i++)
    {
        x_y_grad += (x[i] - x_bar) * (y[i] - y_bar);
        mse += pow(x[i] - x_bar, 2.0);
    }

    line->b = x_y_grad / mse;
    line->a = y_bar - (line->b * x_bar);
}

float extrapolate(RegressionLine* line, float x)
{
    return line->b * x + line->a;
}

int LSRL_example()
{
    RegressionLine line;

    size_t n = 4096;

    float* x = (float*) malloc(sizeof(float) * n);

    if (x == NULL)
    {
        printf("Insufficient memory available.\n");
        return 1;
    }

    float* y = (float*) malloc(sizeof(float) * n);
    if (y == NULL)
    {
        printf("Insufficient memory available.\n");
        return 1;
    }

    // using line y = 2x + 3

    for (int i = 0; i < n; i++)
    {
        x[i] = i;
        y[i] = 2.520 * i + 3.124;
    }

    LSRL(&line, x, y, n);

    printf("Line Equation: y = %fx + %f", line.b, line.a);

    free(x);
    free(y);

    return 0;
}

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

    size_t epochs = 2000;

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        for (size_t n = 0; n < training_data_n; n++)
        {
            feed_input(&net.layers[0], xor_truth_table_inputs, n);
            forward_propogate(&net);
            compute_cost(&net.layers[net.n - 1], xor_truth_table_labels, net.layers[net.n - 1].n, n);
            back_propogate(&net, xor_truth_table_labels, n);
            update_weights(&net, learning_rate);
        }
    }

    // test truth table

    for (size_t i = 0; i < training_data_n; i++)
    {

        printf("Case: %f, %f:\n", xor_truth_table_inputs[i][0], xor_truth_table_inputs[i][1]);

        feed_input(&net.layers[0], xor_truth_table_inputs, i);
        forward_propogate(&net);

        printf("Output: \n");
        
        for (size_t out = 0; out < net.layers[net.n - 1].n; out++)
        {
            printf("%ld: %f\n", out, net.layers[net.n - 1].neurons[out].actv);
        }

    }

    free_network(&net);
}

int main()
{

    test_network_with_XOR_data();
    
    return 0;
}