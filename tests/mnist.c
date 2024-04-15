#include <stdio.h>

#include <net.h>

#define LOGGING

int get_mnist_data(const char* file_name);
void train_network();
void test_model(struct network_t* net);
void free_data();

size_t train_n;
size_t x,y;

float** mnist_data;
float**  mnist_labels;

int get_mnist_data(const char* file_name)
{
    printf("Reading in file %s.\n", file_name);

    FILE* file;
    file = fopen(file_name, "r");

    if (file == NULL)
    {
        printf("Could not open file: %s.\n", file_name);
        return 1;
    }

    printf("Opened file %s.\n", file_name);

    fscanf(file, "%ld\n", &train_n);
    fscanf(file, "%ld %ld\n", &x, &y);

    printf("Reading in %ld data points of size %ldx%ld.\n", train_n, x, y);
    size_t flattened_size = x * y;

    mnist_data = (float**) malloc(train_n * sizeof(float*));

    for (size_t i = 0; i < train_n; i++)
    {
        mnist_data[i] = malloc(flattened_size * sizeof(float));

        for (size_t j = 0; j < flattened_size; j++)
        {
            fscanf(file, "%f ", &mnist_data[i][j]);
        }
    }

    fscanf(file, "\n");

    mnist_labels = (float**) malloc(train_n * sizeof(float*));

    for (size_t i = 0; i < train_n; i++)
    {
        mnist_labels[i] = malloc(sizeof(float));

        fscanf(file, "%f ", &mnist_labels[i][0]);
    }

    fclose(file);

    return 0;
}

void train_model_local(struct network_t* network, float** training_data, float** labels, size_t data_size, size_t epochs)
{
    printf("Entered function train_model_lcoal\n");
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        printf("Epoch %ld:\n", epoch);
        for (size_t n = 0; n < 1000; n++)
        {
            feed_input(&network->layers[0], training_data[n]);
            forward_propogate(network);
            compute_cost(&network->layers[network->n - 1], labels, network->layers[network->n - 1].n, n);
            back_propogate(network, labels, n);
            update_weights(network);
        }

        test_model(network);
    }
}

void train_network()
{
    printf("Beginning initializing network\n");
    struct network_t net;

    size_t layers = 4;
    size_t layer_sizes[] = {28 * 28, 64, 256, 10};

    initialize_network(&net, layer_sizes, layers);

    size_t epochs = 100;

    printf("Initialized network\n");

    train_model_local(&net, mnist_data, mnist_labels, train_n, epochs);

    printf("Testing model\n");

}


void test_model(struct network_t* net)
{

    struct layer_t* output_layer = &net->layers[net->n - 1];

    for (size_t i = 0; i < 1; i++)
    {

        //printf("Case: %ld:\n", i);


        feed(net, mnist_data[i]);
        
        for (size_t out = 0; out < output_layer->n; out++)
        {
            printf("%ld: %f\n", out, output_layer->neurons[out].actv);
        }
    }
}

void free_data()
{
    for (size_t i = 0; i < train_n; i++)
    {
        free(mnist_data[i]);
        free(mnist_labels[i]);
    }

    free(mnist_data);
    free(mnist_labels);
}

int main()
{
    int opened_file = get_mnist_data("./mnist_training.dat");
    if (opened_file != 0)
    {
        return 1;
    }

    for (size_t r = 0; r < x; r++)
    {
        for (size_t c = 0; c < y; c++)
        {
            printf("%d ", (unsigned int) mnist_data[0][r * x + c]);
        }
        printf("\n");
    }

    printf("Beginning training network.\n");

    train_network();

    printf("Freeing data\n");

    free_data();
}