#include <stdio.h>
#include <math.h>
#include <malloc.h>

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

int main()
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