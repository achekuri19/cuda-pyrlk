#include <example.hpp>

void add_arr(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}