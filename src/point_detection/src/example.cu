#include <cstdio>

#include <point_detection/example.hpp>

__global__
void add_kernel(const float* x, float* y, int n) {
  // Naive implementation: single thread processes all elements
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] + y[i];
  }
}

void add_arr(int n, float* x, float* y) {

  float  *d_x, *d_y;

  // Allocate GPU memory
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_y, n * sizeof(float));

  // Copy arrays x and y to the GPU
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  // Naive - launch a single thread
  add_kernel<<<1, 1>>>(d_x, d_y, n);
  
  // Wait for the GPU to finish
  cudaDeviceSynchronize();

  // Copy result array d_y back to CPU y
  cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(d_x);
  cudaFree(d_y);

}