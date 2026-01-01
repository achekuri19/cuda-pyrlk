#include <iostream>
#include <point_detection/example.hpp>

void cleanup(float* x, float* y) {
    delete[] x;
    delete[] y;
}

int main()
{
  constexpr int N = 1 << 20;

  float* x = new float[N];
  float* y = new float[N];

  for (int i = 0; i < N; i++) {
    x[i] = static_cast<float>(i);
    y[i] = static_cast<float>(2 * i);
  }

  add_arr(N, x, y);
  for (int i = 0; i < N; i++) {
    if (y[i] != static_cast<float>(3 * i)) {
      std::cerr << "Error at index " << i << ": expected " << (x[i] + 2 * i)
                << ", got " << y[i] << std::endl;
      cleanup(x, y);
      return -1;
    }
  }
  cleanup(x, y);
  return 0;
}