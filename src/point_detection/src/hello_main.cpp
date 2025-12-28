#include <iostream>
#include <example.hpp>

int main()
{
  float x[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  float y[10] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  int n = 10;

  add_arr(n, x, y);
  for (int i = 0; i < n; i++) {
    std::cout << "y[" << i << "] = " << y[i] << std::endl;
  }
  return 0;
}