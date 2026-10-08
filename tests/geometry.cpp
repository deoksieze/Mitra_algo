// SPDX-License-Identifier: MPL-2.0
#include "symmetry/math.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace symmetry;
void near(double a, double b) {
  if (std::abs(a - b) > 1e-10)
    throw std::runtime_error("Geometry check failed");
}
int main() {
  try {
    Vec a{0, 0, 0}, b{1, 0, 0}, c{0, 1, 0};
    near(point_triangle_distance2({.2, .3, 2}, a, b, c), 4);
    near(point_triangle_distance2({2, 0, 0}, a, b, c), 1);
    near(point_triangle_distance2({-1, -1, 0}, a, b, c), 2);
    near(point_triangle_distance2({1, 1, 0}, a, b, c), .5);
    near(point_triangle_distance2({.5, 1, 0}, a, b, b), 1);
    near(point_triangle_distance2({1, 0, 0}, a, a, a), 1);
    Mat tensor{2, 1, .2, 1, 3, .4, .2, .4, 5};
    auto e = eigen_symmetric(tensor);
    for (int i = 0; i < 3; ++i) {
      Vec v{e.vectors[i], e.vectors[3 + i], e.vectors[6 + i]};
      near(norm(v), 1);
      near(norm(transform_vector(tensor, v) - v * e.values[i]), 0);
    }
    near(determinant(identity()), 1);
    std::cout << "Geometry checks passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
