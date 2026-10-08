// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <array>
namespace symmetry {
struct Vec {
  double x = 0, y = 0, z = 0;
  double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  double &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
};
Vec operator+(Vec a, Vec b);
Vec operator-(Vec a, Vec b);
Vec operator*(Vec a, double s);
Vec operator/(Vec a, double s);
double dot(Vec a, Vec b);
Vec cross(Vec a, Vec b);
double norm(Vec a);
Vec unit(Vec a);
using Mat = std::array<double, 9>; // row-major
Mat identity();
Mat reflection_matrix(Vec unit_normal);
Vec transform_vector(const Mat &m, Vec v);
Mat transpose(const Mat &m);
Mat multiply(const Mat &a, const Mat &b);
double determinant(const Mat &m);
struct EigenSystem {
  std::array<double, 3> values;
  Mat vectors;
};
EigenSystem eigen_symmetric(Mat m);
double point_triangle_distance2(Vec p, Vec a, Vec b, Vec c);
} // namespace symmetry
