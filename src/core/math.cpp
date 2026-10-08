// SPDX-License-Identifier: MPL-2.0
#include "symmetry/math.h"
#include <algorithm>
#include <cmath>
namespace symmetry {
namespace {
constexpr double eps = 1e-14;
}
Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec operator*(Vec a, double s) { return {a.x * s, a.y * s, a.z * s}; }
Vec operator/(Vec a, double s) { return a * (1 / s); }
double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec cross(Vec a, Vec b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
double norm(Vec a) { return std::sqrt(dot(a, a)); }
Vec unit(Vec a) {
  double n = norm(a);
  return n > eps ? a / n : Vec{};
}
Mat identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }
Vec transform_vector(const Mat &m, Vec v) {
  return {m[0] * v.x + m[1] * v.y + m[2] * v.z,
          m[3] * v.x + m[4] * v.y + m[5] * v.z,
          m[6] * v.x + m[7] * v.y + m[8] * v.z};
}
Mat transpose(const Mat &m) {
  return {m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]};
}
Mat multiply(const Mat &a, const Mat &b) {
  Mat r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        r[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
  return r;
}
double determinant(const Mat &m) {
  return m[0] * (m[4] * m[8] - m[5] * m[7]) -
         m[1] * (m[3] * m[8] - m[5] * m[6]) +
         m[2] * (m[3] * m[7] - m[4] * m[6]);
}
EigenSystem eigen_symmetric(Mat m) {
  Mat v = identity();
  for (int iteration = 0; iteration < 64; ++iteration) {
    int p = 0, q = 1;
    if (std::abs(m[2]) > std::abs(m[1])) {
      p = 0;
      q = 2;
    }
    if (std::abs(m[5]) > std::abs(m[p * 3 + q])) {
      p = 1;
      q = 2;
    }
    if (std::abs(m[p * 3 + q]) <
        1e-15 * (1 + std::abs(m[0]) + std::abs(m[4]) + std::abs(m[8])))
      break;
    double angle =
        0.5 * std::atan2(2 * m[p * 3 + q], m[q * 3 + q] - m[p * 3 + p]);
    double c = std::cos(angle), s = std::sin(angle);
    Mat j = identity();
    j[p * 3 + p] = c;
    j[q * 3 + q] = c;
    j[p * 3 + q] = s;
    j[q * 3 + p] = -s;
    m = multiply(transpose(j), multiply(m, j));
    v = multiply(v, j);
  }
  return {{m[0], m[4], m[8]}, v};
}
static double segment_distance2(Vec p, Vec a, Vec b) {
  Vec d = b - a;
  double l = dot(d, d);
  double t = l > eps ? std::clamp(dot(p - a, d) / l, 0.0, 1.0) : 0;
  Vec e = p - (a + d * t);
  return dot(e, e);
}
double point_triangle_distance2(Vec p, Vec a, Vec b, Vec c) {
  Vec ab = b - a, ac = c - a, ap = p - a;
  if (dot(cross(ab, ac), cross(ab, ac)) < eps * eps)
    return std::min({segment_distance2(p, a, b), segment_distance2(p, a, c),
                     segment_distance2(p, b, c)});
  double d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0)
    return dot(ap, ap);
  Vec bp = p - b;
  double d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3)
    return dot(bp, bp);
  double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    Vec e = p - (a + ab * (d1 / (d1 - d3)));
    return dot(e, e);
  }
  Vec cp = p - c;
  double d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6)
    return dot(cp, cp);
  double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    Vec e = p - (a + ac * (d2 / (d2 - d6)));
    return dot(e, e);
  }
  double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    Vec e = p - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))));
    return dot(e, e);
  }
  Vec e = p - (a + ab * (vb / (va + vb + vc)) + ac * (vc / (va + vb + vc)));
  return std::max(0.0, dot(e, e));
}
Mat reflection_matrix(Vec n) {
  Mat r = identity();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r[i * 3 + j] -= 2 * n[i] * n[j];
  return r;
}
} // namespace symmetry
