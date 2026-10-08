// SPDX-License-Identifier: MPL-2.0
#include "internal.h"
#include "symmetry/pipeline.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

namespace symmetry::cpu {
Features curvature(MeshView m, TopologyView g, const Options &options) {
  std::vector<Mat> tensors(m.vertices.size());
  std::vector<int> count(m.vertices.size());
  double mean_length = 0;
  for (const auto &edge : g.edges)
    mean_length += norm(m.vertices[edge.hi] - m.vertices[edge.lo]);
  mean_length /= g.edges.size();
  for (const auto &edge : g.edges) {
    if (edge.face1 < 0)
      continue;
    Vec direction = m.vertices[edge.b] - m.vertices[edge.a];
    double length = norm(direction);
    direction = unit(direction);
    Vec n1 = g.face_normals[edge.face0], n2 = g.face_normals[edge.face1];
    double beta = std::atan2(dot(cross(n1, n2), direction),
                             std::clamp(dot(n1, n2), -1.0, 1.0));
    for (int vertex : {edge.a, edge.b}) {
      ++count[vertex];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          tensors[vertex][i * 3 + j] +=
              direction[i] * direction[j] * beta * length / mean_length;
    }
  }
  for (std::size_t i = 0; i < tensors.size(); ++i)
    for (double &x : tensors[i])
      x /= std::max(1, count[i]);
  for (int iteration = 0; iteration < options.smoothing; ++iteration) {
    auto next = tensors;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      for (int neighbor : g.neighbors[i])
        for (int k = 0; k < 9; ++k)
          next[i][k] += tensors[neighbor][k];
      for (double &x : next[i])
        x /= (g.neighbors[i].size() + 1);
    }
    tensors.swap(next);
  }
  Features result;
  result.resize(m.vertices.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    Vec n = g.normals[i];
    if (norm(n) < 0.5 || g.neighbors[i].size() < 3)
      continue;
    // Restrict to the tangent plane: keeps direction/eigenvalue association
    // explicit.
    Vec axis = std::abs(n.x) < 0.8 ? Vec{1, 0, 0} : Vec{0, 1, 0};
    Vec u = unit(cross(n, axis)), v = cross(n, u);
    double a = dot(u, transform_vector(tensors[i], u)),
           b = dot(u, transform_vector(tensors[i], v)),
           c = dot(v, transform_vector(tensors[i], v));
    double angle = 0.5 * std::atan2(2 * b, a - c);
    Vec high = u * std::cos(angle) + v * std::sin(angle);
    Vec low = unit(cross(high, n));
    double delta = std::hypot(a - c, 2 * b);
    Feature f;
    f.normal = n;
    f.kmin = (a + c - delta) / 2;
    f.kmax = (a + c + delta) / 2;
    f.dmin = low;
    f.dmax = unit(cross(n, low));
    f.valid = std::isfinite(f.kmin) && std::isfinite(f.kmax) && delta > 1e-10;
    result.set(i, f);
  }
  return result;
}

} // namespace symmetry::cpu
