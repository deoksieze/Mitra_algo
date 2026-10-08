// SPDX-License-Identifier: MPL-2.0
#include "internal.h"
#include "surface.h"
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
namespace {
struct Verified : ClusterRecord {
  std::vector<Index> side_a, side_b;
  std::vector<std::vector<Index>> patches_a, patches_b;
  double rms = 0, coverage = 0;
};
} // namespace
std::vector<std::vector<int>> components(ArrayView<const Index> ids,
                                         TopologyView g) {
  std::vector<bool> allowed(g.neighbors.size()), seen(g.neighbors.size());
  for (int v : ids)
    allowed[v] = true;
  std::vector<std::vector<int>> result;
  for (int seed : ids) {
    if (seen[seed])
      continue;
    std::queue<int> q;
    q.push(seed);
    seen[seed] = true;
    std::vector<int> patch;
    while (!q.empty()) {
      int v = q.front();
      q.pop();
      patch.push_back(v);
      for (int n : g.neighbors[v])
        if (allowed[n] && !seen[n]) {
          seen[n] = true;
          q.push(n);
        }
    }
    std::sort(patch.begin(), patch.end());
    result.push_back(std::move(patch));
  }
  return result;
}
Verification verification(MeshView mesh, TopologyView g, ClustersView clusters,
                          ModelContext context, const Options &options) {
  Surface surface(mesh);
  double threshold = options.verify_threshold * context.box_size;
  Verification result;
  for (Offset ci = 0; ci < clusters.size(); ++ci) {
    Verified c;
    static_cast<ClusterRecord &>(c) = clusters.records[ci];
    double error = 0, area = 0,
           total = std::accumulate(g.weights.begin(), g.weights.end(), 0.0);
    Mat inverse = transpose(c.rotation);
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
      Vec p = mesh.vertices[i],
          q = transform_vector(c.rotation, p) + c.translation;
      if (norm(q - p) < threshold * 2)
        continue;
      if (c.reflection && dot(c.plane_normal, p) + c.plane_offset >= -threshold)
        continue;
      auto nearest = surface.distance(q);
      if (nearest.first > threshold * threshold)
        continue;
      auto face = mesh.faces[nearest.second];
      Vec n = unit(cross(mesh.vertices[face[1]] - mesh.vertices[face[0]],
                         mesh.vertices[face[2]] - mesh.vertices[face[0]]));
      if (dot(transform_vector(c.rotation, g.normals[i]), n) < 0.5)
        continue;
      c.side_a.push_back(i);
      error += nearest.first;
      area += g.weights[i];
    }
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
      Vec p = mesh.vertices[i],
          q = transform_vector(inverse, p - c.translation);
      if (norm(q - p) < threshold * 2)
        continue;
      if (c.reflection && dot(c.plane_normal, p) + c.plane_offset <= threshold)
        continue;
      auto nearest = surface.distance(q);
      if (nearest.first > threshold * threshold)
        continue;
      auto face = mesh.faces[nearest.second];
      Vec n = unit(cross(mesh.vertices[face[1]] - mesh.vertices[face[0]],
                         mesh.vertices[face[2]] - mesh.vertices[face[0]]));
      if (dot(transform_vector(inverse, g.normals[i]), n) < 0.5)
        continue;
      c.side_b.push_back(i);
      area += g.weights[i];
    }
    c.rms = c.side_a.empty() ? 0 : std::sqrt(error / c.side_a.size());
    c.coverage = total > 0 ? std::min(1.0, area / total) : 0;
    c.patches_a = components(view(c.side_a), g);
    c.patches_b = components(view(c.side_b), g);
    result.side_a.append(c.side_a);
    result.side_b.append(c.side_b);
    result.patches_a.append(c.patches_a);
    result.patches_b.append(c.patches_b);
    result.coverage.push_back(c.coverage);
    result.rms.push_back(c.rms);
  }
  return result;
}

} // namespace symmetry::cpu
