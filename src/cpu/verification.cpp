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
namespace {
struct Verified : ClusterRecord {
  std::vector<Index> side_a, side_b;
  std::vector<std::vector<Index>> patches_a, patches_b;
  double rms = 0, coverage = 0;
};
} // namespace
struct Node {
  Vec lo, hi;
  int left = -1, right = -1, begin = 0, end = 0;
};
class Surface {
  MeshView mesh;
  std::vector<int> ids;
  std::vector<Node> nodes;
  int build(int begin, int end) {
    Node n;
    n.lo = {1e300, 1e300, 1e300};
    n.hi = {-1e300, -1e300, -1e300};
    for (int i = begin; i < end; ++i)
      for (int v : mesh.faces[ids[i]])
        for (int k = 0; k < 3; ++k) {
          n.lo[k] = std::min(n.lo[k], mesh.vertices[v][k]);
          n.hi[k] = std::max(n.hi[k], mesh.vertices[v][k]);
        }
    n.begin = begin;
    n.end = end;
    int index = nodes.size();
    nodes.push_back(n);
    if (end - begin > 8) {
      Vec extent = n.hi - n.lo;
      int axis = extent.y > extent.x ? 1 : 0;
      if (extent.z > extent[axis])
        axis = 2;
      int mid = (begin + end) / 2;
      auto centroid = [&](int i) {
        auto f = mesh.faces[i];
        return (mesh.vertices[f[0]][axis] + mesh.vertices[f[1]][axis] +
                mesh.vertices[f[2]][axis]) /
               3;
      };
      std::nth_element(ids.begin() + begin, ids.begin() + mid,
                       ids.begin() + end,
                       [&](int a, int b) { return centroid(a) < centroid(b); });
      int l = build(begin, mid), r = build(mid, end);
      nodes[index].left = l;
      nodes[index].right = r;
    }
    return index;
  }
  double bound(Vec p, const Node &n) const {
    double s = 0;
    for (int k = 0; k < 3; ++k) {
      double d = std::max({n.lo[k] - p[k], 0.0, p[k] - n.hi[k]});
      s += d * d;
    }
    return s;
  }
  void search(int index, Vec p, double &best, int &nearest) const {
    const auto &n = nodes[index];
    if (bound(p, n) > best)
      return;
    if (n.left < 0) {
      for (int i = n.begin; i < n.end; ++i) {
        auto f = mesh.faces[ids[i]];
        double d = point_triangle_distance2(
            p, mesh.vertices[f[0]], mesh.vertices[f[1]], mesh.vertices[f[2]]);
        if (d < best) {
          best = d;
          nearest = ids[i];
        }
      }
    } else {
      int first = n.left, second = n.right;
      if (bound(p, nodes[first]) > bound(p, nodes[second]))
        std::swap(first, second);
      search(first, p, best, nearest);
      search(second, p, best, nearest);
    }
  }

public:
  explicit Surface(MeshView m) : mesh(m) {
    ids.resize(m.faces.size());
    std::iota(ids.begin(), ids.end(), 0);
    build(0, ids.size());
  }
  std::pair<double, int> distance(Vec p) const {
    double best = std::numeric_limits<double>::infinity();
    int face = -1;
    search(0, p, best, face);
    return {best, face};
  }
};
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
