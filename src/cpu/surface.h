// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/pipeline.h"
#include <algorithm>
#include <limits>
#include <numeric>
namespace symmetry::cpu {
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
  const std::vector<Node> &tree() const { return nodes; }
  const std::vector<int> &face_ids() const { return ids; }
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
} // namespace symmetry::cpu
