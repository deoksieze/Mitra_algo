// SPDX-License-Identifier: MPL-2.0
#include "symmetry/pipeline.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
namespace symmetry {
namespace {
struct EdgeFace {
  Index face, a, b;
};
Topology topology(const Mesh &m, Diagnostics &diagnostics) {
  Topology g;
  g.normals.resize(m.vertices.size());
  g.weights.resize(m.vertices.size());
  std::vector<std::vector<Index>> ring(m.vertices.size()),
      incident(m.vertices.size());
  std::map<std::pair<Index, Index>, std::vector<EdgeFace>> edges;
  for (Offset i = 0; i < m.faces.size(); ++i) {
    auto f = m.faces[i];
    Vec n = cross(m.vertices[f[1]] - m.vertices[f[0]],
                  m.vertices[f[2]] - m.vertices[f[0]]);
    double area = norm(n) / 2;
    n = unit(n);
    g.face_normals.push_back(n);
    for (int k = 0; k < 3; ++k) {
      Index a = f[k], b = f[(k + 1) % 3];
      g.normals.set(a, g.normals[a] + n);
      g.weights[a] += area / 3;
      ring[a].push_back(b);
      ring[b].push_back(a);
      incident[a].push_back(static_cast<Index>(i));
      edges[std::minmax(a, b)].push_back({static_cast<Index>(i), a, b});
    }
  }
  for (Offset i = 0; i < m.vertices.size(); ++i) {
    g.normals.set(i, unit(g.normals[i]));
    auto &r = ring[i];
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
  }
  for (const auto &edge : edges) {
    if (edge.second.size() > 2)
      ++diagnostics.nonmanifold_edges;
    if (edge.second.size() == 2 && edge.second[0].a != edge.second[1].b)
      ++diagnostics.orientation_conflicts;
    auto a = edge.second[0];
    g.edges.push_back({edge.first.first, edge.first.second, a.a, a.b, a.face,
                       edge.second.size() == 2 ? edge.second[1].face : -1});
  }
  if (diagnostics.nonmanifold_edges || diagnostics.orientation_conflicts)
    throw std::runtime_error("Mesh must have manifold edges and consistent "
                             "face winding; repair input mesh first");
  g.neighbors = Csr::from_rows(ring);
  g.incident_faces = Csr::from_rows(incident);
  return g;
}
void subdivide(Mesh &m) {
  if (m.faces.size() > 500000)
    throw std::runtime_error("Subdivision face budget exceeded");
  std::map<std::pair<Index, Index>, Index> mid;
  Faces faces;
  auto midpoint = [&](Index a, Index b) {
    auto key = std::minmax(a, b);
    auto it = mid.find(key);
    if (it != mid.end())
      return it->second;
    Index id = static_cast<Index>(m.vertices.size());
    m.vertices.push_back((m.vertices[a] + m.vertices[b]) / 2);
    mid[key] = id;
    return id;
  };
  for (auto f : m.faces) {
    Index a = midpoint(f[0], f[1]), b = midpoint(f[1], f[2]),
          c = midpoint(f[2], f[0]);
    faces.push_back({f[0], a, c});
    faces.push_back({a, f[1], b});
    faces.push_back({c, b, f[2]});
    faces.push_back({a, b, c});
  }
  m.faces = std::move(faces);
}
} // namespace
void validate_options(const Options &o) {
  if (o.max_pairs >
          static_cast<std::uint64_t>(std::numeric_limits<Index>::max()) ||
      o.max_votes >
          static_cast<std::uint64_t>(std::numeric_limits<Index>::max()))
    throw std::invalid_argument("Pair/vote budgets exceed 32-bit ID capacity");
  if (!(o.sample_ratio > 0 && o.sample_ratio <= 1) ||
      !(o.prune > 0 && o.prune < 1) ||
      !(o.pair_threshold > 0 && o.pair_threshold <= 1) ||
      !(o.bandwidth > 0 && o.bandwidth <= 2) ||
      !(o.verify_threshold > 0 && o.verify_threshold < 0.5) ||
      o.smoothing < 0 || o.smoothing > 1000 || o.subdivide < 0 ||
      o.subdivide > 3 || o.max_iterations < 1 || o.max_iterations > 10000 ||
      o.max_clusters < 1 || o.max_clusters > 1000 || o.max_pairs < 1 ||
      o.max_votes < 1)
    throw std::runtime_error("Invalid parameter range; see --help");
}
Prepared prepare(Mesh mesh, const Options &options) {
  validate_options(options);
  mesh.as_view();
  if (mesh.vertices.empty() || mesh.faces.empty())
    throw std::runtime_error("Mesh is empty");
  if (mesh.vertices.size() >
          static_cast<Offset>(std::numeric_limits<Index>::max()) ||
      mesh.faces.size() >
          static_cast<Offset>(std::numeric_limits<Index>::max()))
    throw std::runtime_error("Mesh exceeds 32-bit index capacity");
  Prepared p;
  Vec lo = mesh.vertices.front(), hi = lo;
  for (Vec v : mesh.vertices)
    for (int k = 0; k < 3; ++k) {
      if (!std::isfinite(v[k]))
        throw std::runtime_error("Non-finite coordinate");
      lo[k] = std::min(lo[k], v[k]);
      hi[k] = std::max(hi[k], v[k]);
    }
  p.context.box_size = norm(hi - lo);
  if (p.context.box_size < 1e-14)
    throw std::runtime_error("Mesh has zero extent");
  Faces usable;
  for (auto f : mesh.faces) {
    for (Index i : f)
      if (i < 0 || static_cast<Offset>(i) >= mesh.vertices.size())
        throw std::runtime_error("Face index out of range");
    if (norm(cross(mesh.vertices[f[1]] - mesh.vertices[f[0]],
                   mesh.vertices[f[2]] - mesh.vertices[f[0]])) <=
        1e-14 * p.context.box_size * p.context.box_size) {
      ++p.diagnostics.rejected_faces;
      continue;
    }
    usable.push_back(f);
  }
  mesh.faces = std::move(usable);
  if (mesh.faces.empty())
    throw std::runtime_error("No nondegenerate triangles");
  for (int i = 0; i < options.subdivide; ++i)
    subdivide(mesh);
  p.mesh = std::move(mesh);
  p.topology = topology(p.mesh, p.diagnostics);
  for (Vec v : p.mesh.vertices)
    p.context.center = p.context.center + v;
  p.context.center =
      p.context.center / static_cast<double>(p.mesh.vertices.size());
  return p;
}
} // namespace symmetry
