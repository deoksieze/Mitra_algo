// SPDX-License-Identifier: MPL-2.0
#include "../cpu/surface.h"
#include "geometry.cuh"
#include <map>
#include <numeric>
namespace symmetry::cuda_detail {
namespace {
struct TreeNode {
  V lo, hi;
  int left, right, begin, end;
};
struct Motion {
  M rotation;
  V translation, normal;
  double offset;
  bool reflection;
};
__device__ double bound(V p, const TreeNode &n) {
  double a[3] = {p.x, p.y, p.z}, lo[3] = {n.lo.x, n.lo.y, n.lo.z},
         hi[3] = {n.hi.x, n.hi.y, n.hi.z};
  double sum = 0;
  for (int k = 0; k < 3; ++k) {
    double d = fmax(fmax(lo[k] - a[k], 0.), a[k] - hi[k]);
    sum += d * d;
  }
  return sum;
}
__device__ double nearest(V p, const double *source, Offset vertices,
                          const Index *faces, Offset face_count,
                          const TreeNode *tree, const Index *ids, Index &face) {
  double best = INFINITY;
  face = -1;
  int stack[64], size = 1;
  stack[0] = 0;
  // The host tree is balanced and face IDs are int32: depth <= 32, stack <= 64.
  while (size) {
    const auto &node = tree[stack[--size]];
    if (bound(p, node) > best)
      continue;
    if (node.left < 0) {
      for (int j = node.begin; j < node.end; ++j) {
        Index f = ids[j];
        double d = point_triangle_distance2(
            p, read(source, vertices, faces[f], 0),
            read(source, vertices, faces[face_count + f], 0),
            read(source, vertices, faces[2 * face_count + f], 0));
        if (d < best) {
          best = d;
          face = f;
        }
      }
    } else {
      int first = node.left, second = node.right;
      if (bound(p, tree[first]) > bound(p, tree[second])) {
        int temp = first;
        first = second;
        second = temp;
      }
      stack[size++] = second;
      stack[size++] = first;
    }
  }
  return best;
}
__global__ void verify_vertices(const double *source, Offset count,
                                const Index *faces, Offset fcount,
                                const TreeNode *tree, const Index *ids,
                                Motion motion, double threshold,
                                std::uint8_t *flags, double *errors) {
  Offset row = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= 2 * count)
    return;
  Offset i = row % count;
  bool inverse = row >= count;
  flags[row] = 0;
  if (!inverse)
    errors[i] = 0;
  M r = motion.rotation;
  if (inverse) {
    for (int a = 0; a < 3; ++a)
      for (int b = a + 1; b < 3; ++b) {
        double t = r.a[3 * a + b];
        r.a[3 * a + b] = r.a[3 * b + a];
        r.a[3 * b + a] = t;
      }
  }
  V p = read(source, count, i, 0);
  V q = inverse ? apply(r, sub(p, motion.translation))
                : add(apply(r, p), motion.translation);
  if (length(sub(q, p)) < threshold * 2)
    return;
  double side = dotv(motion.normal, p) + motion.offset;
  if (motion.reflection && (inverse ? side <= threshold : side >= -threshold))
    return;
  Index face = -1;
  double distance = nearest(q, source, count, faces, fcount, tree, ids, face);
  if (distance > threshold * threshold || face < 0)
    return;
  V a = read(source, count, faces[face], 0),
    b = read(source, count, faces[fcount + face], 0),
    c = read(source, count, faces[2 * fcount + face], 0);
  V normal = unitv(crossv(sub(b, a), sub(c, a)));
  if (dotv(apply(r, read(source, count, i, 3)), normal) < .5)
    return;
  flags[row] = 1;
  if (!inverse)
    errors[i] = distance;
}
__global__ void initialize_labels(const std::uint8_t *allowed, Index *labels,
                                  Offset count) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    labels[i] = allowed[i] ? static_cast<Index>(i) : -1;
}
__global__ void propagate_labels(const Offset *offsets, const Index *neighbors,
                                 const Index *previous, Index *next,
                                 Offset count, int *changed) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  Index label = previous[i];
  if (label >= 0)
    for (Offset j = offsets[i]; j < offsets[i + 1]; ++j) {
      Index other = previous[neighbors[j]];
      if (other >= 0 && other < label)
        label = other;
    }
  next[i] = label;
  if (label != previous[i])
    atomicExch(changed, 1);
}
} // namespace
Verification verification(MeshView mesh, TopologyView topology,
                          ClustersView clusters, ModelContext context,
                          const Options &options) {
  Offset count = mesh.vertices.size(), fcount = mesh.faces.size();
  if (!clusters.size())
    return Verification::empty(0);
  if (!count || !fcount)
    throw std::invalid_argument(
        "CUDA verification requires a nonempty surface");
  cpu::Surface surface(mesh);
  std::vector<TreeNode> nodes;
  for (const auto &node : surface.tree())
    nodes.push_back({{node.lo.x, node.lo.y, node.lo.z},
                     {node.hi.x, node.hi.y, node.hi.z},
                     node.left,
                     node.right,
                     node.begin,
                     node.end});
  Buffer<TreeNode> tree(nodes.size());
  tree.upload(nodes.data(), nodes.size());
  Buffer<Index> ids(fcount), faces(3 * fcount);
  ids.upload(surface.face_ids().data(), fcount);
  for (int k = 0; k < 3; ++k)
    faces.upload(mesh.faces.component[k].data, fcount, k * fcount);
  Buffer<double> source(6 * count), errors(count);
  upload_points(source, mesh.vertices, 0);
  upload_points(source, topology.normals, 3 * count);
  Buffer<std::uint8_t> flags(2 * count), allowed(count);
  Buffer<Offset> offsets(topology.neighbors.offsets.size());
  offsets.upload(topology.neighbors.offsets.data,
                 topology.neighbors.offsets.size());
  Buffer<Index> neighbors(topology.neighbors.indices.size());
  neighbors.upload(topology.neighbors.indices.data,
                   topology.neighbors.indices.size());
  Buffer<Index> labels_a(count), labels_b(count);
  Buffer<int> changed(1);
  std::vector<std::uint8_t> mask(2 * count), permit(count);
  std::vector<double> distances(count);
  auto patches = [&](const std::vector<Index> &side) {
    std::vector<std::vector<Index>> result;
    if (side.empty())
      return result;
    std::fill(permit.begin(), permit.end(), 0);
    for (Index i : side)
      permit[i] = 1;
    allowed.upload(permit.data(), count);
    initialize_labels<<<blocks(count), threads>>>(allowed.data, labels_a.data,
                                                  count);
    check(cudaGetLastError());
    Index *previous = labels_a.data, *next = labels_b.data;
    for (Offset iteration = 0;; ++iteration) {
      if (iteration >= count)
        throw std::runtime_error("CUDA component propagation did not converge");
      check(cudaMemset(changed.data, 0, sizeof(int)));
      propagate_labels<<<blocks(count), threads>>>(
          offsets.data, neighbors.data, previous, next, count, changed.data);
      check(cudaGetLastError());
      int flag = 0;
      changed.download(&flag, 1);
      std::swap(previous, next);
      if (!flag)
        break;
    }
    std::vector<Index> host(count);
    check(cudaMemcpy(host.data(), previous, count * sizeof(Index),
                     cudaMemcpyDeviceToHost));
    std::map<Index, std::vector<Index>> components;
    for (Index i : side)
      components[host[i]].push_back(i);
    for (auto &entry : components)
      result.push_back(std::move(entry.second));
    return result;
  };
  Verification result;
  double total =
      std::accumulate(topology.weights.begin(), topology.weights.end(), 0.0);
  for (Offset ci = 0; ci < clusters.size(); ++ci) {
    auto c = clusters.records[ci];
    Motion motion{};
    for (int k = 0; k < 9; ++k)
      motion.rotation.a[k] = c.rotation[k];
    motion.translation = {c.translation.x, c.translation.y, c.translation.z};
    motion.normal = {c.plane_normal.x, c.plane_normal.y, c.plane_normal.z};
    motion.offset = c.plane_offset;
    motion.reflection = c.reflection;
    verify_vertices<<<blocks(2 * count), threads>>>(
        source.data, count, faces.data, fcount, tree.data, ids.data, motion,
        options.verify_threshold * context.box_size, flags.data, errors.data);
    check(cudaGetLastError());
    flags.download(mask.data(), 2 * count);
    errors.download(distances.data(), count);
    std::vector<Index> a, b;
    double error = 0, area = 0;
    for (Offset i = 0; i < count; ++i)
      if (mask[i]) {
        a.push_back(i);
        error += distances[i];
        area += topology.weights[i];
      }
    for (Offset i = 0; i < count; ++i)
      if (mask[count + i]) {
        b.push_back(i);
        area += topology.weights[i];
      }
    result.side_a.append(a);
    result.side_b.append(b);
    result.patches_a.append(patches(a));
    result.patches_b.append(patches(b));
    result.coverage.push_back(total > 0 ? std::min(1., area / total) : 0);
    result.rms.push_back(a.empty() ? 0 : std::sqrt(error / a.size()));
  }
  return result;
}
} // namespace symmetry::cuda_detail

#include "test_support.h"
namespace symmetry::cuda_detail {
namespace {
__device__ V convert(Vec v) { return {v.x, v.y, v.z}; }
__global__ void test_triangles(const TriangleQuery *q, Offset count,
                               double *result) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    result[i] = point_triangle_distance2(convert(q[i].p), convert(q[i].a),
                                         convert(q[i].b), convert(q[i].c));
}
} // namespace
std::vector<double>
triangle_distances(const std::vector<TriangleQuery> &queries) {
  std::vector<double> result(queries.size());
  if (queries.empty())
    return result;
  Buffer<TriangleQuery> input(queries.size());
  input.upload(queries.data(), queries.size());
  Buffer<double> output(queries.size());
  test_triangles<<<blocks(queries.size()), threads>>>(
      input.data, queries.size(), output.data);
  check(cudaGetLastError());
  output.download(result.data(), result.size());
  return result;
}
} // namespace symmetry::cuda_detail
