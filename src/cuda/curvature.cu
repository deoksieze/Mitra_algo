// SPDX-License-Identifier: MPL-2.0
#include "internal.cuh"
namespace symmetry::cuda_detail {
namespace {
__global__ void face_normals(const double *vertices, Offset n,
                             const Index *faces, Offset count,
                             double *normals) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  V a = read(vertices, n, faces[i], 0),
    b = read(vertices, n, faces[count + i], 0),
    c = read(vertices, n, faces[2 * count + i], 0);
  V normal = unitv(crossv(b - a, c - a));
  normals[i] = normal.x;
  normals[count + i] = normal.y;
  normals[2 * count + i] = normal.z;
}
__global__ void vertex_normals(const double *faces, Offset fcount,
                               const Offset *offsets, const Index *incident,
                               Offset count, double *normals) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  V sum{};
  for (Offset j = offsets[i]; j < offsets[i + 1]; ++j)
    sum = add(sum, read(faces, fcount, incident[j], 0));
  V n = unitv(sum);
  normals[i] = n.x;
  normals[count + i] = n.y;
  normals[2 * count + i] = n.z;
}
__global__ void edge_lengths(const double *vertices, Offset n,
                             const Edge *edges, Offset count, double *lengths) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    lengths[i] = length(read(vertices, n, edges[i].hi, 0) -
                        read(vertices, n, edges[i].lo, 0));
}
__global__ void average_length(const double *lengths, Offset count,
                               double *mean) {
  double sum = 0;
  for (Offset i = 0; i < count; ++i)
    sum += lengths[i];
  *mean = count ? sum / count : 0;
}
__global__ void edge_tensors(const double *vertices, Offset n,
                             const double *normals, Offset fcount,
                             const Edge *edges, Offset count,
                             const double *mean, double *tensors) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  for (int k = 0; k < 9; ++k)
    tensors[k * count + i] = 0;
  auto edge = edges[i];
  if (edge.face1 < 0)
    return;
  V direction = read(vertices, n, edge.b, 0) - read(vertices, n, edge.a, 0);
  double edge_length = length(direction);
  direction = unitv(direction);
  V n1 = read(normals, fcount, edge.face0, 0),
    n2 = read(normals, fcount, edge.face1, 0);
  double cosine = fmin(1., fmax(-1., dotv(n1, n2)));
  double beta = atan2(dotv(crossv(n1, n2), direction), cosine);
  double d[3] = {direction.x, direction.y, direction.z};
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      tensors[(3 * a + b) * count + i] =
          d[a] * d[b] * beta * edge_length / (*mean);
}
__global__ void gather_tensors(const double *edges, Offset ecount,
                               const Offset *offsets, const Index *incident,
                               Offset count, double *tensors) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  for (int k = 0; k < 9; ++k) {
    double sum = 0;
    for (Offset j = offsets[i]; j < offsets[i + 1]; ++j)
      sum += edges[k * ecount + incident[j]];
    Offset degree = offsets[i + 1] - offsets[i];
    tensors[k * count + i] = sum / (degree ? degree : 1);
  }
}
__global__ void smooth(const double *previous, double *next, Offset count,
                       const Offset *offsets, const Index *neighbors) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  for (int k = 0; k < 9; ++k) {
    double sum = previous[k * count + i];
    for (Offset j = offsets[i]; j < offsets[i + 1]; ++j)
      sum += previous[k * count + neighbors[j]];
    next[k * count + i] = sum / (offsets[i + 1] - offsets[i] + 1);
  }
}
__global__ void local_frames(const double *tensors, const double *normals,
                             Offset count, const Offset *neighbors,
                             double *features, std::uint8_t *valid) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  for (int k = 0; k < 11; ++k)
    features[k * count + i] = 0;
  valid[i] = 0;
  V n = read(normals, count, i, 0);
  if (length(n) < .5 || neighbors[i + 1] - neighbors[i] < 3)
    return;
  V axis = fabs(n.x) < .8 ? V{1, 0, 0} : V{0, 1, 0};
  V u = unitv(crossv(n, axis)), v = crossv(n, u);
  M tensor{};
  for (int k = 0; k < 9; ++k)
    tensor.a[k] = tensors[k * count + i];
  double a = dotv(u, apply(tensor, u)), b = dotv(u, apply(tensor, v)),
         c = dotv(v, apply(tensor, v));
  double angle = .5 * atan2(2 * b, a - c);
  V high = u * cos(angle) + v * sin(angle);
  V low = unitv(crossv(high, n)), maximum = unitv(crossv(n, low));
  double delta = hypot(a - c, 2 * b), kmin = (a + c - delta) / 2,
         kmax = (a + c + delta) / 2;
  double values[11] = {n.x,       n.y,       n.z,       low.x, low.y, low.z,
                       maximum.x, maximum.y, maximum.z, kmin,  kmax};
  for (int k = 0; k < 11; ++k)
    features[k * count + i] = values[k];
  valid[i] = isfinite(kmin) && isfinite(kmax) && delta > 1e-10;
}
} // namespace
Features curvature(MeshView mesh, TopologyView topology,
                   const Options &options) {
  Offset n = mesh.vertices.size(), f = mesh.faces.size(),
         e = topology.edges.size();
  Features result;
  result.resize(n);
  if (!n)
    return result;
  Buffer<double> vertices(3 * n), fn(3 * f), vn(3 * n), lengths(e), mean(1),
      et(9 * e), a(9 * n), b(9 * n), output(11 * n);
  upload_points(vertices, mesh.vertices, 0);
  Buffer<Index> faces(3 * f);
  for (int k = 0; k < 3; ++k)
    faces.upload(mesh.faces.component[k].data, f, k * f);
  Buffer<Edge> edges(e);
  edges.upload(topology.edges.data, e);
  Buffer<Offset> face_offsets(topology.incident_faces.offsets.size()),
      neighbor_offsets(topology.neighbors.offsets.size());
  face_offsets.upload(topology.incident_faces.offsets.data,
                      topology.incident_faces.offsets.size());
  neighbor_offsets.upload(topology.neighbors.offsets.data,
                          topology.neighbors.offsets.size());
  Buffer<Index> incident_faces(topology.incident_faces.indices.size()),
      neighbors(topology.neighbors.indices.size());
  incident_faces.upload(topology.incident_faces.indices.data,
                        topology.incident_faces.indices.size());
  neighbors.upload(topology.neighbors.indices.data,
                   topology.neighbors.indices.size());
  std::vector<std::vector<Index>> rows(n);
  for (Offset i = 0; i < e; ++i)
    if (topology.edges[i].face1 >= 0) {
      rows[topology.edges[i].a].push_back(i);
      rows[topology.edges[i].b].push_back(i);
    }
  auto csr = Csr::from_rows(rows);
  Buffer<Offset> edge_offsets(csr.offsets.size());
  Buffer<Index> incident_edges(csr.indices.size());
  edge_offsets.upload(csr.offsets.data(), csr.offsets.size());
  incident_edges.upload(csr.indices.data(), csr.indices.size());
  if (f) {
    face_normals<<<blocks(f), threads>>>(vertices.data, n, faces.data, f,
                                         fn.data);
    check(cudaGetLastError());
  }
  vertex_normals<<<blocks(n), threads>>>(fn.data, f, face_offsets.data,
                                         incident_faces.data, n, vn.data);
  check(cudaGetLastError());
  if (e) {
    edge_lengths<<<blocks(e), threads>>>(vertices.data, n, edges.data, e,
                                         lengths.data);
    check(cudaGetLastError());
  }
  average_length<<<1, 1>>>(lengths.data, e, mean.data);
  check(cudaGetLastError());
  if (e) {
    edge_tensors<<<blocks(e), threads>>>(vertices.data, n, fn.data, f,
                                         edges.data, e, mean.data, et.data);
    check(cudaGetLastError());
  }
  gather_tensors<<<blocks(n), threads>>>(et.data, e, edge_offsets.data,
                                         incident_edges.data, n, a.data);
  check(cudaGetLastError());
  double *previous = a.data, *next = b.data;
  for (int iteration = 0; iteration < options.smoothing; ++iteration) {
    smooth<<<blocks(n), threads>>>(previous, next, n, neighbor_offsets.data,
                                   neighbors.data);
    check(cudaGetLastError());
    std::swap(previous, next);
  }
  Buffer<std::uint8_t> valid(n);
  local_frames<<<blocks(n), threads>>>(
      previous, vn.data, n, neighbor_offsets.data, output.data, valid.data);
  check(cudaGetLastError());
  auto lane = [&](std::vector<double> &v, int k) {
    output.download(v.data(), n, k * n);
  };
  lane(result.normal.x, 0);
  lane(result.normal.y, 1);
  lane(result.normal.z, 2);
  lane(result.dmin.x, 3);
  lane(result.dmin.y, 4);
  lane(result.dmin.z, 5);
  lane(result.dmax.x, 6);
  lane(result.dmax.y, 7);
  lane(result.dmax.z, 8);
  lane(result.kmin, 9);
  lane(result.kmax, 10);
  valid.download(result.valid.data(), n);
  return result;
}
} // namespace symmetry::cuda_detail
