// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/pipeline.h"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
namespace symmetry::cuda_detail {
inline void check(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA: ") +
                             cudaGetErrorString(status));
}
template <class T> class Buffer {
public:
  T *data = nullptr;
  explicit Buffer(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::overflow_error("CUDA allocation size overflow");
    if (n)
      check(cudaMalloc(reinterpret_cast<void **>(&data), n * sizeof(T)));
  }
  ~Buffer() {
    if (data)
      cudaFree(data);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void upload(const T *source, std::size_t n, std::size_t offset = 0) {
    if (n)
      check(cudaMemcpy(data + offset, source, n * sizeof(T),
                       cudaMemcpyHostToDevice));
  }
  void download(T *target, std::size_t n, std::size_t offset = 0) const {
    if (n)
      check(cudaMemcpy(target, data + offset, n * sizeof(T),
                       cudaMemcpyDeviceToHost));
  }
};
constexpr unsigned threads = 256;
inline unsigned blocks(Offset n) {
  return static_cast<unsigned>((n + threads - 1) / threads);
}

// Device PODs never contain host views or STL owners.
struct V {
  double x, y, z;
};
struct M {
  double a[9];
};
__device__ inline V add(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ inline V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ inline V scale(V a, double s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ inline double dotv(V a, V b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ inline V apply(M m, V v) {
  return {m.a[0] * v.x + m.a[1] * v.y + m.a[2] * v.z,
          m.a[3] * v.x + m.a[4] * v.y + m.a[5] * v.z,
          m.a[6] * v.x + m.a[7] * v.y + m.a[8] * v.z};
}
__device__ inline V read(const double *s, Offset n, Index i, int lane) {
  return {s[lane * n + i], s[(lane + 1) * n + i], s[(lane + 2) * n + i]};
}
__device__ inline M frame(V a, V b, V c) {
  return {{a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z}};
}
__device__ inline V crossv(V a, V b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ inline double length(V a) { return sqrt(dotv(a, a)); }
__device__ inline V unitv(V a) {
  double n = length(a);
  return n > 1e-14 ? scale(a, 1 / n) : V{};
}
__device__ inline V operator+(V a, V b) { return add(a, b); }
__device__ inline V operator-(V a, V b) { return sub(a, b); }
__device__ inline V operator*(V a, double s) { return scale(a, s); }
__device__ inline V operator/(V a, double s) { return scale(a, 1 / s); }
inline void upload_points(Buffer<double> &device, PointsView p, Offset offset) {
  device.upload(p.x.data, p.size(), offset);
  device.upload(p.y.data, p.size(), offset + p.size());
  device.upload(p.z.data, p.size(), offset + 2 * p.size());
}
Features curvature(MeshView, TopologyView, const Options &);
Verification verification(MeshView, TopologyView, ClustersView, ModelContext,
                          const Options &);
ClusterOutput clustering(VotesView, ModelContext, const Options &,
                         std::mt19937_64 &);
} // namespace symmetry::cuda_detail
