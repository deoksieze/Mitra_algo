// SPDX-License-Identifier: MPL-2.0
#include "symmetry/cuda.h"
#include "symmetry/sampling.h"
#include <cmath>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace symmetry {
namespace {
void check(cudaError_t status) {
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
unsigned blocks(Offset n) {
  return static_cast<unsigned>((n + threads - 1) / threads);
}

__global__ void prune(const Index *sample, Offset count, const double *k,
                      const std::uint8_t *valid, Offset vertices,
                      double threshold, std::uint8_t *flags) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  Index v = sample[i];
  double a = k[v], b = k[vertices + v];
  double maximum = fabs(a) < fabs(b) ? fabs(b) : fabs(a);
  flags[i] = valid[v] && maximum > 1e-10 && fabs(a - b) / maximum > threshold;
}
// Device PODs never contain host views or STL owners.
struct V {
  double x, y, z;
};
struct M {
  double a[9];
};
__device__ V add(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ V sub(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ V scale(V a, double s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ double dotv(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ V apply(M m, V v) {
  return {m.a[0] * v.x + m.a[1] * v.y + m.a[2] * v.z,
          m.a[3] * v.x + m.a[4] * v.y + m.a[5] * v.z,
          m.a[6] * v.x + m.a[7] * v.y + m.a[8] * v.z};
}
__device__ V read(const double *s, Offset n, Index i, int lane) {
  return {s[lane * n + i], s[(lane + 1) * n + i], s[(lane + 2) * n + i]};
}
__device__ M frame(V a, V b, V c) {
  return {{a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z}};
}
struct PairGeometry {
  V p, q, an, ad, ax, bn, bd, bx, n;
  M mirror;
  double d;
  bool reflection;
};
__device__ PairGeometry geometry(const double *s, Offset n, const Index *pairs,
                                 Offset count, Offset i) {
  Index a = pairs[i], b = pairs[count + i];
  PairGeometry g{};
  g.p = read(s, n, a, 0);
  g.q = read(s, n, b, 0);
  g.an = read(s, n, a, 3);
  g.ad = read(s, n, a, 6);
  g.ax = read(s, n, a, 9);
  g.bn = read(s, n, b, 3);
  g.bd = read(s, n, b, 6);
  g.bx = read(s, n, b, 9);
  V difference = sub(g.q, g.p);
  double length = sqrt(dotv(difference, difference));
  g.n = length > 1e-14 ? scale(difference, 1 / length) : V{};
  double components[3] = {g.n.x, g.n.y, g.n.z};
  int dominant = 0;
  for (int k = 1; k < 3; ++k)
    if (fabs(components[k]) > fabs(components[dominant]))
      dominant = k;
  if (components[dominant] < 0)
    g.n = scale(g.n, -1);
  g.d = -dotv(g.n, scale(add(g.p, g.q), .5));
  double normal[3] = {g.n.x, g.n.y, g.n.z};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      g.mirror.a[r * 3 + c] = (r == c ? 1.0 : 0.0) - 2 * normal[r] * normal[c];
  // Degenerate point pairs retain the CPU reference's zero-normal behavior.
  g.reflection = dotv(apply(g.mirror, g.an), g.bn) > .75 &&
                 fabs(dotv(apply(g.mirror, g.ad), g.bd)) > .75;
  return g;
}
__global__ void count_votes(const double *s, Offset n, const Index *pairs,
                            Offset count, Offset *counts) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    counts[i] = geometry(s, n, pairs, count, i).reflection ? 3 : 2;
  if (i == 0)
    counts[count] = 0;
}
// Output scalar lanes: R[9], t[3], plane normal[3], d, descriptor[12].
__device__ void write_vote(double *out, Index *pair_ids, std::uint8_t *kinds,
                           Offset total, Offset row, Index pair,
                           bool reflection, M rotation, V t, V normal, double d,
                           const double *desc) {
  pair_ids[row] = pair;
  kinds[row] = reflection ? 1 : 0;
  for (int k = 0; k < 9; ++k)
    out[k * total + row] = rotation.a[k];
  double values[7] = {t.x, t.y, t.z, normal.x, normal.y, normal.z, d};
  for (int k = 0; k < 7; ++k)
    out[(9 + k) * total + row] = values[k];
  for (int k = 0; k < 12; ++k)
    out[(16 + k) * total + row] = desc[k];
}
__global__ void fill_votes(const double *s, Offset n, const Index *pairs,
                           Offset count, const Offset *offsets, Offset total,
                           double box, V center, double *out, Index *ids,
                           std::uint8_t *kinds) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  PairGeometry g = geometry(s, n, pairs, count, i);
  Offset row = offsets[i];
  if (g.reflection) {
    double desc[12] = {g.n.x, g.n.y, g.n.z, (g.d + dotv(g.n, center)) / box};
    write_vote(out, ids, kinds, total, row++, static_cast<Index>(i), true,
               g.mirror, scale(g.n, -2 * g.d), g.n, g.d, desc);
  }
  M a = frame(g.ad, g.ax, g.an);
  for (int sign = 1; sign >= -1; sign -= 2) {
    M b = frame(scale(g.bd, sign), scale(g.bx, sign), g.bn), r{};
    for (int x = 0; x < 3; ++x)
      for (int y = 0; y < 3; ++y)
        for (int k = 0; k < 3; ++k)
          r.a[x * 3 + y] += b.a[x * 3 + k] * a.a[y * 3 + k];
    V t = sub(g.q, apply(r, g.p));
    V centered = scale(sub(add(apply(r, center), t), center), 1 / box);
    double desc[12];
    for (int k = 0; k < 9; ++k)
      desc[k] = r.a[k] / 2;
    desc[9] = centered.x;
    desc[10] = centered.y;
    desc[11] = centered.z;
    write_vote(out, ids, kinds, total, row++, static_cast<Index>(i), false, r,
               t, {}, 0, desc);
  }
}
void upload_points(Buffer<double> &device, PointsView p, Offset offset) {
  device.upload(p.x.data, p.size(), offset);
  device.upload(p.y.data, p.size(), offset + p.size());
  device.upload(p.z.data, p.size(), offset + 2 * p.size());
}
class CudaBackend final : public Backend {
  std::unique_ptr<Backend> cpu = make_cpu_backend();

public:
  CudaBackend() {
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (!count)
      throw std::runtime_error("CUDA: no available device");
    check(cudaFree(nullptr)); // Context initialization outside stage timings.
  }
  const char *name() const noexcept override { return "cuda"; }
  Features curvature(MeshView m, TopologyView t,
                     const Options &o) const override {
    return cpu->curvature(m, t, o);
  }
  Pairs pairing(MeshView m, FeaturesView f, ArrayView<const Index> p,
                ModelContext c, const Options &o) const override {
    return cpu->pairing(m, f, p, c, o);
  }
  ClusterOutput clustering(VotesView v, ModelContext c, const Options &o,
                           std::mt19937_64 &rng) const override {
    return cpu->clustering(v, c, o, rng);
  }
  Verification verification(MeshView m, TopologyView t, ClustersView c,
                            ModelContext ctx, const Options &o) const override {
    return cpu->verification(m, t, c, ctx, o);
  }
  Selection select(MeshView m, FeaturesView f, ArrayView<const Scalar> w,
                   const Options &o, std::mt19937_64 &rng) const override {
    require_size(w.size(), m.vertices.size());
    require_size(f.size(), w.size());
    Selection result;
    result.sample = sample_vertices(w, o.sample_ratio, rng);
    Offset count = result.sample.size(), n = w.size();
    if (!count)
      return result;
    Buffer<double> k(2 * n);
    k.upload(f.kmin.data, n);
    k.upload(f.kmax.data, n, n);
    Buffer<std::uint8_t> valid(n), flags(count);
    valid.upload(f.valid.data, n);
    Buffer<Index> sample(count), selected(count);
    sample.upload(result.sample.data(), count);
    prune<<<blocks(count), threads>>>(sample.data, count, k.data, valid.data, n,
                                      1 - o.prune, flags.data);
    check(cudaGetLastError());
    Buffer<Offset> selected_count(1);
    std::size_t bytes = 0;
    check(cub::DeviceSelect::Flagged(nullptr, bytes, sample.data, flags.data,
                                     selected.data, selected_count.data,
                                     count));
    Buffer<std::uint8_t> temporary(bytes);
    check(cub::DeviceSelect::Flagged(temporary.data, bytes, sample.data,
                                     flags.data, selected.data,
                                     selected_count.data, count));
    Offset kept = 0;
    selected_count.download(&kept, 1);
    result.pruned.resize(kept);
    selected.download(result.pruned.data(), kept);
    return result;
  }
  Votes transformations(MeshView m, FeaturesView f, PairsView p, ModelContext c,
                        const Options &o) const override {
    Votes result;
    Offset count = p.size(), n = m.vertices.size();
    if (!count)
      return result;
    require_size(f.size(), n);
    if (!(c.box_size > 0) || !std::isfinite(c.box_size) ||
        !std::isfinite(c.center.x) || !std::isfinite(c.center.y) ||
        !std::isfinite(c.center.z))
      throw std::invalid_argument("CUDA transformations require a positive "
                                  "finite box size and finite center");
    if (count > static_cast<Offset>(std::numeric_limits<Index>::max()))
      throw std::overflow_error("CUDA pair IDs exceed int32 range");
    for (Offset i = 0; i < count; ++i)
      if (p.a[i] < 0 || p.b[i] < 0 || static_cast<Offset>(p.a[i]) >= n ||
          static_cast<Offset>(p.b[i]) >= n)
        throw std::invalid_argument("CUDA pair vertex index out of range");
    Buffer<double> source(12 * n);
    upload_points(source, m.vertices, 0);
    upload_points(source, f.normal, 3 * n);
    upload_points(source, f.dmin, 6 * n);
    upload_points(source, f.dmax, 9 * n);
    Buffer<Index> pairs(2 * count);
    pairs.upload(p.a.data, count);
    pairs.upload(p.b.data, count, count);
    Buffer<Offset> counts(count + 1), offsets(count + 1);
    count_votes<<<blocks(count), threads>>>(source.data, n, pairs.data, count,
                                            counts.data);
    check(cudaGetLastError());
    std::size_t bytes = 0;
    check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts.data,
                                        offsets.data, count + 1));
    Buffer<std::uint8_t> temporary(bytes);
    check(cub::DeviceScan::ExclusiveSum(temporary.data, bytes, counts.data,
                                        offsets.data, count + 1));
    Offset total = 0;
    offsets.download(&total, 1, count);
    if (total > o.max_votes ||
        total > static_cast<Offset>(std::numeric_limits<Index>::max()))
      throw std::runtime_error("Vote budget exceeded; lower --sample-ratio or "
                               "tighten --pair-threshold");
    Buffer<double> output(28 * total);
    Buffer<Index> ids(total);
    Buffer<std::uint8_t> kinds(total);
    fill_votes<<<blocks(count), threads>>>(source.data, n, pairs.data, count,
                                           offsets.data, total, c.box_size,
                                           {c.center.x, c.center.y, c.center.z},
                                           output.data, ids.data, kinds.data);
    check(cudaGetLastError());
    result.pairs.resize(total);
    ids.download(result.pairs.data(), total);
    std::vector<std::uint8_t> host_kind(total);
    kinds.download(host_kind.data(), total);
    result.kind.reserve(total);
    for (auto k : host_kind)
      result.kind.push_back(static_cast<TransformKind>(k));
    auto lane = [&](std::vector<double> &v, int k) {
      v.resize(total);
      output.download(v.data(), total, k * total);
    };
    for (int k = 0; k < 9; ++k)
      lane(result.rotation.component[k], k);
    lane(result.translation.x, 9);
    lane(result.translation.y, 10);
    lane(result.translation.z, 11);
    lane(result.plane_normal.x, 12);
    lane(result.plane_normal.y, 13);
    lane(result.plane_normal.z, 14);
    lane(result.plane_offset, 15);
    for (int k = 0; k < 12; ++k)
      lane(result.descriptor.component[k], 16 + k);
    result.as_view();
    return result;
  }
};
} // namespace
std::unique_ptr<Backend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();
}
} // namespace symmetry
