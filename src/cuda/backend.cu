// SPDX-License-Identifier: MPL-2.0
#include "internal.cuh"
#include "symmetry/cuda.h"
#include "symmetry/sampling.h"
#include <cmath>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <limits>
#include <math_constants.h>
#include <stdexcept>
#include <string>

namespace symmetry {
namespace {
using namespace cuda_detail;
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
// A conservative kmin interval indexes candidates without changing the CPU
// predicate. Selected ordinals, not mesh IDs, define pair order.
__global__ void initialize_ordinals(Index *ordinals, Offset count) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    ordinals[i] = static_cast<Index>(i);
}
__device__ Offset lower_key(const double *keys, Offset count, double key,
                            bool upper) {
  Offset first = 0;
  while (count) {
    Offset half = count / 2, mid = first + half;
    if (keys[mid] < key || (upper && keys[mid] == key)) {
      first = mid + 1;
      count -= half + 1;
    } else
      count = half;
  }
  return first;
}
__global__ void candidate_ranges(const double *source, const double *keys,
                                 Offset count, double threshold,
                                 Offset *ranges) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  Offset lo = 0, hi = count;
  // The CPU predicate rounds both differences and tolerance products.
  // Inflate t before deriving a necessary interval; inflating just the final
  // radius is insufficient when 1-t is tiny. Near t=1 use the full range.
  double safe_threshold = threshold + 8 * 2.2204460492503131e-16;
  if (safe_threshold < 1) {
    double a = source[3 * count + i], b = source[4 * count + i];
    double magnitude = fmax(fmax(fabs(a), fabs(b)), 1e-10);
    // If a pair passes, Mj <= Mi/(1-t), hence |ki-kj| <= t*Mi/(1-t).
    // Round the search interval outwards; the exact predicate still decides.
    double maximum = magnitude / (1 - safe_threshold),
           radius = safe_threshold * maximum;
    double padding = 64 * 2.2204460492503131e-16 * (fabs(a) + maximum + radius);
    double low = nextafter(a - radius - padding, -CUDART_INF);
    double high = nextafter(a + radius + padding, CUDART_INF);
    lo = lower_key(keys, count, low, false);
    hi = lower_key(keys, count, high, true);
  }
  ranges[i] = lo;
  ranges[count + i] = hi;
}
__device__ bool accepts_pair(const double *source, Offset count, Index a,
                             Index b, double threshold, double distance) {
  if (b <= a)
    return false;
  double amin = source[3 * count + a], amax = source[4 * count + a];
  double bmin = source[3 * count + b], bmax = source[4 * count + b];
  double magnitude = fmax(
      fmax(fmax(fmax(fabs(amin), fabs(amax)), fabs(bmin)), fabs(bmax)), 1e-10);
  double tolerance = threshold * magnitude;
  if (fabs(amin - bmin) > tolerance || fabs(amax - bmax) > tolerance)
    return false;
  V difference = sub(read(source, count, a, 0), read(source, count, b, 0));
  return !(sqrt(dotv(difference, difference)) < distance);
}
__global__ void count_pairs(const double *source, const Index *sorted,
                            Offset count, const Offset *ranges,
                            double threshold, double distance, Offset *counts) {
  Index a = static_cast<Index>(blockIdx.x);
  Offset sum = 0;
  for (Offset k = ranges[a] + threadIdx.x; k < ranges[count + a];
       k += blockDim.x)
    sum += accepts_pair(source, count, a, sorted[k], threshold, distance);
  __shared__ cub::BlockReduce<Offset, threads>::TempStorage temporary;
  Offset total = cub::BlockReduce<Offset, threads>(temporary).Sum(sum);
  if (threadIdx.x == 0) {
    counts[a] = total;
    if (a == 0)
      counts[count] = 0;
  }
}
__global__ void fill_pairs(const double *source, const Index *sorted,
                           Offset count, const Offset *ranges,
                           const Offset *offsets, double threshold,
                           double distance, Index *accepted) {
  Index a = static_cast<Index>(blockIdx.x);
  Offset cursor = offsets[a];
  __shared__ cub::BlockScan<Offset, threads>::TempStorage temporary;
  // Candidate packets are exactly one block (256); no candidate matrix exists.
  for (Offset first = ranges[a]; first < ranges[count + a];
       first += blockDim.x) {
    Offset k = first + threadIdx.x;
    Index b = k < ranges[count + a] ? sorted[k] : -1;
    Offset flag = k < ranges[count + a] &&
                  accepts_pair(source, count, a, b, threshold, distance);
    Offset position = 0, total = 0;
    cub::BlockScan<Offset, threads>(temporary).ExclusiveSum(flag, position,
                                                            total);
    if (flag)
      accepted[cursor + position] = b;
    cursor += total;
    __syncthreads();
  }
}
__global__ void materialize_pairs(const Index *selected, const Index *accepted,
                                  const Offset *offsets, Offset total,
                                  Index *output) {
  Index a = static_cast<Index>(blockIdx.x);
  for (Offset i = offsets[a] + threadIdx.x; i < offsets[a + 1];
       i += blockDim.x) {
    output[i] = selected[a];
    output[total + i] = selected[accepted[i]];
  }
}
Pairs cuda_pairing(MeshView mesh, FeaturesView features,
                   ArrayView<const Index> selected, ModelContext context,
                   const Options &options) {
  Pairs result;
  Offset count = selected.size();
  if (count < 2)
    return result;
  require_size(features.size(), mesh.vertices.size());
  if (count > static_cast<Offset>(std::numeric_limits<Index>::max()))
    throw std::overflow_error("CUDA selected ordinals exceed int32 range");
  if (!(context.box_size > 0) || !std::isfinite(context.box_size) ||
      !(options.pair_threshold > 0 && options.pair_threshold <= 1))
    throw std::invalid_argument("CUDA pairing requires positive finite box "
                                "size and threshold in (0,1]");
  std::vector<double> packed(5 * count);
  for (Offset a = 0; a < count; ++a) {
    Index i = selected[a];
    if (i < 0 || static_cast<Offset>(i) >= mesh.vertices.size())
      throw std::invalid_argument("CUDA selected vertex index out of range");
    Vec p = mesh.vertices[i];
    double values[5] = {p.x, p.y, p.z, features.kmin[i], features.kmax[i]};
    for (int k = 0; k < 5; ++k) {
      if (!std::isfinite(values[k]))
        throw std::invalid_argument(
            "CUDA pairing requires finite coordinates and curvatures");
      packed[k * count + a] = values[k];
    }
  }
  Buffer<double> source(5 * count), keys(count);
  source.upload(packed.data(), packed.size());
  Buffer<Index> ordinals(count), sorted(count), ids(count);
  ids.upload(selected.data, count);
  initialize_ordinals<<<blocks(count), threads>>>(ordinals.data, count);
  check(cudaGetLastError());
  std::size_t bytes = 0;
  check(cub::DeviceRadixSort::SortPairs(nullptr, bytes, source.data + 3 * count,
                                        keys.data, ordinals.data, sorted.data,
                                        count));
  {
    Buffer<std::uint8_t> temporary(bytes);
    check(cub::DeviceRadixSort::SortPairs(temporary.data, bytes,
                                          source.data + 3 * count, keys.data,
                                          ordinals.data, sorted.data, count));
  }
  Buffer<Offset> ranges(2 * count), counts(count + 1), offsets(count + 1);
  candidate_ranges<<<blocks(count), threads>>>(
      source.data, keys.data, count, options.pair_threshold, ranges.data);
  check(cudaGetLastError());
  count_pairs<<<static_cast<unsigned>(count), threads>>>(
      source.data, sorted.data, count, ranges.data, options.pair_threshold,
      .03 * context.box_size, counts.data);
  check(cudaGetLastError());
  bytes = 0;
  check(cub::DeviceScan::ExclusiveSum(nullptr, bytes, counts.data, offsets.data,
                                      count + 1));
  {
    Buffer<std::uint8_t> temporary(bytes);
    check(cub::DeviceScan::ExclusiveSum(temporary.data, bytes, counts.data,
                                        offsets.data, count + 1));
  }
  Offset total = 0;
  offsets.download(&total, 1, count);
  if (total > options.max_pairs ||
      total > static_cast<Offset>(std::numeric_limits<Index>::max()))
    throw std::runtime_error("Pair budget exceeded; lower --sample-ratio or "
                             "tighten --pair-threshold");
  if (!total)
    return result;
  Buffer<Index> accepted(total), ordered(total), output(2 * total);
  fill_pairs<<<static_cast<unsigned>(count), threads>>>(
      source.data, sorted.data, count, ranges.data, offsets.data,
      options.pair_threshold, .03 * context.box_size, accepted.data);
  check(cudaGetLastError());
  // Sorting each source's accepted target ordinals restores the CPU nested-loop
  // order even if kmin keys tie or selected IDs are not sorted.
  bytes = 0;
  check(cub::DeviceSegmentedRadixSort::SortKeys(
      nullptr, bytes, accepted.data, ordered.data, total, count, offsets.data,
      offsets.data + 1));
  {
    Buffer<std::uint8_t> temporary(bytes);
    check(cub::DeviceSegmentedRadixSort::SortKeys(
        temporary.data, bytes, accepted.data, ordered.data, total, count,
        offsets.data, offsets.data + 1));
  }
  materialize_pairs<<<static_cast<unsigned>(count), threads>>>(
      ids.data, ordered.data, offsets.data, total, output.data);
  check(cudaGetLastError());
  result.a.resize(total);
  result.b.resize(total);
  output.download(result.a.data(), total);
  output.download(result.b.data(), total, total);
  return result;
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
    return cuda_detail::curvature(m, t, o);
  }
  Pairs pairing(MeshView m, FeaturesView f, ArrayView<const Index> p,
                ModelContext c, const Options &o) const override {
    return cuda_pairing(m, f, p, c, o);
  }
  ClusterOutput clustering(VotesView v, ModelContext c, const Options &o,
                           std::mt19937_64 &rng) const override {
    return cuda_detail::clustering(v, c, o, rng);
  }
  Verification verification(MeshView m, TopologyView t, ClustersView c,
                            ModelContext ctx, const Options &o) const override {
    return cuda_detail::verification(m, t, c, ctx, o);
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
