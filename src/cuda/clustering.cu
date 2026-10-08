// SPDX-License-Identifier: MPL-2.0
#include "../cpu/internal.h"
#include "internal.cuh"
#include <cub/device/device_select.cuh>
namespace symmetry::cuda_detail {
namespace {
struct Center {
  double value[12];
  int dimension;
  bool reflection;
};
struct TrajectoryState {
  Center center;
  int iterations;
  bool converged, exhausted;
};
// One cooperative block retains the trajectory between iterations. Components
// sum in vote order, preserving CPU arithmetic; distance evaluations are
// parallel.
__global__ void trajectory_batch(const double *descriptors,
                                 const std::uint8_t *kinds, Offset count,
                                 double h2, int maximum, TrajectoryState *state,
                                 double *weights, Offset *support, Index *ids) {
  __shared__ TrajectoryState shared;
  __shared__ double sums[13];
  if (threadIdx.x == 0)
    shared = *state;
  __syncthreads();
  if (shared.iterations == 0)
    for (Offset i = threadIdx.x; i < count; i += blockDim.x) {
      support[i] = 0;
      ids[i] = i;
    }
  __syncthreads();
  // Bounded batches reduce host round trips without unbounded kernel duration.
  for (int batch = 0; batch < 8 && shared.iterations < maximum &&
                      !shared.converged && !shared.exhausted;
       ++batch) {
    for (Offset i = threadIdx.x; i < count; i += blockDim.x) {
      double distance = 0;
      bool inside = false;
      if ((kinds[i] != 0) == shared.center.reflection) {
        for (int k = 0; k < shared.center.dimension; ++k) {
          double d = shared.center.value[k] - descriptors[k * count + i];
          distance += d * d;
        }
        inside = distance < h2;
      }
      weights[i] = inside ? 1 - distance / h2 : 0;
      if (inside)
        ++support[i];
    }
    __syncthreads();
    unsigned k = threadIdx.x;
    if (k <= static_cast<unsigned>(shared.center.dimension)) {
      double sum = 0;
      for (Offset i = 0; i < count; ++i)
        if (weights[i] != 0)
          sum += k ? weights[i] * descriptors[(k - 1) * count + i] : weights[i];
      sums[k] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      ++shared.iterations;
      if (sums[0] <= 1e-14)
        shared.exhausted = true;
      else {
        double movement = 0;
        for (int k = 0; k < shared.center.dimension; ++k) {
          double next = sums[k + 1] / sums[0],
                 d = shared.center.value[k] - next;
          movement += d * d;
          shared.center.value[k] = next;
        }
        shared.converged = movement < 1e-6 * h2;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    *state = shared;
}
__global__ void gather_support(const Index *members, const Offset *support,
                               Offset count, Offset *output) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    output[i] = support[members[i]];
}
} // namespace
ClusterOutput clustering(VotesView votes, ModelContext context,
                         const Options &options, std::mt19937_64 &rng) {
  Offset count = votes.size();
  if (!count)
    return cpu::clustering(votes, context, options, rng);
  Buffer<double> descriptors(12 * count), weights(count);
  for (int k = 0; k < 12; ++k)
    descriptors.upload(votes.descriptor.component[k].data, count, k * count);
  Buffer<std::uint8_t> kinds(count);
  std::vector<std::uint8_t> types(count);
  for (Offset i = 0; i < count; ++i)
    types[i] = votes.is_reflection(i);
  kinds.upload(types.data(), count);
  Buffer<Offset> support(count), counts(count), kept(1);
  Buffer<Index> ids(count), members(count);
  Buffer<TrajectoryState> state(1);
  std::size_t bytes = 0;
  check(cub::DeviceSelect::Flagged(nullptr, bytes, ids.data, support.data,
                                   members.data, kept.data, count));
  Buffer<std::uint8_t> temporary(bytes);
  return cpu::clustering_trajectories(
      votes, context, options, rng,
      [&](const Descriptor &seed, bool reflection, double h2, int maximum) {
        TrajectoryState value{};
        value.center.dimension = seed.size();
        value.center.reflection = reflection;
        for (int k = 0; k < value.center.dimension; ++k)
          value.center.value[k] = seed[k];
        state.upload(&value, 1);
        do {
          begin_kernel();
          trajectory_batch<<<1, threads>>>(
              descriptors.data, kinds.data, count, h2, maximum, state.data,
              weights.data, support.data, ids.data);
          check(cudaGetLastError());
          end_kernel();
          state.download(&value, 1);
        } while (value.iterations < maximum && !value.converged &&
                 !value.exhausted);
        check(cub::DeviceSelect::Flagged(temporary.data, bytes, ids.data,
                                         support.data, members.data, kept.data,
                                         count));
        Offset size = 0;
        kept.download(&size, 1);
        if (size) {
          begin_kernel();
          gather_support<<<blocks(size), threads>>>(members.data, support.data,
                                                    size, counts.data);
          check(cudaGetLastError());
          end_kernel();
        }
        std::vector<Index> host_ids(size);
        std::vector<Offset> host_counts(size);
        members.download(host_ids.data(), size);
        counts.download(host_counts.data(), size);
        cpu::Trajectory result;
        result.mean = Descriptor::zero(seed.size());
        for (int k = 0; k < value.center.dimension; ++k)
          result.mean[k] = value.center.value[k];
        result.converged = value.converged;
        result.seed_exhausted = value.exhausted;
        for (Offset i = 0; i < size; ++i)
          result.support.emplace(host_ids[i], host_counts[i]);
        return result;
      });
}
} // namespace symmetry::cuda_detail
