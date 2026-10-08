// SPDX-License-Identifier: MPL-2.0
#include "../cpu/internal.h"
#include "internal.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_select.cuh>
namespace symmetry::cuda_detail {
namespace {
struct Center {
  double value[12];
  int dimension;
  bool reflection;
};
__global__ void neighborhood(const double *descriptors,
                             const std::uint8_t *kinds, Offset count,
                             Center center, double h2, double *weights,
                             std::uint8_t *flags, Index *ids) {
  Offset i = static_cast<Offset>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  ids[i] = static_cast<Index>(i);
  double distance = 0;
  if ((kinds[i] != 0) == center.reflection) {
    for (int k = 0; k < center.dimension; ++k) {
      double d = center.value[k] - descriptors[k * count + i];
      distance += d * d;
    }
    flags[i] = distance < h2;
  } else
    flags[i] = 0;
  weights[i] = flags[i] ? 1 - distance / h2 : 0;
}
// Ordered per-component accumulation reproduces CPU arithmetic. Parallel
// tree sums can perturb exact nearest-face ties in downstream verification.
__global__ void ordered_sums(const double *descriptors,const double *weights,
                             const std::uint8_t *flags,Offset count,int dimension,
                             double *total) {
  unsigned k=threadIdx.x;
  if(k>static_cast<unsigned>(dimension)) return;
  double sum=0;
  for(Offset i=0;i<count;++i) if(flags[i])
    sum += k ? weights[i]*descriptors[(k-1)*count+i] : weights[i];
  total[k]=sum;
}

} // namespace
ClusterOutput clustering(VotesView votes, ModelContext context,
                         const Options &options, std::mt19937_64 &rng) {
  Offset count = votes.size();
  if (!count)
    return cpu::clustering(votes, context, options, rng);
  Buffer<double> descriptors(12 * count), weights(count),
      total(13);
  for (int k = 0; k < 12; ++k)
    descriptors.upload(votes.descriptor.component[k].data, count, k * count);
  Buffer<std::uint8_t> kinds(count), flags(count);
  std::vector<std::uint8_t> types(count);
  for (Offset i = 0; i < count; ++i)
    types[i] = votes.is_reflection(i);
  kinds.upload(types.data(), count);
  Buffer<Index> ids(count), members(count);
  Buffer<Offset> kept(1);
  std::size_t bytes = 0;
  check(cub::DeviceSelect::Flagged(nullptr, bytes, ids.data, flags.data,
                                   members.data, kept.data, count));
  Buffer<std::uint8_t> temporary(bytes);
  return cpu::clustering_with(
      votes, context, options, rng,
      [&](const Descriptor &mean, bool reflection, double h2) {
        Center center{};
        center.dimension = mean.size();
        center.reflection = reflection;
        for (int k = 0; k < center.dimension; ++k)
          center.value[k] = mean[k];
        neighborhood<<<blocks(count), threads>>>(
            descriptors.data, kinds.data, count, center, h2, weights.data,
            flags.data, ids.data);
        check(cudaGetLastError());
        ordered_sums<<<1,32>>>(descriptors.data,weights.data,flags.data,count,
                                center.dimension,total.data);
        check(cudaGetLastError());
        check(cub::DeviceSelect::Flagged(temporary.data, bytes, ids.data,
                                         flags.data, members.data, kept.data,
                                         count));
        double sums[13]{};
        total.download(sums, center.dimension + 1);
        Offset size = 0;
        kept.download(&size, 1);
        cpu::Neighborhood result;
        result.weight = sums[0];
        result.weighted_sum = Descriptor::zero(center.dimension);
        for (int k = 0; k < center.dimension; ++k)
          result.weighted_sum[k] = sums[k + 1];
        result.members.resize(size);
        members.download(result.members.data(), size);
        return result;
      });
}
} // namespace symmetry::cuda_detail
