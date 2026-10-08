// SPDX-License-Identifier: MPL-2.0
#include "symmetry/sampling.h"
#include <cmath>
namespace symmetry {
std::vector<Index> sample_vertices(ArrayView<const Scalar> weights,
                                   double ratio, std::mt19937_64 &random) {
  std::vector<Index> sample;
  // Weighted sampling without replacement using exponential keys; fixed RNG
  // seed.
  std::vector<std::pair<double, int>> keys;
  for (std::size_t i = 0; i < weights.size(); ++i)
    if (weights[i] > 0) {
      double u =
          (static_cast<double>(random() >> 11) + 0.5) / 9007199254740992.0;
      keys.push_back({-std::log(u) / weights[i], static_cast<int>(i)});
    }
  std::sort(keys.begin(), keys.end());
  std::size_t target = std::min(
      keys.size(), static_cast<std::size_t>(std::ceil(weights.size() * ratio)));
  for (std::size_t i = 0; i < target; ++i)
    sample.push_back(keys[i].second);
  std::sort(sample.begin(), sample.end());
  return sample;
}
} // namespace symmetry
