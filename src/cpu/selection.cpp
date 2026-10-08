// SPDX-License-Identifier: MPL-2.0
#include "internal.h"
#include "symmetry/pipeline.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>

namespace symmetry::cpu {
Selection select(MeshView m, FeaturesView features,
                 ArrayView<const Scalar> weights, const Options &options,
                 std::mt19937_64 &random) {
  Selection result;
  // Weighted sampling without replacement using exponential keys; fixed RNG
  // seed.
  std::vector<std::pair<double, int>> keys;
  for (std::size_t i = 0; i < m.vertices.size(); ++i)
    if (weights[i] > 0) {
      double u =
          (static_cast<double>(random() >> 11) + 0.5) / 9007199254740992.0;
      keys.push_back({-std::log(u) / weights[i], static_cast<int>(i)});
    }
  std::sort(keys.begin(), keys.end());
  std::size_t target =
      std::min(keys.size(), static_cast<std::size_t>(std::ceil(
                                m.vertices.size() * options.sample_ratio)));
  for (std::size_t i = 0; i < target; ++i)
    result.sample.push_back(keys[i].second);
  std::sort(result.sample.begin(), result.sample.end());
  for (int i : result.sample) {
    auto f = features[i];
    double maximum = std::max(std::abs(f.kmin), std::abs(f.kmax));
    if (f.valid && maximum > 1e-10 &&
        std::abs(f.kmin - f.kmax) / maximum > 1 - options.prune)
      result.pruned.push_back(i);
  }
  return result;
}

} // namespace symmetry::cpu
