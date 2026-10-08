// SPDX-License-Identifier: MPL-2.0
#include "internal.h"
#include "symmetry/pipeline.h"
#include "symmetry/sampling.h"
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
  require_size(weights.size(), m.vertices.size());
  result.sample = sample_vertices(weights, options.sample_ratio, random);
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
