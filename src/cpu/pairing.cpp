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
Pairs pairing(MeshView mesh, FeaturesView features,
              ArrayView<const Index> selected, ModelContext context,
              const Options &options) {
  Pairs result;
  for (std::size_t a = 0; a < selected.size(); ++a)
    for (std::size_t b = a + 1; b < selected.size(); ++b) {
      int i = selected[a], j = selected[b];
      auto p = features[i], q = features[j];
      double magnitude = std::max({std::abs(p.kmin), std::abs(p.kmax),
                                   std::abs(q.kmin), std::abs(q.kmax), 1e-10});
      double tolerance = options.pair_threshold * magnitude;
      if (std::abs(p.kmin - q.kmin) > tolerance ||
          std::abs(p.kmax - q.kmax) > tolerance ||
          norm(mesh.vertices[i] - mesh.vertices[j]) < 0.03 * context.box_size)
        continue;
      if (result.size() >= options.max_pairs)
        throw std::runtime_error("Pair budget exceeded; lower --sample-ratio "
                                 "or tighten --pair-threshold");
      result.push_back({i, j});
    }
  return result;
}

} // namespace symmetry::cpu
