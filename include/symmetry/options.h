// SPDX-License-Identifier: MPL-2.0
#pragma once
#include <cstdint>
namespace symmetry {
struct Options {
  double sample_ratio = 0.5, prune = 0.85, pair_threshold = 0.15;
  double bandwidth = 0.08, verify_threshold = 0.025;
  int smoothing = 3, subdivide = 0, max_iterations = 100, max_clusters = 20;
  std::uint64_t seed = 42, max_pairs = 200000, max_votes = 600000;
  bool patching = true;
};
void validate_options(const Options &options);
} // namespace symmetry
