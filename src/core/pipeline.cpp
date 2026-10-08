// SPDX-License-Identifier: MPL-2.0
#include "symmetry/pipeline.h"
#include <chrono>
namespace symmetry {
Result detect(Mesh mesh, const Options &options, const Backend &backend,
              Progress progress) {
  auto clock = std::chrono::steady_clock::now();
  Result r;
  r.options = options;
  auto timed = [&](std::string stage) {
    auto now = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - clock).count();
    r.timings.push_back({stage, ms});
    if (progress)
      progress(stage, ms);
    clock = now;
  };
  auto prepared = prepare(std::move(mesh), options);
  static_cast<Diagnostics &>(r) = prepared.diagnostics;
  r.mesh = std::move(prepared.mesh);
  r.box_size = prepared.context.box_size;
  auto m = r.mesh.as_view();
  auto t = prepared.topology.as_view();
  auto context = prepared.context;
  std::mt19937_64 random(options.seed);
  timed("prepare");
  r.features = backend.curvature(m, t, options);
  require_size(r.features.size(), m.vertices.size());
  auto f = r.features.as_view();
  timed("curvature");
  auto selection = backend.select(m, f, t.weights, options, random);
  r.sample = std::move(selection.sample);
  r.pruned = std::move(selection.pruned);
  timed("sample_prune");
  r.pairs = backend.pairing(m, f, view(r.pruned), context, options);
  auto pairs = r.pairs.as_view();
  timed("pairing");
  r.votes = backend.transformations(m, f, pairs, context, options);
  auto votes = r.votes.as_view();
  timed("transformation");
  auto clustered = backend.clustering(votes, context, options, random);
  r.limited_iterations = clustered.limited_iterations;
  r.clusters = std::move(clustered.clusters);
  timed("mean_shift");
  if (options.patching)
    r.clusters.verification =
        backend.verification(m, t, r.clusters.as_view(), context, options);
  r.clusters.as_view();
  r.clusters.verification.validate(r.clusters.size());
  timed("verification");
  return r;
}
} // namespace symmetry
