// SPDX-License-Identifier: MPL-2.0
#include "internal.h"
namespace symmetry {
namespace {
class CpuBackend final : public Backend {
public:
  const char *name() const noexcept override { return "cpu"; }
  Features curvature(MeshView m, TopologyView t,
                     const Options &o) const override {
    return cpu::curvature(m, t, o);
  }
  Selection select(MeshView m, FeaturesView f, ArrayView<const Scalar> w,
                   const Options &o, std::mt19937_64 &rng) const override {
    return cpu::select(m, f, w, o, rng);
  }
  Pairs pairing(MeshView m, FeaturesView f, ArrayView<const Index> s,
                ModelContext c, const Options &o) const override {
    return cpu::pairing(m, f, s, c, o);
  }
  Votes transformations(MeshView m, FeaturesView f, PairsView p, ModelContext c,
                        const Options &o) const override {
    return cpu::transformations(m, f, p, c, o);
  }
  ClusterOutput clustering(VotesView v, ModelContext c, const Options &o,
                           std::mt19937_64 &rng) const override {
    return cpu::clustering(v, c, o, rng);
  }
  Verification verification(MeshView m, TopologyView t, ClustersView c,
                            ModelContext ctx, const Options &o) const override {
    return cpu::verification(m, t, c, ctx, o);
  }
};
} // namespace
std::unique_ptr<Backend> make_cpu_backend() {
  return std::make_unique<CpuBackend>();
}
Result detect(Mesh mesh, const Options &options) {
  auto cpu = make_cpu_backend();
  return detect(std::move(mesh), options, *cpu);
}
} // namespace symmetry
