// SPDX-License-Identifier: MPL-2.0
#include "symmetry/pipeline.h"
#include <iostream>
#include <stdexcept>
#include <type_traits>
using namespace symmetry;
namespace {
void check(bool condition) {
  if (!condition)
    throw std::runtime_error("Storage/backend contract failed");
}
template <class F> void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected);
}
Mesh tetrahedron() {
  Mesh m;
  for (Vec v : {Vec{0, 0, 0}, Vec{1, 0, 0}, Vec{0, 1, 0}, Vec{0, 0, 1}})
    m.vertices.push_back(v);
  for (auto f : {std::array<Index, 3>{0, 2, 1}, std::array<Index, 3>{0, 1, 3},
                 std::array<Index, 3>{0, 3, 2}, std::array<Index, 3>{1, 2, 3}})
    m.faces.push_back(f);
  return m;
}
class RecordingBackend final : public Backend {
  std::unique_ptr<Backend> cpu = make_cpu_backend();

public:
  mutable std::vector<std::string> calls;
  bool broken = false;
  const char *name() const noexcept override { return "recording-cpu"; }
  Features curvature(MeshView m, TopologyView t,
                     const Options &o) const override {
    calls.push_back("curvature");
    auto f = cpu->curvature(m, t, o);
    if (broken)
      f.kmax.pop_back();
    return f;
  }
  Selection select(MeshView m, FeaturesView f, ArrayView<const Scalar> w,
                   const Options &o, std::mt19937_64 &r) const override {
    calls.push_back("select");
    return cpu->select(m, f, w, o, r);
  }
  Pairs pairing(MeshView m, FeaturesView f, ArrayView<const Index> p,
                ModelContext c, const Options &o) const override {
    calls.push_back("pairing");
    return cpu->pairing(m, f, p, c, o);
  }
  Votes transformations(MeshView m, FeaturesView f, PairsView p, ModelContext c,
                        const Options &o) const override {
    calls.push_back("transformations");
    return cpu->transformations(m, f, p, c, o);
  }
  ClusterOutput clustering(VotesView v, ModelContext c, const Options &o,
                           std::mt19937_64 &r) const override {
    calls.push_back("clustering");
    return cpu->clustering(v, c, o, r);
  }
  Verification verification(MeshView m, TopologyView t, ClustersView c,
                            ModelContext ctx, const Options &o) const override {
    calls.push_back("verification");
    return cpu->verification(m, t, c, ctx, o);
  }
};
} // namespace
int main() {
  try {
    static_assert(std::is_same<Scalar, double>::value);
    static_assert(sizeof(Offset) == 8 && sizeof(Index) == 4);
    Points points;
    points.push_back({1, 2, 3});
    points.push_back({4, 5, 6});
    auto borrowed = points.as_view();
    check(borrowed.x.data == points.x.data() && borrowed[1].z == 6);
    points.set(0, {7, 8, 9});
    check(borrowed[0].y == 8); // mutation without reallocation is visible
    Points moved = std::move(points);
    check(moved.as_view()[0].x == 7);
    moved.y.pop_back();
    rejects([&] { moved.as_view(); });
    auto csr = Csr::from_rows({{0, 2}, {}, {1}});
    auto rows = csr.as_view();
    check(rows.size() == 3 && rows[0][0] == 0 && rows[1].empty() &&
          rows[2][0] == 1);
    check(csr.offsets == std::vector<Offset>({0, 2, 2, 3}));
    csr.offsets[2] = std::numeric_limits<Offset>::max();
    rejects([&] { csr.as_view(); });
    Features f;
    f.resize(1);
    f.set(0, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, .2, .8, true});
    check(f.as_view().normal.x.data == f.normal.x.data() && f[0].kmax == .8);
    Votes votes;
    votes.push_back({0,
                     true,
                     reflection_matrix({1, 0, 0}),
                     {2, 0, 0},
                     {1, 0, 0},
                     -1,
                     {1, 0, 0, -1}});
    auto vv = votes.as_view();
    check(vv.kind[0] == TransformKind::Reflection &&
          vv.descriptor_at(0).size() == 4);
    check(vv.rotation.component[0].data == votes.rotation.component[0].data());
    check(vv[0].rotation[0] == -1 && vv[0].translation.x == 2);
    for (int k = 4; k < 12; ++k)
      check(vv.descriptor.component[k][0] == 0);
    rejects(
        [&] { votes.push_back({1, false, identity(), {}, {}, 0, {1, 2, 3}}); });
    Clusters clusters;
    clusters.append(
        {true, reflection_matrix({1, 0, 0}), {2, 0, 0}, {1, 0, 0}, -1}, {0});
    clusters.verification = Verification::empty(1);
    clusters.verification.side_a = Csr::from_rows({{0, 2}});
    clusters.verification.patches_a = GroupedCsr{};
    clusters.verification.patches_a.append({{0}, {2}});
    clusters.verification.validate(1);
    auto cv = clusters[0];
    check(cv.votes[0] == 0 && cv.side_a.size() == 2 &&
          cv.patches_a.size() == 2 && cv.patches_a[1][0] == 2);
    clusters.verification.coverage.clear();
    rejects([&] { clusters.verification.validate(1); });
    Options oversized;
    oversized.max_votes =
        static_cast<Offset>(std::numeric_limits<Index>::max()) + 1;
    rejects([&] { validate_options(oversized); });
    auto m = tetrahedron();
    m.vertices.push_back({5, 5, 5});
    auto prepared = prepare(m, Options{});
    auto topology = prepared.topology.as_view();
    check(topology.neighbors.size() == 5 &&
          topology.incident_faces.size() == 5);
    check(topology.neighbors[4].empty() && topology.incident_faces[4].empty());
    check(topology.neighbors[0].size() == 3 && topology.neighbors[0][0] == 1);
    check(topology.incident_faces[0].size() == 3 &&
          topology.incident_faces[0][0] == 0);
    for (Offset vertex = 0; vertex < 4; ++vertex)
      for (Index face : topology.incident_faces[vertex]) {
        auto triangle = prepared.mesh.faces[face];
        check(std::find(triangle.begin(), triangle.end(), vertex) !=
              triangle.end());
      }
    RecordingBackend backend;
    std::vector<std::string> progress;
    Options options;
    options.sample_ratio = 1;
    auto result = detect(tetrahedron(), options, backend,
                         [&](const std::string &stage, double ms) {
                           check(ms >= 0);
                           progress.push_back(stage);
                         });
    check(backend.calls ==
          std::vector<std::string>({"curvature", "select", "pairing",
                                    "transformations", "clustering",
                                    "verification"}));
    check(progress.size() == 7 && progress.front() == "prepare" &&
          progress.back() == "verification");
    auto direct = detect(tetrahedron(), options);
    check(direct.sample == result.sample && direct.pruned == result.pruned &&
          direct.pairs.size() == result.pairs.size());
    backend.calls.clear();
    options.patching = false;
    detect(tetrahedron(), options, backend);
    check(backend.calls.size() == 5);
    backend.calls.clear();
    backend.broken = true;
    rejects([&] { detect(tetrahedron(), options, backend); });
    check(backend.calls.size() == 1);
    std::cout << "Storage/backend checks passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
