// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/options.h"
#include "symmetry/storage.h"
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
namespace symmetry {
struct Edge {
  Index lo, hi, a, b, face0, face1;
};
struct TopologyView {
  PointsView normals, face_normals;
  ArrayView<const Scalar> weights;
  CsrView neighbors, incident_faces;
  ArrayView<const Edge> edges;
};
struct Topology {
  Points normals, face_normals;
  std::vector<Scalar> weights;
  Csr neighbors, incident_faces;
  std::vector<Edge> edges;
  TopologyView as_view() const {
    return {normals.as_view(),   face_normals.as_view(),   view(weights),
            neighbors.as_view(), incident_faces.as_view(), view(edges)};
  }
};
struct ModelContext {
  Scalar box_size = 0;
  Vec center;
};
struct Diagnostics {
  int rejected_faces = 0, nonmanifold_edges = 0, orientation_conflicts = 0,
      limited_iterations = 0;
};
struct Prepared {
  Mesh mesh;
  Topology topology;
  ModelContext context;
  Diagnostics diagnostics;
};
Prepared prepare(Mesh mesh, const Options &options);
struct Selection {
  std::vector<Index> sample, pruned;
};
struct ClusterOutput {
  Clusters clusters;
  int limited_iterations = 0;
};
// All views passed here refer to HOST memory. CUDA owns its device allocations
// and transfers and returns completed host-owned results. RNG stays on host.
class Backend {
public:
  virtual ~Backend() = default;
  virtual const char *name() const noexcept = 0;
  virtual Features curvature(MeshView, TopologyView, const Options &) const = 0;
  virtual Selection select(MeshView, FeaturesView, ArrayView<const Scalar>,
                           const Options &, std::mt19937_64 &) const = 0;
  virtual Pairs pairing(MeshView, FeaturesView, ArrayView<const Index>,
                        ModelContext, const Options &) const = 0;
  virtual Votes transformations(MeshView, FeaturesView, PairsView, ModelContext,
                                const Options &) const = 0;
  virtual ClusterOutput clustering(VotesView, ModelContext, const Options &,
                                   std::mt19937_64 &) const = 0;
  virtual Verification verification(MeshView, TopologyView, ClustersView,
                                    ModelContext, const Options &) const = 0;
};
std::unique_ptr<Backend> make_cpu_backend();
struct Result : Diagnostics {
  Mesh mesh;
  Options options;
  Scalar box_size = 0;
  Features features;
  std::vector<Index> sample, pruned;
  Pairs pairs;
  Votes votes;
  Clusters clusters;
  std::vector<std::pair<std::string, double>> timings;
};
using Progress = std::function<void(const std::string &, double)>;
Result detect(Mesh mesh, const Options &options, const Backend &backend,
              Progress progress = {});
Result detect(Mesh mesh, const Options &options);
} // namespace symmetry
