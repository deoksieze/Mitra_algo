// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/pipeline.h"
#include <map>
namespace symmetry::cpu {
Features curvature(MeshView, TopologyView, const Options &);
Selection select(MeshView, FeaturesView, ArrayView<const Scalar>,
                 const Options &, std::mt19937_64 &);
Pairs pairing(MeshView, FeaturesView, ArrayView<const Index>, ModelContext,
              const Options &);
Votes transformations(MeshView, FeaturesView, PairsView, ModelContext,
                      const Options &);
struct Neighborhood {
  Descriptor weighted_sum;
  double weight = 0;
  std::vector<Index> members;
};
struct Trajectory {
  Descriptor mean;
  std::map<Index, std::uint64_t> support;
  bool converged = false, seed_exhausted = false;
};
using TrajectoryEvaluator =
    std::function<Trajectory(const Descriptor &, bool, double, int)>;
ClusterOutput clustering_trajectories(VotesView, ModelContext, const Options &,
                                      std::mt19937_64 &,
                                      const TrajectoryEvaluator &);
using NeighborhoodEvaluator =
    std::function<Neighborhood(const Descriptor &, bool, double)>;
ClusterOutput clustering_with(VotesView, ModelContext, const Options &,
                              std::mt19937_64 &, const NeighborhoodEvaluator &);
ClusterOutput clustering(VotesView, ModelContext, const Options &,
                         std::mt19937_64 &);
Verification verification(MeshView, TopologyView, ClustersView, ModelContext,
                          const Options &);
} // namespace symmetry::cpu
