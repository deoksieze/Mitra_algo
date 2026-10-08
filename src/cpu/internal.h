// SPDX-License-Identifier: MPL-2.0
#pragma once
#include "symmetry/pipeline.h"
namespace symmetry::cpu {
Features curvature(MeshView, TopologyView, const Options &);
Selection select(MeshView, FeaturesView, ArrayView<const Scalar>,
                 const Options &, std::mt19937_64 &);
Pairs pairing(MeshView, FeaturesView, ArrayView<const Index>, ModelContext,
              const Options &);
Votes transformations(MeshView, FeaturesView, PairsView, ModelContext,
                      const Options &);
ClusterOutput clustering(VotesView, ModelContext, const Options &,
                         std::mt19937_64 &);
Verification verification(MeshView, TopologyView, ClustersView, ModelContext,
                          const Options &);
} // namespace symmetry::cpu
