// SPDX-License-Identifier: MPL-2.0
#include "../src/cuda/test_support.h"
#include "symmetry/cuda.h"
#include "symmetry/io.h"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace symmetry;
void need(bool b, const char *m) {
  if (!b)
    throw std::runtime_error(m);
}
int main(int argc, char **argv) {
  try {
    need(argc == 2, "repository root required");
    auto cpu = make_cpu_backend(), gpu = make_cuda_backend();
    for (const char *name : {"reflection", "rigid", "partial"}) {
      Options o;
      o.verify_threshold = .01;
      auto p = prepare(read_mesh(std::filesystem::path(argv[1]) / "examples" /
                                 (std::string(name) + ".off")),
                       o);
      ClusterRecord c{};
      if (std::string(name) == "rigid") {
        double angle = .65;
        c.rotation = {std::cos(angle),
                      -std::sin(angle),
                      0,
                      std::sin(angle),
                      std::cos(angle),
                      0,
                      0,
                      0,
                      1};
        c.translation = {3.2, .3, .2};
      } else {
        c.reflection = true;
        c.rotation = reflection_matrix({1, 0, 0});
        c.plane_normal = {1, 0, 0};
      }
      Clusters clusters;
      clusters.append(c, {});
      // A stationary transform must yield empty verification as on CPU.
      ClusterRecord stationary{};
      stationary.rotation = identity();
      clusters.append(stationary, {});
      auto a = cpu->verification(p.mesh.as_view(), p.topology.as_view(),
                                 clusters.as_view(), p.context, o);
      auto b = gpu->verification(p.mesh.as_view(), p.topology.as_view(),
                                 clusters.as_view(), p.context, o);
      auto again = gpu->verification(p.mesh.as_view(), p.topology.as_view(),
                                     clusters.as_view(), p.context, o);
      need(a.side_a.indices == b.side_a.indices &&
               a.side_b.indices == b.side_b.indices,
           "verification membership differs");
      need(a.side_a.offsets == b.side_a.offsets &&
               a.side_b.offsets == b.side_b.offsets,
           "verification offsets differ");
      need(a.patches_a.rows.indices == b.patches_a.rows.indices &&
               a.patches_b.rows.indices == b.patches_b.rows.indices,
           "patch members differ");
      need(a.patches_a.groups == b.patches_a.groups &&
               a.patches_a.rows.offsets == b.patches_a.rows.offsets &&
               a.patches_b.rows.offsets == b.patches_b.rows.offsets,
           "patch groups differ");
      need(b.side_a.indices == again.side_a.indices &&
               b.patches_a.rows.indices == again.patches_a.rows.indices,
           "verification repeat differs");
      need(!b.side_a[0].empty() && b.side_a[0][0] == 0, "vertex zero lost");
      need(b.side_a[1].empty() && b.side_b[1].empty(),
           "stationary points accepted");
      for (Offset i = 0; i < a.coverage.size(); ++i) {
        need(a.coverage[i] == b.coverage[i], "coverage differs");
        need(std::abs(a.rms[i] - b.rms[i]) < 1e-12 * p.context.box_size,
             "rms differs");
      }
    }
    std::vector<cuda_detail::TriangleQuery> queries;
    std::mt19937_64 rng(12);
    std::uniform_real_distribution<double> d(-3, 3);
    for (int i = 0; i < 1024; ++i) {
      auto v = [&]() { return Vec{d(rng), d(rng), d(rng)}; };
      cuda_detail::TriangleQuery q{v(), v(), v(), v()};
      if (i % 5 == 0)
        q.c = q.a;
      if (i % 7 == 0)
        q.b = q.c = q.a;
      if (i % 11 == 0)
        q.p = q.a;
      queries.push_back(q);
    }
    auto distances = cuda_detail::triangle_distances(queries);
    for (Offset i = 0; i < queries.size(); ++i) {
      auto q = queries[i];
      double reference = point_triangle_distance2(q.p, q.a, q.b, q.c);
      need(std::isfinite(distances[i]) &&
               std::abs(reference - distances[i]) <= 1e-12 * (1 + reference),
           "GPU triangle distance differs");
    }
    need(cuda_detail::triangle_distances({}).empty(),
         "empty triangle query failed");
    std::cout << "CUDA verification passed: forward/inverse, vertex zero, "
                 "patches, BVH, stationary, degenerate triangles\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
