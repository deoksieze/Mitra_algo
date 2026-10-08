// SPDX-License-Identifier: MPL-2.0
#include "symmetry/cuda.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace symmetry;
void need(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void close(double a, double b) {
  need(std::isfinite(a) && std::isfinite(b) &&
           std::abs(a - b) <= 2e-12 * (1 + std::abs(a)),
       "CPU/GPU numeric disagreement");
}
void compare(VotesView a, VotesView b, MeshView mesh, PairsView pairs) {
  need(a.size() == b.size(), "vote counts differ");
  for (Offset i = 0; i < a.size(); ++i) {
    Vote x = a[i], y = b[i];
    need(x.pair == y.pair && x.reflection == y.reflection,
         "vote order/type differs");
    for (int k = 0; k < 9; ++k)
      close(x.rotation[k], y.rotation[k]);
    for (int k = 0; k < 3; ++k) {
      close(x.translation[k], y.translation[k]);
      close(x.plane_normal[k], y.plane_normal[k]);
    }
    close(x.plane_offset, y.plane_offset);
    need(x.descriptor.size() == y.descriptor.size(),
         "descriptor dimension differs");
    for (Offset k = 0; k < x.descriptor.size(); ++k)
      close(x.descriptor[k], y.descriptor[k]);
    // Nondegenerate fixtures must define orthogonal motions mapping p to q.
    auto pair = pairs[y.pair];
    if (pair.a != pair.b) {
      auto gram = multiply(y.rotation, transpose(y.rotation));
      auto id = identity();
      for (int k = 0; k < 9; ++k)
        close(gram[k], id[k]);
      close(determinant(y.rotation), y.reflection ? -1 : 1);
      Vec mapped =
          transform_vector(y.rotation, mesh.vertices[pair.a]) + y.translation;
      for (int k = 0; k < 3; ++k)
        close(mapped[k], mesh.vertices[pair.b][k]);
      if (y.reflection)
        close(norm(y.plane_normal), 1);
    }
    if (y.reflection)
      for (int k = 4; k < 12; ++k)
        close(b.descriptor.component[k][i], 0);
  }
}
int main() {
  try {
    auto cpu = make_cpu_backend(), gpu = make_cuda_backend();
    Options o;
    o.sample_ratio = 1;
    Mesh mesh;
    Features f;
    // Oblique plane with nearly zero first normal component and opposite
    // normals.
    Vec n = unit({1e-16, 1, 2}), d = unit(cross(n, {1, 0, 0})), x = cross(n, d);
    Mat mirror = reflection_matrix(n);
    for (int i = 0; i < 514; ++i) {
      Vec p = {double(i) * .01, -2 - .001 * i, .13};
      mesh.vertices.push_back(p);
      mesh.vertices.push_back(transform_vector(mirror, p));
    }
    f.resize(mesh.vertices.size());
    for (Offset i = 0; i < f.size(); ++i) {
      bool reflected = i % 2;
      Vec normal = n, minimum = d;
      // Principal-frame sign ambiguity must retain the same two rigid
      // hypotheses.
      if (i % 4 == 3)
        minimum = minimum * -1;
      if (reflected) {
        normal = transform_vector(mirror, normal);
        minimum = transform_vector(mirror, minimum);
      }
      f.set(i, {normal, minimum, cross(normal, minimum), .2, 1, true});
    }
    Pairs pairs;
    for (int i = 0; i < 514; ++i) {
      pairs.push_back({2 * i, 2 * i + 1});
      pairs.push_back({2 * i + 1, 2 * i});
    }
    ModelContext ctx{20, {3, -1, 2}};
    auto a = cpu->transformations(mesh.as_view(), f.as_view(), pairs.as_view(),
                                  ctx, o);
    auto b = gpu->transformations(mesh.as_view(), f.as_view(), pairs.as_view(),
                                  ctx, o);
    need(a.size() == 3 * pairs.size(),
         "reflection fixtures must emit three votes per pair");
    compare(a.as_view(), b.as_view(), mesh.as_view(), pairs.as_view());
    auto again = gpu->transformations(mesh.as_view(), f.as_view(),
                                      pairs.as_view(), ctx, o);
    need(b.rotation.component == again.rotation.component &&
             b.descriptor.component == again.descriptor.component,
         "GPU repeat is not deterministic");
    // Mixed 2/3 counts exercise prefix offsets across several blocks.
    for (Offset i = 1; i < f.size(); i += 4) {
      f.normal.set(i, n);
      f.dmax.set(i, cross(n, f.dmin[i]));
    }
    a = cpu->transformations(mesh.as_view(), f.as_view(), pairs.as_view(), ctx,
                             o);
    b = gpu->transformations(mesh.as_view(), f.as_view(), pairs.as_view(), ctx,
                             o);
    need(b.size() > 2 * pairs.size() && b.size() < 3 * pairs.size(),
         "fixture needs mixed counts");
    compare(a.as_view(), b.as_view(), mesh.as_view(), pairs.as_view());
    o.max_votes = b.size();
    need(gpu->transformations(mesh.as_view(), f.as_view(), pairs.as_view(), ctx,
                              o)
                 .size() == b.size(),
         "exact vote budget rejected");
    o.max_votes = 600000;
    // Parallel normals across the plane suppress pure reflection, leaving rigid
    // votes.
    for (Offset i = 0; i < f.size(); ++i)
      f.set(i, {n, d, x, .2, 1, true});
    a = cpu->transformations(mesh.as_view(), f.as_view(), pairs.as_view(), ctx,
                             o);
    b = gpu->transformations(mesh.as_view(), f.as_view(), pairs.as_view(), ctx,
                             o);
    need(b.size() == 2 * pairs.size(),
         "unexpected reflection for incompatible normals");
    compare(a.as_view(), b.as_view(), mesh.as_view(), pairs.as_view());
    Pairs same;
    same.push_back({0, 0}); // Zero displacement follows CPU reference exactly.
    a = cpu->transformations(mesh.as_view(), f.as_view(), same.as_view(), ctx,
                             o);
    b = gpu->transformations(mesh.as_view(), f.as_view(), same.as_view(), ctx,
                             o);
    compare(a.as_view(), b.as_view(), mesh.as_view(), same.as_view());
    f.set(0, {{}, {}, {}, 0, 0, false}); // Zero directions must remain finite
                                         // and match CPU.
    a = cpu->transformations(mesh.as_view(), f.as_view(), same.as_view(), ctx,
                             o);
    b = gpu->transformations(mesh.as_view(), f.as_view(), same.as_view(), ctx,
                             o);
    compare(a.as_view(), b.as_view(), mesh.as_view(), same.as_view());
    std::vector<double> weights(f.size(), 1);
    for (Offset i = 0; i < f.size(); ++i) {
      f.valid[i] = (i % 7) != 0;
      f.kmin[i] = (i % 5 == 0 ? 1 : .2);
      f.kmax[i] = 1;
      if (i % 11 == 0)
        f.kmin[i] = f.kmax[i] = 1e-12;
      if (i % 13 == 0)
        weights[i] = 0;
    }
    for (double ratio : {1.0, .5, .001}) {
      o.sample_ratio = ratio;
      std::mt19937_64 r1(42), r2(42);
      auto s1 = cpu->select(mesh.as_view(), f.as_view(), view(weights), o, r1);
      auto s2 = gpu->select(mesh.as_view(), f.as_view(), view(weights), o, r2);
      need(s1.sample == s2.sample && s1.pruned == s2.pruned && r1 == r2,
           "sampling/pruning/RNG differs");
    }
    // Strict pruning threshold, zero curvature, and signed curvatures.
    o.prune = .75;
    o.sample_ratio = 1;
    for (Offset i = 0; i < f.size(); ++i) {
      f.valid[i] = 1;
      f.kmax[i] = 1;
      f.kmin[i] =
          (i % 3 == 0 ? .75 : (i % 3 == 1 ? std::nextafter(.75, 0.0) : -1));
      weights[i] = 1;
    }
    std::mt19937_64 boundary1(42), boundary2(42);
    auto boundary_cpu =
        cpu->select(mesh.as_view(), f.as_view(), view(weights), o, boundary1);
    auto boundary_gpu =
        gpu->select(mesh.as_view(), f.as_view(), view(weights), o, boundary2);
    need(boundary_cpu.pruned == boundary_gpu.pruned &&
             boundary_cpu.pruned.size() == f.size() * 2 / 3,
         "strict pruning threshold differs");
    for (Offset i = 0; i < f.size(); ++i)
      f.valid[i] = 0;
    std::mt19937_64 rng(42);
    need(gpu->select(mesh.as_view(), f.as_view(), view(weights), o, rng)
             .pruned.empty(),
         "empty compaction failed");
    for (Offset i = 0; i < f.size(); ++i) {
      f.valid[i] = 1;
      f.kmin[i] = 0;
      f.kmax[i] = 1;
      weights[i] = 1;
    }
    need(gpu->select(mesh.as_view(), f.as_view(), view(weights), o, rng)
                 .pruned.size() == f.size(),
         "full compaction failed");
    std::fill(weights.begin(), weights.end(), 0);
    need(gpu->select(mesh.as_view(), f.as_view(), view(weights), o, rng)
             .sample.empty(),
         "empty sample failed");
    Mesh empty;
    Features ef;
    std::vector<double> ew;
    Pairs ep;
    need(gpu->select(empty.as_view(), ef.as_view(), view(ew), o, rng)
             .sample.empty(),
         "empty mesh failed");
    need(
        gpu->transformations(empty.as_view(), ef.as_view(), ep.as_view(), {}, o)
                .size() == 0,
        "empty pairs failed");
    auto fails = [&](ModelContext c, PairsView p, const Options &options) {
      try {
        gpu->transformations(mesh.as_view(), f.as_view(), p, c, options);
      } catch (const std::exception &) {
        return;
      }
      throw std::runtime_error("invalid input/budget was not rejected");
    };
    o.max_votes = 1;
    fails(ctx, pairs.as_view(), o);
    o.max_votes = 600000;
    fails({0, {}}, pairs.as_view(), o);
    fails({NAN, {}}, pairs.as_view(), o);
    Pairs bad;
    bad.push_back({-1, 0});
    fails(ctx, bad.as_view(), o);
    bad.a[0] = static_cast<Index>(f.size());
    fails(ctx, bad.as_view(), o);
    std::cout << "CUDA contract passed: stable compaction, counts/scan/fill, "
                 "CPU agreement, geometry, empty inputs, budgets\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
