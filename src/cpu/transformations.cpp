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
static Mat frame(Vec a, Vec b, Vec c) {
  return {a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z};
}
Votes transformations(MeshView mesh, FeaturesView features, PairsView pairs,
                      ModelContext context, const Options &options) {
  double box = context.box_size;
  Vec center = context.center;
  Votes result;
  auto add = [&](Vote vote) {
    if (result.size() >= options.max_votes)
      throw std::runtime_error("Vote budget exceeded; lower --sample-ratio or "
                               "tighten --pair-threshold");
    result.push_back(std::move(vote));
  };
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    auto pair = pairs[i];
    Vec p = mesh.vertices[pair.a], q = mesh.vertices[pair.b];
    auto a = features[pair.a], b = features[pair.b];
    // A pure mirror plane maps p to q; signs of principal directions are
    // irrelevant.
    Vec n = unit(q - p);
    // Prefer the dominant component: tiny numerical x components must not
    // reverse a plane whose normal is nearly parallel to the y/z axes.
    int dominant = 0;
    for (int k = 1; k < 3; ++k)
      if (std::abs(n[k]) > std::abs(n[dominant]))
        dominant = k;
    if (n[dominant] < 0)
      n = n * -1;
    double d = -dot(n, (p + q) / 2);
    Mat mirror = reflection_matrix(n);
    if (dot(transform_vector(mirror, a.normal), b.normal) > 0.75 &&
        std::abs(dot(transform_vector(mirror, a.dmin), b.dmin)) > 0.75) {
      Vote vote{static_cast<int>(i),
                true,
                mirror,
                n * (-2 * d),
                n,
                d,
                {n.x, n.y, n.z, (d + dot(n, center)) / box}};
      add(std::move(vote));
    }
    Mat fa = frame(a.dmin, a.dmax, a.normal);
    for (double sign : {1.0, -1.0}) {
      Mat fb = frame(b.dmin * sign, b.dmax * sign, b.normal);
      Mat rotation = multiply(fb, transpose(fa));
      Vec t = q - transform_vector(rotation, p);
      Vec centered = (transform_vector(rotation, center) + t - center) / box;
      std::vector<double> desc;
      for (double x : rotation)
        desc.push_back(x / 2);
      desc.push_back(centered.x);
      desc.push_back(centered.y);
      desc.push_back(centered.z);
      add(Vote{
          static_cast<int>(i), false, rotation, t, {}, 0, std::move(desc)});
    }
  }
  return result;
}

} // namespace symmetry::cpu
