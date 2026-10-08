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
namespace {
constexpr double eps = 1e-14;
struct UnpackedCluster : ClusterRecord {
  std::vector<Index> votes;
};
} // namespace
static Vec column(const Mat &m, int i) { return {m[i], m[3 + i], m[6 + i]}; }
static Mat frame(Vec a, Vec b, Vec c) {
  return {a.x, b.x, c.x, a.y, b.y, c.y, a.z, b.z, c.z};
}
double descriptor_distance(const Descriptor &a, const Descriptor &b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    double d = a[i] - b[i];
    s += d * d;
  }
  return s;
}
ClusterOutput clustering_trajectories(VotesView votes, ModelContext context,
                                      const Options &options,
                                      std::mt19937_64 &random,
                                      const TrajectoryEvaluator &evaluate) {
  std::vector<UnpackedCluster> clusters;
  int limited_iterations = 0;
  double h = options.bandwidth, h2 = h * h;
  std::vector<Descriptor> centers;
  std::vector<bool> types;
  std::vector<std::map<int, std::uint64_t>> counters;
  std::vector<bool> visited(votes.size());
  std::size_t support_entries = 0;
  while (true) {
    std::vector<int> seeds;
    for (std::size_t i = 0; i < visited.size(); ++i)
      if (!visited[i])
        seeds.push_back(i);
    if (seeds.empty())
      break;
    int seed = seeds[random() % seeds.size()];
    bool reflection = votes.is_reflection(seed);
    auto mean = votes.descriptor_at(seed);
    auto trajectory = evaluate(mean, reflection, h2, options.max_iterations);
    mean = trajectory.mean;
    auto support = std::move(trajectory.support);
    for (const auto &entry : support)
      visited[entry.first] = true;
    if (trajectory.seed_exhausted)
      visited[seed] = true;
    bool converged = trajectory.converged;
    if (!converged)
      ++limited_iterations;
    int merge = -1;
    for (std::size_t c = 0; c < centers.size(); ++c)
      if (types[c] == reflection &&
          descriptor_distance(mean, centers[c]) < h2 / 4) {
        merge = c;
        break;
      }
    if (merge < 0) {
      centers.push_back(mean);
      types.push_back(reflection);
      support_entries += support.size();
      counters.push_back(std::move(support));
    } else {
      double old = 0, now = 0;
      old = counters[merge].size();
      now = support.size();
      if (old + now > 0)
        for (std::size_t k = 0; k < mean.size(); ++k)
          centers[merge][k] =
              (old * centers[merge][k] + now * mean[k]) / (old + now);
      std::size_t before = counters[merge].size();
      for (const auto &entry : support)
        counters[merge][entry.first] += entry.second;
      support_entries += counters[merge].size() - before;
    }
    // Sparse counters avoid the historical C x P dense matrix.
    if (support_entries > 5000000)
      throw std::runtime_error(
          "Mean-shift support budget exceeded; lower --sample-ratio");
    if (centers.size() > 20000)
      throw std::runtime_error("Too many mean-shift modes; increase "
                               "--bandwidth or lower --sample-ratio");
  }
  std::vector<std::vector<int>> members(centers.size());
  std::vector<std::uint64_t> best(votes.size());
  std::vector<int> winners(votes.size(), -1);
  for (std::size_t c = 0; c < counters.size(); ++c)
    for (const auto &entry : counters[c])
      if (entry.second > best[entry.first]) {
        best[entry.first] = entry.second;
        winners[entry.first] = c;
      }
  for (std::size_t i = 0; i < winners.size(); ++i)
    if (winners[i] >= 0)
      members[winners[i]].push_back(i);
  Vec origin = context.center;
  for (std::size_t c = 0; c < centers.size(); ++c) {
    if (members[c].empty())
      continue;
    UnpackedCluster out{};
    out.reflection = types[c];
    out.votes = std::move(members[c]);
    auto descriptor = centers[c];
    if (out.reflection) {
      Vec n = {descriptor[0], descriptor[1], descriptor[2]};
      double length = norm(n);
      out.plane_normal = unit(n);
      out.plane_offset = descriptor[3] * context.box_size / length -
                         dot(out.plane_normal, origin);
      out.rotation = reflection_matrix(out.plane_normal);
      out.translation = out.plane_normal * (-2 * out.plane_offset);
    } else {
      Mat m{};
      for (int k = 0; k < 9; ++k)
        m[k] = descriptor[k] * 2;
      Vec a = unit(column(m, 0)),
          b = unit(column(m, 1) - a * dot(a, column(m, 1)));
      if (norm(a) < 0.5 || norm(b) < 0.5)
        m = votes[out.votes[0]].rotation;
      else
        m = frame(a, b, cross(a, b));
      out.rotation = m;
      Vec t{descriptor[9], descriptor[10], descriptor[11]};
      out.translation =
          t * context.box_size + origin - transform_vector(m, origin);
    }
    clusters.push_back(std::move(out));
  }
  std::stable_sort(clusters.begin(), clusters.end(),
                   [](const UnpackedCluster &a, const UnpackedCluster &b) {
                     return a.votes.size() > b.votes.size();
                   });
  if (clusters.size() > static_cast<std::size_t>(options.max_clusters))
    clusters.resize(options.max_clusters);
  ClusterOutput result;
  result.limited_iterations = limited_iterations;
  for (const auto &c : clusters)
    result.clusters.append(static_cast<const ClusterRecord &>(c), c.votes);
  result.clusters.verification = Verification::empty(result.clusters.size());
  return result;
}

ClusterOutput clustering_with(VotesView votes, ModelContext context,
                              const Options &options, std::mt19937_64 &random,
                              const NeighborhoodEvaluator &evaluate) {
  return clustering_trajectories(
      votes, context, options, random,
      [&](const Descriptor &seed, bool reflection, double h2,
          int max_iterations) {
        Trajectory result;
        result.mean = seed;
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
          auto neighborhood = evaluate(result.mean, reflection, h2);
          for (Index j : neighborhood.members)
            ++result.support[j];
          if (neighborhood.weight <= eps) {
            result.seed_exhausted = true;
            break;
          }
          for (double &value : neighborhood.weighted_sum)
            value /= neighborhood.weight;
          double movement =
              descriptor_distance(result.mean, neighborhood.weighted_sum);
          result.mean = neighborhood.weighted_sum;
          if (movement < 1e-6 * h2) {
            result.converged = true;
            break;
          }
        }
        return result;
      });
}
ClusterOutput clustering(VotesView votes, ModelContext context,
                         const Options &options, std::mt19937_64 &random) {
  return clustering_with(
      votes, context, options, random,
      [&](const Descriptor &mean, bool reflection, double h2) {
        Neighborhood result;
        result.weighted_sum = Descriptor::zero(mean.size());
        for (Offset j = 0; j < votes.size(); ++j) {
          if (votes.is_reflection(j) != reflection)
            continue;
          auto descriptor = votes.descriptor_at(j);
          double distance = descriptor_distance(mean, descriptor);
          if (distance < h2) {
            double w = 1 - distance / h2;
            result.weight += w;
            result.members.push_back(j);
            for (Offset k = 0; k < mean.size(); ++k)
              result.weighted_sum[k] += w * descriptor[k];
          }
        }
        return result;
      });
}

} // namespace symmetry::cpu
