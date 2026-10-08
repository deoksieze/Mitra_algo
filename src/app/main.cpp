// SPDX-License-Identifier: MPL-2.0
#include "symmetry/io.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
static void help() {
  std::cout
      << R"(CPU partial symmetry detector (rigid motion + plane reflection, scale=1)
Usage: symmetry_cpu_cli MODEL.off|MODEL.obj --output DIRECTORY [options]
  --sample-ratio X      (0,1], default 0.5, area-weighted vertex sampling
  --prune X             (0,1), default 0.85, larger retains more vertices
  --pair-threshold X    (0,1], default 0.15, curvature tolerance
  --bandwidth X         (0,2], default 0.08, normalized mean-shift radius
  --verify-threshold X  (0,0.5), default 0.025, fraction of mesh box diagonal
  --smoothing N         0..1000, default 3, tensor smoothing iterations
  --subdivide N         0..3, default 0, uniform midpoint subdivision
  --seed N              default 42, reproducible sampling and mean shift
  --max-iterations N    1..10000, default 100 per mean-shift trajectory
  --max-clusters N      1..1000, default 20 exported/verified hypotheses
  --max-pairs N         default 200000, fail explicitly at resource limit
  --max-votes N         default 600000, fail explicitly at resource limit
  --no-patching         export hypotheses without geometric verification
  --help
Requires consistently oriented manifold triangles. Boundaries are supported.
Polygon faces are fan-triangulated (use convex polygons or triangles).
Output directory must be new or empty. No MATLAB/libigl/CUDA required.
)";
}
int main(int argc, char **argv) {
  try {
    if (argc == 1) {
      help();
      return 2;
    }
    symmetry::Options o;
    std::filesystem::path input, out;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--help") {
        help();
        return 0;
      }
      if (a == "--no-patching") {
        o.patching = false;
        continue;
      }
      if (a.rfind("--", 0) != 0) {
        if (!input.empty())
          throw std::runtime_error("Only one input mesh is allowed");
        input = a;
        continue;
      }
      if (i + 1 >= argc)
        throw std::runtime_error("Missing value for " + a);
      std::string value = argv[++i];
      auto number = [&]() {
        std::size_t used = 0;
        double x = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(x))
          throw std::runtime_error("Invalid numeric value for " + a);
        return x;
      };
      auto integer = [&]() {
        double x = number();
        if (x < 0 || x > std::numeric_limits<int>::max() || std::floor(x) != x)
          throw std::runtime_error("Invalid integer for " + a);
        return static_cast<int>(x);
      };
      auto large = [&]() {
        if (value.empty() || value[0] == '-')
          throw std::runtime_error("Invalid unsigned value for " + a);
        std::size_t used = 0;
        auto x = std::stoull(value, &used);
        if (used != value.size())
          throw std::runtime_error("Invalid unsigned value for " + a);
        return x;
      };
      if (a == "--output")
        out = value;
      else if (a == "--sample-ratio")
        o.sample_ratio = number();
      else if (a == "--prune")
        o.prune = number();
      else if (a == "--pair-threshold")
        o.pair_threshold = number();
      else if (a == "--bandwidth")
        o.bandwidth = number();
      else if (a == "--verify-threshold")
        o.verify_threshold = number();
      else if (a == "--smoothing")
        o.smoothing = integer();
      else if (a == "--subdivide")
        o.subdivide = integer();
      else if (a == "--seed")
        o.seed = large();
      else if (a == "--max-iterations")
        o.max_iterations = integer();
      else if (a == "--max-clusters")
        o.max_clusters = integer();
      else if (a == "--max-pairs")
        o.max_pairs = large();
      else if (a == "--max-votes")
        o.max_votes = large();
      else
        throw std::runtime_error("Unknown option: " + a);
    }
    if (input.empty() || out.empty())
      throw std::runtime_error("Input and --output are required");
    if (std::filesystem::exists(out) && (!std::filesystem::is_directory(out) ||
                                         !std::filesystem::is_empty(out)))
      throw std::runtime_error("Output directory must be new or empty "
                               "(existing results are preserved)");
    auto backend = symmetry::make_cpu_backend();
    auto r = symmetry::detect(symmetry::read_mesh(input), o, *backend,
                              [](const std::string &stage, double ms) {
                                std::cout << stage << ": " << ms << " ms\n";
                              });
    symmetry::export_result(r, out);
    std::cout << "vertices=" << r.mesh.vertices.size()
              << " sample=" << r.sample.size() << " pruned=" << r.pruned.size()
              << " pairs=" << r.pairs.size() << " votes=" << r.votes.size()
              << " clusters=" << r.clusters.size() << "\n";
    for (std::size_t i = 0; i < r.clusters.size(); ++i) {
      auto c = r.clusters[i];
      std::cout << i << " " << (c.reflection ? "reflection" : "rigid")
                << " support=" << c.votes.size() << " coverage=" << c.coverage
                << " rms=" << c.rms << "\n";
    }
    if (r.limited_iterations)
      std::cout << "Warning: " << r.limited_iterations
                << " mean-shift trajectories reached the iteration limit. "
                   "Increase --max-iterations if needed.\n";
    std::cout << "Results: " << out << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
