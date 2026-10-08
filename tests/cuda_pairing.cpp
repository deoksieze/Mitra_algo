// SPDX-License-Identifier: MPL-2.0
#include "symmetry/cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace symmetry;
void need(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
struct Fixture {
  Mesh mesh;
  Features features;
  std::vector<Index> selected;
  explicit Fixture(int count) {
    features.resize(count);
    for (int i = 0; i < count; ++i) {
      mesh.vertices.push_back(
          {double(i), double(i % 7) * .1, double(i % 11) * .2});
      selected.push_back(i);
      features.kmin[i] = .5;
      features.kmax[i] = 1;
    }
  }
};
Pairs compare(const Backend &cpu, const Backend &gpu, const Fixture &f,
              double threshold, std::uint64_t budget = 2000000) {
  Options o;
  o.pair_threshold = threshold;
  o.max_pairs = budget;
  auto a = cpu.pairing(f.mesh.as_view(), f.features.as_view(), view(f.selected),
                       {1, {}}, o);
  auto b = gpu.pairing(f.mesh.as_view(), f.features.as_view(), view(f.selected),
                       {1, {}}, o);
  need(a.a == b.a && a.b == b.b, "CPU/GPU pair values or order differ");
  return b;
}
int main() {
  try {
    auto cpu = make_cpu_backend(), gpu = make_cuda_backend();
    Fixture empty(0), one(1);
    need(compare(*cpu, *gpu, empty, .15).size() == 0, "empty selection");
    need(compare(*cpu, *gpu, one, .15).size() == 0, "single selection");
    Fixture dense(513);
    auto pairs = compare(*cpu, *gpu, dense, .15);
    need(pairs.size() == 513ULL * 512 / 2, "dense triangle count");
    // Exact budget succeeds; no random subsampling and no truncation.
    compare(*cpu, *gpu, dense, .15, pairs.size());
    Options o;
    o.max_pairs = pairs.size() - 1;
    bool failed = false;
    try {
      gpu->pairing(dense.mesh.as_view(), dense.features.as_view(),
                   view(dense.selected), {1, {}}, o);
    } catch (const std::runtime_error &e) {
      failed = std::string(e.what()).find("budget") != std::string::npos;
    }
    need(failed, "pair budget was not rejected");
    Fixture random(777);
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> values(-2, 2);
    for (int i = 0; i < 777; ++i) {
      random.features.kmin[i] = values(rng);
      random.features.kmax[i] = values(rng);
      if (i % 13 == 0)
        random.features.kmin[i] = 0;
      if (i % 17 == 0)
        random.features.kmax[i] = 1e-12;
    }
    std::shuffle(random.selected.begin(), random.selected.end(), rng);
    // Source order is an input contract, independent of mesh IDs and sorted
    // keys.
    for (double t : {1e-8, .01, .15, .7, std::nextafter(1., 0.), 1.})
      compare(*cpu, *gpu, random, t);
    random.selected.push_back(
        random.selected[0]); // Duplicate IDs preserve ordinal semantics.
    compare(*cpu, *gpu, random, .7);
    auto repeat = compare(*cpu, *gpu, random, .15);
    o.pair_threshold = .15;
    o.max_pairs = 2000000;
    auto repeat2 =
        gpu->pairing(random.mesh.as_view(), random.features.as_view(),
                     view(random.selected), {1, {}}, o);
    need(repeat.a == repeat2.a && repeat.b == repeat2.b,
         "pairing is not deterministic");
    Fixture boundary(12);
    double tiny = std::nextafter(.75, 0.0);
    double curvatures[12] = {.75, 1,     tiny,   -.75,  -1,    0,
                             -0., 1e-12, -1e-12, 1e-10, 1e200, 1e200};
    for (int i = 0; i < 12; ++i) {
      boundary.features.kmin[i] = curvatures[i];
      boundary.features.kmax[i] = curvatures[i];
    }
    boundary.mesh.vertices.set(0, {0, 0, 0});
    boundary.mesh.vertices.set(1, {.03, 0, 0});
    boundary.mesh.vertices.set(2, {std::nextafter(.03, 0.0), 0, 0});
    for (double t : {.25, .5, 1.})
      compare(*cpu, *gpu, boundary, t);
    // Cases at the magnitude-dependent edge of the conservative search
    // interval.
    for (double t : {.01, .15, .5, .9, std::nextafter(1., 0.)}) {
      Fixture edge(4);
      double target = 1 / (1 - t);
      edge.features.kmin = {1, target, std::nextafter(target, 0.),
                            std::nextafter(target, INFINITY)};
      edge.features.kmax = edge.features.kmin;
      compare(*cpu, *gpu, edge, t);
    }
    // Floating-point subtraction/tolerance can admit a pair beyond the
    // exact-real radius Mi/(1-t). This requires widening t itself, not only
    // nextafter on the final interval endpoints.
    Fixture rounded(2);
    rounded.features.kmin = {1, 13510798882111490.0};
    rounded.features.kmax = rounded.features.kmin;
    need(compare(*cpu, *gpu, rounded, std::nextafter(1., 0.)).size() == 1,
         "rounded near-one threshold pair was missed");
    Fixture zeros(20);
    for (int i = 0; i < 20; ++i) {
      zeros.features.kmin[i] = i % 2 ? -0. : 0.;
      zeros.features.kmax[i] = 0.;
    }
    need(compare(*cpu, *gpu, zeros, .15).size() == 190,
         "signed zero keys differ");
    auto rejects = [&](Fixture &f, ModelContext context, double threshold) {
      Options options;
      options.pair_threshold = threshold;
      try {
        gpu->pairing(f.mesh.as_view(), f.features.as_view(), view(f.selected),
                     context, options);
      } catch (const std::exception &) {
        return;
      }
      throw std::runtime_error("invalid pairing input accepted");
    };
    rejects(dense, {0, {}}, .15);
    rejects(dense, {NAN, {}}, .15);
    rejects(dense, {1, {}}, 0);
    rejects(dense, {1, {}}, NAN);
    rejects(dense, {1, {}}, 1.1);
    dense.selected[0] = -1;
    rejects(dense, {1, {}}, .15);
    dense.selected[0] = 513;
    rejects(dense, {1, {}}, .15);
    dense.selected[0] = 0;
    dense.features.kmin[0] = NAN;
    rejects(dense, {1, {}}, .15);
    // 65,536 selected points would require ~16 GiB for a dense upper-triangle
    // pair-ID table. Disjoint intervals finish with zero output using only
    // linear buffers.
    Fixture sparse(65536);
    for (int i = 0; i < 65536; ++i)
      sparse.features.kmin[i] = sparse.features.kmax[i] = double(i + 1);
    o.pair_threshold = 1e-10;
    o.max_pairs = 1;
    auto start = std::chrono::steady_clock::now();
    auto sparse_pairs =
        gpu->pairing(sparse.mesh.as_view(), sparse.features.as_view(),
                     view(sparse.selected), {1, {}}, o);
    need(sparse_pairs.size() == 0,
         "disjoint signature intervals emitted pairs");
    double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count();
    std::cout << "CUDA pairing passed: exact order, thresholds, dense/sparse "
                 "ranges, budgets, 65536-point sparse case "
              << ms << " ms\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
