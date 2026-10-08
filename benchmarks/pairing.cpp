// SPDX-License-Identifier: MPL-2.0
// Isolated pairing benchmark on identical host input, including transfers.
#include "symmetry/cuda.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace symmetry;
int main() {
  try {
    constexpr int count = 10000, groups = 100;
    Mesh mesh;
    Features features;
    features.resize(count);
    std::vector<Index> selected;
    for (int i = 0; i < count; ++i) {
      mesh.vertices.push_back({double(i), double(i % 7), double(i % 11)});
      selected.push_back(i);
      double k = double((i * 37) % groups + 1);
      features.kmin[i] = k;
      features.kmax[i] = 2 * k;
    }
    Options options;
    options.pair_threshold = 1e-6;
    options.max_pairs = 600000;
    auto cpu = make_cpu_backend(), gpu = make_cuda_backend();
    auto run = [&](const Backend &backend) {
      return backend.pairing(mesh.as_view(), features.as_view(), view(selected),
                             {1, {}}, options);
    };
    auto reference = run(*cpu),
         candidate = run(*gpu); // Warmup + exact comparison.
    if (reference.a != candidate.a || reference.b != candidate.b ||
        candidate.size() != 495000)
      throw std::runtime_error("Pairing benchmark outputs differ");
    std::cout << "selected=" << count << " groups=" << groups
              << " pairs=" << candidate.size() << '\n';
    for (const Backend *backend : {cpu.get(), gpu.get()}) {
      std::vector<double> times;
      for (int iteration = 0; iteration < 3; ++iteration) {
        auto start = std::chrono::steady_clock::now();
        auto result = run(*backend);
        times.push_back(std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count());
        if (result.a != reference.a || result.b != reference.b)
          throw std::runtime_error("Pairing benchmark repeat differs");
      }
      std::sort(times.begin(), times.end());
      std::cout << backend->name() << ",median_ms=" << times[1] << '\n';
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
